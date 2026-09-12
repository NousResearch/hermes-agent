"""Gateway slash commands that rotate, switch, fork or rewrite the session transcript:
/new, /resume, /sessions, /branch, /title, /save, /undo, /retry, /topic, /compress.
Split out of ``gateway/slash_commands.py``; bound onto ``GatewayRunner`` through
``GatewaySlashCommandsMixin``.  Origin internals are imported lazily inside the bodies to avoid
the import cycle."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shlex
from typing import Any, Optional, Union

from agent.i18n import t
from agent.turn_context import extract_api_content_sidecar
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key, is_shared_multi_user_session
from gateway.session_transcript import TranscriptReadError
from gateway.slash_commands_status import HISTORY_UNREADABLE

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py

# Bound on the off-loop agent cleanup during /new; past it the reset proceeds and the teardown
# finishes (or leaks) in its worker thread rather than blocking the event loop.
_RESET_CLEANUP_TIMEOUT_S = 30.0
# chat_type values whose session key is per-user (DM-like), incl. the unknown/blank case.
_DM_CHAT_TYPES = {"dm", "direct", "private", ""}

_BRANCH_COPIED_FIELDS = ("content", "tool_calls", "tool_call_id", "finish_reason", "reasoning",
                         "reasoning_content", "reasoning_details", "codex_reasoning_items",
                         "codex_message_items", "timestamp")


def _sattr(obj, name: str) -> str:
    """``str(getattr(obj, name) or "")`` — normalized identity field for origin comparisons."""
    return str(getattr(obj, name, "") or "")


def _manual_compression_reply_lines(summary: dict, compressor, focus_topic) -> list[str]:
    """Manual /compress confirmation lines, surfacing summariser/aux-model failures.
    ``_last_compress_aborted`` = no usable summary, messages unchanged.  Provider exception text is
    force-redacted at this UI boundary even when global redaction is off; an aux model recovered
    via main is an info note so the user can fix their config."""
    lines = [f"🗜️ {summary['headline']}"]
    if focus_topic:
        lines.append(t("gateway.compress.focus_line", topic=focus_topic))
    lines.append(summary["token_line"])
    if summary["note"]:
        lines.append(summary["note"])
    summary_err = getattr(compressor, "_last_summary_error", None)
    if summary_err:
        from agent.redact import redact_sensitive_text
        summary_err = redact_sensitive_text(summary_err, force=True)
    aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
    if getattr(compressor, "_last_compress_aborted", False):
        lines.append(t("gateway.compress.aborted", error=(summary_err or "unknown error")))
    elif aux_fail_model:
        aux_err = getattr(compressor, "_last_aux_model_failure_error", None) or "unknown error"
        lines.append(t("gateway.compress.aux_failed", model=aux_fail_model, error=aux_err))
    return lines


def _compress_preview_reply(history, partial: bool, keep_last, focus_topic, agg_note: str) -> str:
    """``/compress --preview``: report what WOULD be compressed — no agent, no writes."""
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import summarize_compress_preview

    pv_msgs = [{"role": m.get("role"), "content": m.get("content")} for m in history
               if m.get("role") in {"user", "assistant"} and m.get("content")]
    report = summarize_compress_preview(pv_msgs, partial, keep_last, focus_topic,
                                        estimate_request_tokens_rough(pv_msgs))
    lines = [f"🗜️ {line}" for line in report["lines"]]
    if agg_note:
        lines.append(agg_note)
    return "\n".join(lines)


def _reset_process_scoped_tool_state() -> None:
    """Drop env-passthrough and credential-file state at a conversation boundary (best-effort)."""
    with contextlib.suppress(Exception):
        from tools.env_passthrough import clear_env_passthrough
        clear_env_passthrough()
    with contextlib.suppress(Exception):
        from tools.credential_files import clear_credential_files
        clear_credential_files()


def _branch_row(msg: dict) -> dict:
    """/branch child row; keeps the api_content sidecar so the branch's first turn replays the
    parent's exact wire bytes (warm provider prompt cache) instead of a cold prefill."""
    row = {k: msg.get(k) for k in _BRANCH_COPIED_FIELDS}
    row["role"] = msg.get("role", "user")
    row["tool_name"] = msg.get("tool_name") or msg.get("name")
    row["api_content"] = extract_api_content_sidecar(msg)
    return row


def _strip_resume_name(parts: list[str]) -> str:
    """Join the non-flag /resume tokens; strip literal ``<...>``/``[...]``/quotes typed from the
    usage hint (mirrors the CLI)."""
    name = " ".join(p for p in parts if p not in {"--all", "--cross-room"}).strip()
    if len(name) >= 2 and (name[0], name[-1]) in {("<", ">"), ("[", "]"), ('"', '"'), ("'", "'")}:
        name = name[1:-1].strip()
    return name


class GatewaySessionCommandsMixin:
    """Session-transcript slash commands (/new, /resume, /sessions, /branch, /title, /save, /undo, /retry, /topic, /compress)."""

    # ------------------------------------------------------------------ /new, /reset

    async def _cleanup_old_agent_for_reset(self, session_key: str) -> None:
        """Close the old agent's tool resources (sandboxes, browsers, subprocesses) before eviction.
        Blocking work on the event loop (confirm-button click) → offloaded with a bounded timeout.
        wait_for cancels the await, not the worker thread: a wedged teardown keeps running (or
        leaks); the reset proceeds either way."""
        _old_agent = self._cached_agent_for(session_key)
        if _old_agent is None:
            return
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(self._cleanup_agent_resources, _old_agent),
                timeout=_RESET_CLEANUP_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup for session %s exceeded %ss during /new reset; proceeding with "
                "reset (the worker thread is left to finish on its own). (#35994)",
                session_key, _RESET_CLEANUP_TIMEOUT_S)
        except Exception as cleanup_exc:
            logger.warning(
                "Agent resource cleanup for session %s failed during /new reset: %s (#35994)",
                session_key, cleanup_exc)

    async def _fire_session_reset_hooks(self, source: SessionSource, session_key: str, old_sid,
                                        new_sid) -> None:
        """Session-boundary hooks: plugin finalize (off-loop + bounded — trace exports can block
        arbitrarily), then session:end and session:reset."""
        platform_value = source.platform.value if source.platform else ""
        with contextlib.suppress(Exception):
            await self._finalize_session_off_loop(
                session_id=old_sid, platform=platform_value, reason="new_session",
                old_session_id=old_sid, new_session_id=new_sid)
        hook_payload = {"platform": platform_value, "user_id": source.user_id, "session_key": session_key}
        await self.hooks.emit("session:end", dict(hook_payload))
        await self.hooks.emit("session:reset", dict(hook_payload))

    async def _handle_reset_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /new or /reset command."""
        source = event.source
        session_key = self._session_key_for_source(source)
        self._invalidate_session_run_generation(session_key, reason="session_reset")
        # Evict the running-agent slot now that the generation is bumped: the in-flight run's own
        # guarded release (old generation) returns False and would leave a zombie slot that silently
        # drops all later messages. Idempotent, so the run's finally calling it again is harmless.
        self._release_running_agent_state(session_key)
        # Snapshot the old entry so on_session_finalize can report the expiring session id.
        # Evict the running-agent slot now that the generation is bumped. The in-flight run's own guarded
        # release (run_generation=old) will return False and leave its dead agent behind; clearing here
        # keeps the slot from becoming a zombie that silently drops all later messages (#28686). Idempotent,
        # so the run's finally calling it again is harmless.
        old_entry = self.session_store._entries.get(session_key)
        await self._cleanup_old_agent_for_reset(session_key)
        self._evict_cached_agent(session_key)
        # Conversation boundary: ALL conversation-scoped per-session state + security state in one
        # funnel call (see _CONVERSATION_SCOPED_STATE in gateway/run.py).
        self._clear_conversation_scope(session_key, reason="session_reset")
        # In-flight async delegations end WITH the conversation: once the id rotates their
        # completions have no live owner. Expire by durable id, routing key as legacy fallback.
        with contextlib.suppress(Exception):
            from tools.async_delegation import interrupt_for_session
            interrupt_for_session(session_key=session_key, reason="session_reset",
                                  parent_session_id=str(getattr(old_entry, "session_id", "") or ""))
        _reset_process_scoped_tool_state()

        new_entry = await self.async_session_store.reset_session(session_key)
        _old_sid = old_entry.session_id if old_entry else None
        await self._fire_session_reset_hooks(source, session_key, _old_sid,
                                             new_entry.session_id if new_entry else None)
        # Scoped to the profile serving this source so a multiplexed /new banner reports the
        # profile's model, not the base config's.
        try:
            session_info = await asyncio.to_thread(self._reset_notice_session_info, source)
        except Exception:
            session_info = ""
        if new_entry:
            default_header = t("gateway.reset.header_default")
        else:  # no existing session: create one
            new_entry = await self.async_session_store.get_or_create_session(source, force_new=True)
            default_header = t("gateway.reset.header_new")
        header = await asyncio.to_thread(self._telegram_topic_new_header, source) or default_header
        _title_arg = event.get_command_args().strip()
        if _title_arg and self._session_db and new_entry:
            header = await self._reset_titled_header(header, new_entry.session_id, _title_arg)
        # Telegram DM topic lane: rebind (chat_id, thread_id) → session_id so the next message uses
        # the fresh session instead of switching back to the old one.
        if await asyncio.to_thread(self._is_telegram_topic_lane, source) and new_entry is not None:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, new_entry)
            except Exception:
                logger.debug("Failed to rebind Telegram topic after /new", exc_info=True)
        _new_sid = new_entry.session_id if new_entry else None
        # Plugin on_session_reset hook (new session guaranteed to exist); best-effort.
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook("on_session_reset", session_id=_new_sid, reason="new_session",
                         platform=source.platform.value if source.platform else "",
                         old_session_id=_old_sid, new_session_id=_new_sid)
        except Exception:
            pass
        try:
            from hermes_cli.tips import get_random_tip
            _tip_line = t("gateway.reset.tip", tip=get_random_tip())
        except Exception:
            _tip_line = ""
        body = f"{header}\n\n{session_info}" if session_info else header
        return EphemeralReply(f"{body}{_tip_line}")

    async def _reset_titled_header(self, header: str, session_id: str, title_arg: str) -> str:
        """``/new <title>``: titled header on success, else the header plus a rejection note."""
        from hermes_state import SessionDB
        note = ""
        try:
            sanitized = SessionDB.sanitize_title(title_arg)
        except ValueError as e:
            sanitized = None
            note = t("gateway.reset.title_rejected", error=str(e))
        if sanitized:
            try:
                await self._session_db.set_session_title(session_id, sanitized)
                header = t("gateway.reset.header_titled", title=sanitized)
            except ValueError as e:
                note = t("gateway.reset.title_error_untitled", error=str(e))
            except Exception:
                pass
        elif not note:  # sanitize_title returned empty (whitespace-only / unprintable)
            note = t("gateway.reset.title_empty_untitled")
        return header + note

    # ------------------------------------------------------- origin / ownership guards

    def _gateway_session_origin_for_id(self, session_id: str) -> Optional[SessionSource]:
        """Best-effort origin lookup for gateway session IDs."""
        lookup = getattr(type(self.session_store), "lookup_by_session_id", None)
        if callable(lookup):
            entry = lookup(self.session_store, session_id)
            return getattr(entry, "origin", None) if entry is not None else None
        # Test doubles / older stores lack the public lookup; fail closed when nothing resolves.
        entries = getattr(self.session_store, "_entries", {}) or {}
        return next((getattr(e, "origin", None) for e in entries.values()
                     if getattr(e, "session_id", None) == session_id), None)

    def _same_matrix_room(
        self, current: SessionSource, origin: Optional[SessionSource]
    ) -> bool:
        """Compatibility wrapper around the unified origin-sharing policy."""
        if (
            origin is None
            or current.platform != Platform.MATRIX
            or origin.platform != Platform.MATRIX
        ):
            return False
        return self._same_origin_chat(current, origin)

    def _same_origin_chat(self, current: SessionSource, origin: Optional[SessionSource]) -> bool:
        """Return whether two origins resolve to the same canonical session key.

        Authorization must use the exact routing contract instead of maintaining
        a second approximation. ``_session_key_for_source`` already incorporates
        platform, profile namespace, chat type, room, thread, alternate user id,
        and the configured group/thread participant-sharing policy.
        """
        if origin is None or current is None:
            return False
        # A routing key can still be built for an anonymous sink. It is not
        # ownership proof: DMs need a chat or participant, other lanes need a
        # room and (unless explicitly shared) a participant.
        for candidate in (current, origin):
            chat = _sattr(candidate, "chat_id")
            participant = candidate.user_id_alt or candidate.user_id
            if _sattr(candidate, "chat_type").lower() in _DM_CHAT_TYPES:
                if not (chat or participant):
                    return False
            elif not chat or (not self._is_shared_session_source(candidate) and not participant):
                return False
        try:
            key_for_source = getattr(self, "_session_key_for_source", None)
            if not callable(key_for_source):
                return False
            current_key = str(key_for_source(current) or "")
            origin_key = str(key_for_source(origin) or "")
        except Exception:
            return False
        return bool(current_key) and current_key == origin_key

    def _is_shared_session_source(self, source: SessionSource) -> bool:
        """Whether *source*'s session key is shared by every participant (not per-user); mirrors
        build_session_key's isolation rules so the guards stay in lock-step with the key."""
        return is_shared_multi_user_session(
            source, group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False))

    def _resume_caller_is_admin(self, source: SessionSource) -> bool:
        """Whether *source* is an EXPLICITLY-configured admin (cross-origin /resume, /sessions).
        Stricter than ``SlashAccessPolicy.is_admin()``, which is True for every caller when slash
        gating is DISABLED — the default config would make everyone cross-origin-capable (IDOR)."""
        try:
            from gateway.slash_access import policy_for_source
            policy = policy_for_source(self.config, source)
            uid = getattr(source, "user_id", None)
            return bool(policy.enabled and uid and policy.is_admin(uid))
        except Exception:
            return False

    @staticmethod
    def _decode_persisted_session_source(
        persisted: Any,
    ) -> tuple[dict[str, Any], SessionSource] | None:
        """Decode durable origin metadata with historical null compatibility."""
        try:
            if isinstance(persisted, str):
                if not persisted.strip():
                    return None
                persisted = json.loads(persisted)
            if not isinstance(persisted, dict):
                return None
            payload = dict(persisted)
            required = {"platform", "user_id", "chat_id", "chat_type", "thread_id"}
            if not required.issubset(payload):
                return None
            if not isinstance(payload.get("platform"), str) or not payload["platform"]:
                return None
            if not isinstance(payload.get("chat_type"), str) or not payload["chat_type"]:
                return None
            for key in ("user_id", "chat_id", "thread_id", "user_id_alt", "profile",
                        "scope_id", "guild_id", "prospective_thread_id"):
                if payload.get(key) is not None and not isinstance(payload[key], str):
                    return None
            for key in ("chat_id", "thread_id", "user_id_alt", "profile"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
                    payload[key] = None
            source = SessionSource.from_dict(payload)
            for key in ("chat_id", "thread_id", "user_id_alt", "profile"):
                if key in payload and payload[key] is None and hasattr(source, key):
                    setattr(source, key, None)
            return payload, source
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _persisted_matrix_origin_matches(
        self, current: SessionSource, row: dict[str, Any]
    ) -> bool:
        """Validate a historical Matrix row against the exact caller origin.

        The dedicated session columns and ``origin_json`` jointly form the durable
        authorization provenance. Both representations must agree, and the
        caller must share their canonical key; missing, malformed, partial, or
        contradictory data remains fail-closed. Explicit admin bypass is handled
        by ``_resume_target_allowed`` before this helper runs.
        """
        if not isinstance(row, dict) or current.platform != Platform.MATRIX:
            return False

        row_chat = row.get("chat_id")
        row_thread = row.get("thread_id")
        row_user = row.get("user_id")
        row_chat_type = row.get("chat_type")

        # Require complete, well-typed dedicated provenance. Caller-to-row
        # equality was already proven by the canonical session-key check.
        if row_thread is not None and not isinstance(row_thread, str):
            return False
        if (
            str(row.get("source") or "") != Platform.MATRIX.value
            or not isinstance(row_chat, str)
            or not row_chat
            or "thread_id" not in row
            or not isinstance(row_user, str)
            or not row_user
            or not isinstance(row_chat_type, str)
            or not row_chat_type
        ):
            return False

        decoded = self._decode_persisted_session_source(row.get("origin_json"))
        if decoded is None:
            return False
        persisted, origin = decoded
        if "thread_id" not in persisted:
            return False
        persisted_chat = persisted.get("chat_id")
        persisted_thread = persisted.get("thread_id")
        persisted_user = persisted.get("user_id")
        if persisted_thread is not None and not isinstance(persisted_thread, str):
            return False
        if (
            persisted.get("platform") != Platform.MATRIX.value
            or not isinstance(persisted_chat, str)
            or not persisted_chat
            or not isinstance(persisted_user, str)
            or not persisted_user
        ):
            return False

        try:
            key_for_source = getattr(self, "_session_key_for_source", None)
            if not callable(key_for_source):
                return False
            origin_session_key = str(key_for_source(origin) or "")
        except Exception:
            return False
        persisted_session_key = row.get("session_key")

        # Full origin metadata must corroborate every DB field, and the stored
        # canonical key must be the key actually produced by that persisted
        # origin. Caller-to-key equality was already enforced by
        # ``_resume_target_allowed``.
        return bool(
            origin.platform == Platform.MATRIX
            and origin.chat_id == row_chat
            and str(origin.thread_id or "") == str(row_thread or "")
            and origin.user_id == row_user
            and origin.chat_type == row_chat_type
            and isinstance(persisted_session_key, str)
            and origin_session_key == persisted_session_key
        )

    async def _legacy_telegram_session_source(
        self, current: SessionSource, row: dict[str, Any]
    ) -> Optional[SessionSource]:
        """Recover only ordinary SQL-NULL Telegram origins owned by the default DB.

        The caller and routing key may corroborate stored facts, never supply
        missing ones. Named/unknown stores cannot prove the lost profile metadata.
        """
        import re

        if (
            row.get("source") != "telegram"
            or current.platform != Platform.TELEGRAM
            or row.get("chat_type") not in ("dm", "group")
            or row.get("chat_type") != current.chat_type
            or row.get("profile_name") != "default"
            or current.profile not in (None, "", "default")
            or current.profile_route_rejected
            or getattr(self.config, "multiplex_profiles", None) is not False
            or "thread_id" not in row
            or row["thread_id"] not in (None, "")
            or current.thread_id not in (None, "")
        ):
            return None
        for field in ("user_id_alt", "chat_id_alt", "scope_id", "guild_id",
                      "prospective_thread_id", "parent_chat_id"):
            if getattr(current, field, None) not in (None, "") or row.get(field) not in (None, ""):
                return None
        chat_id, user_id = row.get("chat_id"), row.get("user_id")
        if (
            not isinstance(chat_id, str) or re.fullmatch(r"-?[0-9]+", chat_id) is None
            or not isinstance(user_id, str) or re.fullmatch(r"[0-9]+", user_id) is None
            or user_id != current.user_id
        ):
            return None
        origin = SessionSource(
            platform=Platform(row["source"]), chat_id=chat_id, user_id=user_id,
            chat_type=row["chat_type"], thread_id=row["thread_id"], profile=row["profile_name"],
        )
        if self._is_shared_session_source(origin):
            return None
        try:
            # Native DB owner derives from the actual db_path, not the active
            # profile, caller, row label, or a parsed agent:main namespace.
            if await self._session_db._own_profile_name() != "default":
                return None
        except Exception:
            return None
        return origin

    async def _resume_target_allowed(
        self,
        source: SessionSource,
        target_id: str,
        allow_override: bool = False,
        persisted_row: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Whether *source* may see or resume persisted session *target_id*.

        This is the single authorization policy for direct ``/resume`` and every
        listing surface. It prefers a live routing
        origin, then validates durable provenance for historical sessions. Missing
        or malformed provenance fails closed. Only an explicitly configured admin
        using an override-capable command form may bypass origin scoping.
        """
        if allow_override and self._resume_caller_is_admin(source):
            return True
        # Use the live origin only when it resolves to a real SessionSource; a
        # store that can't resolve it (or an unexpected lookup error) falls back
        # to the durable DB provenance below.
        try:
            origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            origin = None
        if isinstance(origin, SessionSource):
            return self._same_origin_chat(source, origin)

        row = persisted_row
        if row is None:
            try:
                row = await self._session_db.get_session(target_id) or {}
            except Exception:
                return False
        if not isinstance(row, dict):
            return False

        try:
            key_for_source = getattr(self, "_session_key_for_source", None)
            if not callable(key_for_source):
                return False
            caller_session_key = str(key_for_source(source) or "")
        except Exception:
            return False
        persisted_session_key = row.get("session_key")
        if (
            not caller_session_key
            or not isinstance(persisted_session_key, str)
            or not persisted_session_key
            or persisted_session_key != caller_session_key
        ):
            return False

        if source.platform == Platform.MATRIX:
            return self._persisted_matrix_origin_matches(source, row)

        if "origin_json" in row and row["origin_json"] is None:
            origin_source = await self._legacy_telegram_session_source(source, row)
            if origin_source is None:
                return False
        else:
            decoded = self._decode_persisted_session_source(row.get("origin_json"))
            if decoded is None:
                return False
            _payload, origin_source = decoded
        try:
            origin_session_key = str(key_for_source(origin_source) or "")
        except Exception:
            return False

        # The persisted full origin must corroborate the dedicated row fields.
        # Caller aliases (for example WhatsApp JID/LID flips) are deliberately
        # not compared as raw strings after the canonical session-key match.
        row_source = str(row.get("source") or "")
        origin_platform = (
            origin_source.platform.value
            if isinstance(origin_source.platform, Platform)
            else str(origin_source.platform or "")
        )
        if not row_source or origin_platform != row_source:
            return False
        if str(row.get("user_id") or "") != str(origin_source.user_id or ""):
            return False
        if str(row.get("chat_id") or "") != str(origin_source.chat_id or ""):
            return False
        if str(row.get("chat_type") or "") != str(origin_source.chat_type or ""):
            return False
        if str(row.get("thread_id") or "") != str(origin_source.thread_id or ""):
            return False
        return (bool(origin_session_key) and origin_session_key == persisted_session_key
                and self._same_origin_chat(source, origin_source))

    async def _resume_row_visible(
        self, source: SessionSource, row: dict, allow_all: bool
    ) -> bool:
        """Whether a listing row belongs to the caller's authorized origin.

        Delegate to the exact same policy as direct ``/resume`` so full listings,
        numbered choices, and direct IDs/titles cannot drift into different
        authorization rules.
        """
        sid = str(row.get("id") or "")
        lineage_root_id = str(row.get("_lineage_root_id") or "")
        if lineage_root_id and lineage_root_id != sid:
            session_db = getattr(self, "_session_db", None)
            if session_db is None:
                return False
            for actual_id in (lineage_root_id, sid):
                try:
                    actual_row = await session_db.get_session(actual_id) or {}
                except Exception:
                    return False
                if not isinstance(actual_row, dict) or not actual_row:
                    return False
                if not await self._resume_target_allowed(
                    source,
                    actual_id,
                    allow_override=allow_all,
                    persisted_row=actual_row,
                ):
                    return False
            return True
        return await self._resume_target_allowed(
            source,
            sid,
            allow_override=allow_all,
            persisted_row=row,
        )

    # ------------------------------------------------------------------ /retry, /undo

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        # The canonical projection skips bookkeeping rows (role=user + display_kind) and pure
        # handoffs while still recognizing a real ask embedded in a compaction carrier.
        from agent.context_compressor import (
            history_before_user_originated_turn, retryable_user_text, split_user_originated_turn,
            user_originated_turn_view)

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        last_user_idx = next((i for i in range(len(history) - 1, -1, -1)
                              if user_originated_turn_view(history[i]) is not None), None)
        if last_user_idx is None:
            return t("gateway.retry.no_previous")
        # Resolve text + scaffold-preserving prefix BEFORE any write; messaging retries cannot
        # reconstruct attachments, so media/unknown content is rejected with the session untouched.
        try:
            truncated, live_view = history_before_user_originated_turn(history, last_user_idx)
            last_user_msg = retryable_user_text(live_view.get("content"))
            handoff, _ = split_user_originated_turn(history[last_user_idx])
        except ValueError as exc:
            return f"Cannot retry that message safely: {exc}"

        if handoff is not None:
            # Composite carrier (one row = retained summary + live ask): the carrier-aware rewind
            # archives it and inserts the pure scaffold atomically, reselecting the latest carrier
            # on the same snapshot so a concurrent newer turn is never removed for stale text.
            try:
                # Plain turns keep the existing rewrite path below; #84078 owns its separate
                # archive_dropped/prefix-CAS semantics.
                rewind_result = await self.async_session_store.rewind_session(
                    session_entry.session_id, 1, require_retryable_composite=True)
            except ValueError as exc:
                return f"Cannot retry that message safely: {exc}"
            if rewind_result is None:
                return "Retry failed; transcript was not changed."
            last_user_msg = rewind_result["target_text"]
        # active_only preserves the active=0/compacted=1 archive left by in-place compaction.
        elif not await self.async_session_store.rewrite_transcript(
            session_entry.session_id, truncated, active_only=True, reject_active_turn_lease=True):
            return "Retry failed; transcript was not changed."
        session_entry.last_prompt_tokens = 0  # transcript was truncated
        return await self._handle_message(MessageEvent(
            text=last_user_msg, message_type=MessageType.TEXT, source=source,
            raw_message=event.raw_message, channel_prompt=event.channel_prompt))

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo [N] — back up N user turns (default 1), soft-deleting the truncated rows and
        echoing the backed-up text; evicts the cached agent so the next turn rebuilds from the
        active-only transcript."""
        source = event.source
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = max(1, int(raw_args.split()[0]))
            except ValueError:
                return t("gateway.undo.invalid_count", arg=raw_args.split()[0])
        session_entry = await self.async_session_store.get_or_create_session(source)
        result = await self.async_session_store.rewind_session(session_entry.session_id, n)
        if result is None:
            return t("gateway.undo.nothing")
        session_entry.last_prompt_tokens = 0  # transcript was truncated
        try:
            # The cache is keyed by the profile-namespaced key; a bare build_session_key(source)
            # yields ``agent:main:…`` and misses for every secondary profile.
            self._evict_cached_agent(self._session_key_for_source(source))
        except Exception as e:
            logger.debug("undo: cached-agent eviction skipped: %s", e)
        target_text = result["target_text"]
        preview = target_text[:200] + "..." if len(target_text) > 200 else target_text
        return t("gateway.undo.removed", turns=result["turns_undone"],
                 count=result["rewound_count"], preview=preview)

    # --------------------------------------------------------------------- /compress

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Profile-scoping wrapper around manual /compress: multiplexed gateways resolve credentials
        through the fail-closed per-profile secret scope, which slash dispatch (unlike ``_run_agent``)
        does not install — unscoped, /compress would raise ``UnscopedSecretError``."""
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._handle_compress_command_inner(event)
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(self._resolve_profile_home_for_source(event.source)):
            return await self._handle_compress_command_inner(event)

    async def _compress_codex_app_server_session(self, session_key: str, session_id: str) -> str:
        """Manual /compress for codex_app_server sessions: compacts the LIVE cached agent's
        app-server thread (``force=True`` bypasses the ``codex_app_server_auto`` gate) and keeps it
        cached. A temporary agent or a mirror rewrite cannot shrink the server-side thread.

        See #73503.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        agent = self._cached_agent_for(session_key, lockless_fallback=True)
        if agent is None or agent is _AGENT_PENDING_SENTINEL or getattr(agent, "_codex_session", None) is None:
            return (
                "🗜️ Nothing to compact: this session runs on the Codex app-server runtime, whose "
                "context lives in a Codex-owned thread that only exists while the agent is active. "
                "Send a message first, then /compress — or /reset to start fresh.")
        compressor = getattr(agent, "context_compressor", None)
        count_before = getattr(compressor, "compression_count", 0)
        try:
            await self._run_in_executor_with_context(lambda: agent._compress_context([], "", force=True))
        except Exception as exc:
            return t("gateway.compress.failed", error=exc)
        if getattr(compressor, "compression_count", 0) > count_before:
            return (
                "🗜️ Codex app-server thread compacted (thread/compact). The transcript mirror is "
                "unchanged by design — the app-server now carries the compacted context.")
        return (
            "⚠️ Codex app-server compaction did not complete — the thread is unchanged. Check the "
            "app-server logs, retry /compress, or /reset for a clean session.")

    async def _handle_compress_command_inner(self, event: MessageEvent) -> str:
        """Handle /compress -- manually compress conversation context; ``/compress <focus>`` tells
        the summariser what to preserve."""
        from hermes_cli.partial_compress import extract_compress_flags, parse_partial_compress_args

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")
        # Flags are stripped before positional parsing so they coexist with the boundary-aware
        # "here [N]" (partial) and focus-topic (full) forms.
        _raw_args, _preview, _aggressive = extract_compress_flags((event.get_command_args() or "").strip())
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)
        _agg_note = ""
        if _aggressive:
            # LLM-free hard truncation would need its own persistence branch outside the guarded
            # _compress_context rotation machinery — unsupported on this surface.
            _agg_note = t("gateway.compress.aggressive_unsupported")
            if not _preview:
                return _agg_note
        if _preview:
            return _compress_preview_reply(history, partial, keep_last, focus_topic, _agg_note)
        try:
            return await self._run_manual_compression(source, session_entry, history, partial,
                                                      keep_last, focus_topic)
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return t("gateway.compress.failed", error=e)

    async def _run_manual_compression(self, source, session_entry, history: list, partial: bool,
                                      keep_last, focus_topic) -> str:
        """Build a temporary agent, compress the transcript, persist, and describe the outcome."""
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough
        from gateway.run import _platform_config_key
        from hermes_cli.partial_compress import (rejoin_compressed_head_and_tail,
                                                 split_history_for_partial_compress)

        session_key = self._session_key_for_source(source)
        # Platform + stable gateway session key bind this agent (for external context engines) to
        # the original conversation, not a default "cli" host.
        platform_key = _platform_config_key(source.platform) if source.platform else None
        model, runtime_kwargs = self._resolve_session_agent_runtime(source=source, session_key=session_key)
        if str(runtime_kwargs.get("api_mode") or "").lower() == "codex_app_server":
            # Context lives in the server-side thread of the LIVE cached agent; a temporary agent
            # has none (and finally-eviction would destroy the real context).
            return await self._compress_codex_app_server_session(session_key, session_entry.session_id)
        if not runtime_kwargs.get("api_key"):
            return t("gateway.compress.no_provider")
        # FULL transcript (tool results included), like auto-compress: user/assistant-only starves
        # tool-result pruning and can trip the protect-first/last early-return.
        msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
        # Partial: only the head is summarized; the tail snaps to a user-turn start so role
        # alternation holds after rejoin.
        head, tail = msgs, []
        if partial:
            head, tail = split_history_for_partial_compress(msgs, keep_last)
            if not tail:  # degenerate split — fall back to full compression
                partial, head = False, msgs
        # Assign, not setdefault (a resolver value would be a stale placeholder); platform only when
        # known so None -> "cli" holds.
        if platform_key is not None:
            runtime_kwargs["platform"] = platform_key
        runtime_kwargs["gateway_session_key"] = session_key

        tmp_agent = await self._build_manual_compression_agent(session_entry.session_id, model, runtime_kwargs)
        try:
            # Estimate with system prompt + tool schemas (real request pressure); needs the built agent.
            # Must be computed after tmp_agent is built so _cached_system_prompt/tools are populated. See
            # #6217.
            _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
            _tools = getattr(tmp_agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(msgs, system_prompt=_sys_prompt, tools=_tools)
            compressor = tmp_agent.context_compressor
            if not compressor.has_content_to_compress(head):
                return t("gateway.compress.nothing_to_do")
            # Not a bare run_in_executor: the profile secret scope is a contextvar the default
            # executor hop would drop, failing aux-client credential resolution closed.
            compressed, _ = await self._run_in_executor_with_context(
                lambda: tmp_agent._compress_context(
                    head, "", approx_tokens=approx_tokens, focus_topic=focus_topic, force=True,
                    defer_context_engine_notification=True))
            # A held compression lock returns unchanged; say so instead of the misleading no-op text.
            _lock_skipped = getattr(tmp_agent, "_compression_skipped_due_to_lock", None)
            if _lock_skipped is True or isinstance(_lock_skipped, str):
                from agent.manual_compression_feedback import describe_compression_lock_skip
                return describe_compression_lock_skip(_lock_skipped)
            if partial and tail:
                compressed = rejoin_compressed_head_and_tail(compressed, tail)
            await self._persist_manual_compression(tmp_agent, session_entry, source, compressed)
            finalize_context_engine_compression_notification(tmp_agent, committed=True)
            new_tokens = estimate_request_tokens_rough(compressed, system_prompt=_sys_prompt, tools=_tools)
            summary = summarize_manual_compression(msgs, compressed, approx_tokens, new_tokens,
                                                   compression_state=compressor)
        finally:
            finalize_context_engine_compression_notification(tmp_agent, committed=False)
            self._evict_cached_agent(session_key)  # next turn rebuilds the prompt from current files
            # Off-loop + bounded: teardown can block on subprocess/network/SQLite.
            await self._cleanup_agent_resources_off_loop(tmp_agent, context="manual compression")
        return "\n".join(_manual_compression_reply_lines(summary, compressor, focus_topic))

    async def _build_manual_compression_agent(self, session_id: str, model, runtime_kwargs: dict):
        """Build the throwaway AIAgent that performs a manual /compress rewrite of *session_id*."""
        from run_agent import AIAgent
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        # _compress_context may persist its cached system prompt, and this agent runs outside the
        # live session's prompt environment — restore the exact live prompt so provider blocks stay.
        session_row = None
        get_session = getattr(self._session_db, "get_session", None)
        if callable(get_session):
            try:
                session_row = await get_session(session_id)
            except Exception as exc:
                logger.warning(
                    "Manual compression could not restore the system prompt for session %s: %s. "
                    "Preserving an empty prompt so the live turn rebuilds it with its configured "
                    "providers.", session_id, exc, exc_info=True)

        # compression.checkpoint_required needs the memory provider loaded so _compress_context()
        # can write the pre-compression checkpoint; otherwise keep the fast path (no provider init).
        _checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"),
            default=False)
        tmp_agent = AIAgent(**runtime_kwargs, model=model, max_iterations=4, quiet_mode=True,
                            skip_memory=not _checkpoint_required, enabled_toolsets=["memory"],
                            session_id=session_id,
                            session_db=getattr(self._session_db, "_db", self._session_db))
        _seed_hygiene_system_prompt(tmp_agent, session_row)
        # Real platform during construction (context engines bind correctly); afterwards a prompt
        # rebuilt by compression is stamped as the provider-less fallback, stale for the next turn.
        tmp_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        tmp_agent._print_fn = lambda *a, **kw: None
        # close() must not end the rotated session the gateway entry now points at.
        tmp_agent._end_session_on_close = False
        return tmp_agent

    async def _persist_manual_compression(self, tmp_agent, session_entry, source, compressed) -> None:
        """Commit a manual /compress result to the session store.  Rotation (new continuation id)
        writes the compressed messages into the NEW session so the original stays searchable;
        persist BEFORE repointing so a failed write is fatal and old history stays reachable.
        In-place compaction already archived + inserted rows, and a rewrite would DELETE the
        archive; an unchanged id without in-place means rotation FAILED."""
        new_session_id = tmp_agent.session_id
        if new_session_id != session_entry.session_id:
            if not await self.async_session_store.rewrite_transcript(new_session_id, compressed):
                raise RuntimeError(
                    f"failed to persist compressed transcript for session {new_session_id}")
            session_entry.session_id = new_session_id
            await self.async_session_store._save()
            await asyncio.to_thread(self._sync_telegram_topic_binding, source, session_entry,
                                    reason="compress-command")
        elif not getattr(tmp_agent, "_last_compaction_in_place", False):
            logger.warning(
                "Manual /compress: session rotation did not occur (session_id unchanged) and in-place "
                "mode is off — preserving original transcript instead of overwriting it (#44794).")
        await self.async_session_store.update_session(session_entry.session_key, last_prompt_tokens=0)

    # ------------------------------------------------------------------------ /topic

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return t("gateway.topic.not_telegram_dm")
        if not self._session_db:
            return self._session_db_unavailable_reply()

        # Defense in depth: /topic mutates SQLite side tables, so re-check the allowlist here.
        try:
            if not self._is_user_authorized_for_source(source):
                return t("gateway.topic.unauthorized")
        except Exception:
            logger.debug("Topic auth check failed", exc_info=True)

        args = event.get_command_args().strip()
        if args.lower() in {"help", "?", "-h", "--help"}:
            return self._telegram_topic_help_text()
        if args.lower() in {"off", "disable", "stop"}:
            return await self._disable_telegram_topic_mode_for_chat(source)
        if args:
            if not source.thread_id:
                return t("gateway.topic.restore_needs_topic")
            return await self._restore_telegram_topic_session(event, args)

        capabilities = await self._get_telegram_topic_capabilities(source)
        if capabilities.get("checked"):
            blocked_key = None
            if capabilities.get("has_topics_enabled") is False:
                blocked_key = "gateway.topic.topics_disabled"
            elif capabilities.get("allows_users_to_create_topics") is False:
                blocked_key = "gateway.topic.topics_user_disallowed"
            if blocked_key:  # debounced BotFather screenshot
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t(blocked_key)

        profile_name = self._telegram_topic_profile_name(source)
        try:
            await self._session_db.enable_telegram_topic_mode(
                chat_id=str(source.chat_id), user_id=str(source.user_id), profile_name=profile_name,
                has_topics_enabled=capabilities.get("has_topics_enabled"),
                allows_users_to_create_topics=capabilities.get("allows_users_to_create_topics"))
        except Exception as exc:
            logger.exception("Failed to enable Telegram topic mode")
            return t("gateway.topic.enable_failed", error=exc)

        if not source.thread_id:
            await self._ensure_telegram_system_topic(source)
            return await self._telegram_topic_root_status_message(source)
        try:
            binding = await self._session_db.get_telegram_topic_binding(
                chat_id=str(source.chat_id), thread_id=str(source.thread_id),
                profile_name=profile_name)
        except Exception:
            logger.debug("Failed to read Telegram topic binding", exc_info=True)
            binding = None
        if not binding:
            return t("gateway.topic.thread_ready")
        session_id = str(binding.get("session_id") or "")
        try:
            title = await self._session_db.get_session_title(session_id)
        except Exception:
            title = None
        return t("gateway.topic.bound_status", label=title or t("gateway.topic.untitled_session"),
                 session_id=session_id)

    # ------------------------------------------------------------------ /save, /title

    async def _handle_save_command(self, event: MessageEvent) -> str:
        """Handle /save — export the current session and send it as a document."""
        import tempfile
        from hermes_cli.session_export import (
            SAVE_USAGE, default_save_filename, normalize_save_format, render_session_for_save)

        parts = event.get_command_args().split()
        redact = bool(parts) and parts[-1].lower() in ("redact", "--redact")
        if redact:
            parts = parts[:-1]
        if not parts:
            return SAVE_USAGE
        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            return f"{e}\n\n{SAVE_USAGE}"

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id
        if not self._session_db:
            return "Session database not available."
        # Never trust path separators from chat input; the filename is only echoed to the platform.
        filename = parts[1] if len(parts) > 1 else default_save_filename(session_id, fmt)
        filename = os.path.basename(filename) or default_save_filename(session_id, fmt)
        export_data = await self._session_db.export_session(session_id)
        if not export_data:
            return f"No stored messages found for this session ({session_id})."
        if redact:
            from hermes_cli.session_export_md import redact_session_data
            export_data = redact_session_data(export_data)
        temp_dir = tempfile.mkdtemp(prefix="hermes_save_")
        temp_path = os.path.join(temp_dir, filename)
        try:
            # Off-loop: render + write scale with transcript size (multi-MB) and would stall the loop.
            def _render_and_write() -> None:
                rendered = render_session_for_save(export_data, fmt)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(rendered)

            await asyncio.to_thread(_render_and_write)
            adapter = self.get_adapter(source.platform)
            if not adapter:
                return "Platform adapter not found to send the document."
            await adapter.send_document(chat_id=source.chat_id, file_path=temp_path,
                                        caption=f"Session export: {filename}", file_name=filename)
            return "Export complete."
        except Exception as e:
            logger.warning("Session /save failed: %s", e)
            return f"Error exporting session: {e}"
        finally:
            with contextlib.suppress(Exception):
                os.remove(temp_path)
                os.rmdir(temp_dir)

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id
        if not self._session_db:
            return self._session_db_unavailable_reply()

        # The session may only exist in session_store so far (first command in a new session).
        # The messaging origin is persisted so a later /resume of this titled-but-inactive session
        # can prove it belongs to the caller's chat/thread (IDOR scoping).
        if await self._session_db.get_session_title(session_id) is None:
            with contextlib.suppress(Exception):  # may already exist
                await self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id, chat_id=source.chat_id, chat_type=source.chat_type,
                    thread_id=source.thread_id)
        title_arg = event.get_command_args().strip()
        if not title_arg:
            title = await self._session_db.get_session_title(session_id)
            if title:
                return t("gateway.title.current_with_title", session_id=session_id, title=title)
            return t("gateway.title.current_no_title", session_id=session_id)
        try:
            from hermes_state import SessionDB
            sanitized = SessionDB.sanitize_title(title_arg)
        except ValueError as e:
            return t("gateway.shared.warn_passthrough", error=e)
        if not sanitized:
            return t("gateway.title.empty_after_clean")
        try:
            if not await self._session_db.set_session_title(session_id, sanitized):
                return t("gateway.title.not_found")
        except ValueError as e:
            return t("gateway.shared.warn_passthrough", error=e)
        # Mirror the title onto the Telegram forum topic name (auto titles already do this).
        try:
            await asyncio.to_thread(self._schedule_telegram_topic_title_rename, source, session_id, sanitized)
        except Exception:
            logger.debug("Failed to rename Telegram topic from /title", exc_info=True)
        return t("gateway.title.set_to", title=sanitized)

    # -------------------------------------------------------------- /resume, /sessions

    async def _list_titled_sessions(self, source, session_key: str, allow_all: bool) -> list[dict]:
        """Titled sessions in the same order for bare /resume and numeric selection."""
        return await self._list_visible_sessions(source, session_key, allow_all=allow_all)

    async def _list_visible_sessions(
        self, source, session_key: str, *, allow_all: bool = False,
        include_all_sources: bool = False, include_unnamed: bool = False,
        current_session_id: str | None = None, search_query: str | None = None,
        exclude_sources: list[str] | None = None,
    ) -> list[dict]:
        """Fill a gateway page only after canonical root/tip authorization.

        SQL still narrows the lane before LIMIT. Post-filtered rows must not
        hide older authorized candidates, so advance the raw offset until the
        visible page is full or the database is exhausted.
        """
        widen = allow_all and self._resume_caller_is_admin(source)
        search = (search_query or "").strip()
        visible: list[dict] = []
        offset = 0
        page_size = 50
        while len(visible) < 10:
            rows = await self._session_db.list_sessions_rich(
                source=None if widen and include_all_sources else (
                    source.platform.value if source.platform else None),
                session_key=None if widen else session_key,
                limit=page_size, offset=offset, exclude_sources=exclude_sources,
                search_query=search or None, order_by_last_active=bool(search),
            )
            if not rows:
                break
            offset += len(rows)
            for row in rows:
                is_current = bool(current_session_id and row.get("id") == current_session_id)
                if not include_unnamed and not row.get("title") and not search and not is_current:
                    continue
                if not await self._resume_row_visible(source, row, allow_all=widen):
                    continue
                visible.append({**row, "is_current_session": True} if is_current else row)
                if len(visible) >= 10:
                    break
            if len(rows) < page_size:
                break
        return visible

    def _decorate_matrix_session_row(self, row: dict) -> dict:
        """Label an already-authorized Matrix row without mutating the query result."""
        decorated = dict(row)
        title = row.get("title")
        if not isinstance(title, str) or not title or row.get("source") not in (None, "matrix"):
            return decorated
        try:
            origin = self._gateway_session_origin_for_id(str(row.get("id") or ""))
        except Exception:
            origin = None
        if not isinstance(origin, SessionSource):
            decoded = self._decode_persisted_session_source(row.get("origin_json"))
            origin = decoded[1] if decoded else None
        candidates = []
        if isinstance(origin, SessionSource) and origin.platform == Platform.MATRIX:
            candidates.extend((origin.chat_name, origin.chat_id))
        candidates.extend((row.get("display_name"), row.get("chat_id")))
        room = next((value.strip() for value in candidates
                     if isinstance(value, str) and value.strip()), "")
        if room:
            decorated["title"] = f"{title} — {room}"
        return decorated

    async def _resolve_visible_title_target(self, source, title: str, allow_override: bool):
        """Prefer an authorized title candidate before widening for a configured admin."""
        candidates = await self._session_db.list_session_title_candidates(title)
        override_modes = [False, True] if allow_override and self._resume_caller_is_admin(source) else [False]
        for override in override_modes:
            for candidate in candidates:
                candidate_id = str(candidate.get("id") or "")
                if not candidate_id:
                    continue
                try:
                    tip_id = await self._session_db.resolve_resume_session_id(candidate_id)
                except Exception:
                    tip_id = candidate_id
                root_allowed = await self._resume_target_allowed(
                    source, candidate_id, allow_override=override, persisted_row=candidate)
                tip_allowed = root_allowed if tip_id == candidate_id else await self._resume_target_allowed(
                    source, tip_id, allow_override=override)
                if root_allowed and tip_allowed:
                    return candidate_id
        return None

    async def _resolve_resume_target(self, source, session_key: str, name: str, allow_all: bool,
                                     allow_cross_room: bool = False):
        """Resolve and authorize the requested root and compression tip before any switch."""
        override_requested = allow_all or allow_cross_room
        if name.isdigit():
            try:
                titled = await self._list_titled_sessions(source, session_key, allow_all)
            except Exception as e:
                logger.debug("Failed to list titled sessions for numeric resume: %s", e)
                return t("gateway.resume.list_failed", error=e)
            index = int(name)
            if index < 1 or index > len(titled):
                return t("gateway.resume.out_of_range", index=index)
            target = titled[index - 1]
            target_id = target.get("_lineage_root_id") or target.get("id")
            name = target.get("title") or name
        else:
            session = await self._session_db.get_session(name)
            target_id = session["id"] if session else await self._resolve_visible_title_target(
                source, name, override_requested)
        if not target_id:
            return t("gateway.resume.not_found", name=name)
        requested_target_id = str(target_id)
        resolved_target_id = requested_target_id
        try:
            resolved_target_id = await self._session_db.resolve_resume_session_id(requested_target_id)
        except Exception as e:
            logger.debug("Failed to resolve resume continuation for %s: %s", requested_target_id, e)
        root_same_origin = await self._resume_target_allowed(source, requested_target_id)
        tip_same_origin = root_same_origin if resolved_target_id == requested_target_id else (
            await self._resume_target_allowed(source, resolved_target_id))
        same_origin_allowed = root_same_origin and tip_same_origin
        target_allowed = same_origin_allowed
        if not target_allowed and override_requested:
            root_override = await self._resume_target_allowed(source, requested_target_id, allow_override=True)
            tip_override = root_override if resolved_target_id == requested_target_id else (
                await self._resume_target_allowed(source, resolved_target_id, allow_override=True))
            target_allowed = root_override and tip_override
        if not target_allowed:
            return t("gateway.resume.not_found", name=name)
        return resolved_target_id, name, same_origin_allowed

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — list or switch to a previous session."""
        if not self._session_db:
            return self._session_db_unavailable_reply()
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        try:
            parts = shlex.split(event.get_command_args().strip())
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        allow_all = "--all" in parts
        allow_cross_room = "--cross-room" in parts
        name = _strip_resume_name(parts)
        if not name:
            try:
                titled = await self._list_titled_sessions(source, session_key, allow_all)
                return self._resume_listing_reply(source, titled, allow_all)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return t("gateway.resume.list_failed", error=e)

        resolved = await self._resolve_resume_target(source, session_key, name, allow_all, allow_cross_room)
        if isinstance(resolved, str):
            return resolved
        target_id, name, same_origin_allowed = resolved
        current_entry = await self.async_session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return t("gateway.resume.already_on", name=name)
        self._release_running_agent_state(session_key)
        new_entry = await self.async_session_store.switch_session(session_key, target_id)
        if not new_entry:
            return t("gateway.resume.switch_failed")
        # Conversation boundary: all conversation-scoped state + security state in one funnel call.
        # Conversation boundary: clear ALL conversation-scoped per-session state (model/reasoning overrides
        # #10702, one-turn restores, model notes, last-resolved cache #58403, /queue overflow) + security
        # state in one funnel call. See _CONVERSATION_SCOPED_STATE in gateway/run.py.
        self._clear_conversation_scope(session_key, reason="resume")
        # Evict so the next turn rebuilds with the right session_id — the cached AIAgent's memory
        # provider cached _session_id at initialize() and would keep writing to the wrong session.
        self._evict_cached_agent(session_key)
        title = await self._session_db.get_session_title(target_id) or name
        try:
            history = await self.async_session_store.load_transcript(target_id)
        except TranscriptReadError:
            # The resume itself succeeded; only the count is missing — say so rather than "empty".
            return t("gateway.resume.resumed_no_count", title=title) + "\n" + HISTORY_UNREADABLE
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        if source.platform == Platform.MATRIX and allow_cross_room and not same_origin_allowed:
            msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""
            return t("gateway.resume.matrix_cross_room_success", title=title,
                     room=source.chat_name or source.chat_id, msg_part=msg_part)
        if not msg_count:
            return t("gateway.resume.resumed_no_count", title=title)
        if msg_count == 1:
            return t("gateway.resume.resumed_one", title=title, count=msg_count)
        return t("gateway.resume.resumed_many", title=title, count=msg_count)

    def _resume_listing_reply(self, source, titled: list[dict], allow_all: bool) -> str:
        """Numbered /resume list; a non-admin ``--all`` falls back to same-origin scoping and says so
        (sibling of the /sessions notice)."""
        scope_note = None
        if allow_all and not self._resume_caller_is_admin(source):
            scope_note = t("gateway.resume.all_requires_admin")
        if not titled:
            if source.platform == Platform.MATRIX and not allow_all:
                return t("gateway.resume.matrix_no_named_sessions")
            base = t("gateway.resume.no_named_sessions")
            return f"{base}\n{scope_note}" if scope_note else base
        lines = [t("gateway.resume.list_header")]
        for idx, s in enumerate(titled[:10], start=1):
            title = s["title"]
            if source.platform == Platform.MATRIX and allow_all:
                title = self._decorate_matrix_session_row(s)["title"]
            preview = s.get("preview", "")[:40]
            preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
            lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
        if scope_note:
            lines.append(scope_note)
        lines.append(t("gateway.resume.list_footer_numbered"))
        return "\n".join(lines)

    async def _handle_sessions_command(self, event: MessageEvent) -> str:
        """Handle /sessions — list previous sessions for gateway chats."""
        if not self._session_db:
            return self._session_db_unavailable_reply()
        from hermes_cli.session_listing import (
            format_gateway_session_listing, parse_session_listing_args)
        try:
            include_all, include_unnamed, target, search_query = parse_session_listing_args(
                event.get_command_args().strip())
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        if search_query == "":
            return "Usage: `/sessions search <query>`"
        if target:
            return await self._handle_resume_command(dataclasses.replace(event, text=f"/resume {target}"))
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        # `/sessions all` is admin-only like `/resume --all` (else any caller could enumerate other
        # origins' ids/titles/previews); a non-admin gets explicit feedback, not a silent narrowing.
        cross_origin = include_all and self._resume_caller_is_admin(source)
        scope_notice = None
        if include_all and not cross_origin:
            scope_notice = "_Note: `all` (cross-chat listing) requires a configured admin; showing this chat's sessions only._"
        current_entry = await self.async_session_store.get_or_create_session(source)
        rows = await self._list_visible_sessions(
            source, session_key, allow_all=cross_origin,
            current_session_id=current_entry.session_id,
            include_all_sources=cross_origin, include_unnamed=include_unnamed,
            search_query=search_query, exclude_sources=["tool"])
        if cross_origin and source.platform == Platform.MATRIX:
            rows = [self._decorate_matrix_session_row(row) for row in rows]
        if search_query:
            title = f"Sessions matching “{search_query}”"
        else:
            title = "Sessions" if include_unnamed else "Named Sessions"
        return format_gateway_session_listing(rows, include_source=cross_origin, title=title,
                                              notice=scope_notice)

    # ----------------------------------------------------------------------- /branch

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into an independent copy."""
        import json as _json
        import uuid as _uuid
        from datetime import datetime as _dt

        if not self._session_db:
            return self._session_db_unavailable_reply()
        source = event.source
        session_key = self._session_key_for_source(source)
        current_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(current_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        if not history:
            return t("gateway.branch.no_conversation")
        new_session_id = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
        branch_title = event.get_command_args().strip()
        if not branch_title:
            current_title = await self._session_db.get_session_title(current_entry.session_id)
            branch_title = await self._session_db.get_next_title_in_lineage(current_title or "branch")
        parent_session_id = current_entry.session_id
        # Full parent origin (same shape as the reset path in gateway/session.py); the live entry's
        # origin may hold richer metadata than the triggering event's source.
        # See #82633.
        _branch_origin_json = None
        with contextlib.suppress(Exception):
            _branch_origin_json = _json.dumps((current_entry.origin or source).to_dict())
        # ``_branched_from`` keeps the branch visible in /resume and /sessions after the parent is
        # reopened and re-ended. ALL routing columns go in at CREATE time: a crash before
        # switch_session() records the peer would otherwise leave the branch unroutable.
        try:
            await self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                model_config={"_branched_from": parent_session_id},
                parent_session_id=parent_session_id, user_id=source.user_id,
                session_key=session_key, chat_id=source.chat_id, chat_type=source.chat_type,
                thread_id=source.thread_id, origin_json=_branch_origin_json,
                display_name=current_entry.display_name)
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return t("gateway.branch.create_failed", error=e)

        # Chunked transactions; best-effort — a failed copy still yields a usable (partial) branch.
        with contextlib.suppress(Exception):
            # Copy conversation history to the new session in bounded-chunk transactions (see #23254): one
            # txn per row was the removed write-amplification pattern, and a history can be hundreds of
            # rows.
            await self._session_db.append_messages_batch(
                new_session_id, [_branch_row(msg) for msg in history], chunk_rows=500)
        with contextlib.suppress(Exception):
            await self._session_db.set_session_title(new_session_id, branch_title)
        new_entry = await self.async_session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)
        self._evict_cached_agent(session_key)
        msg_count = len([m for m in history if m.get("role") == "user"])
        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)
