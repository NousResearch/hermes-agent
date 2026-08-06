"""/reset (/new), /branch, /resume, /sessions, /title, /topic, /rollback, /undo, /retry, /diff slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Union
import asyncio
import dataclasses
import os
import shlex

from agent.i18n import t
from agent.turn_context import extract_api_content_sidecar
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply
from gateway.platforms.base import MessageEvent
from gateway.platforms.base import MessageType
from gateway.session import SessionSource
from gateway.session import build_session_key
from gateway.session import is_shared_multi_user_session

from gateway.slash_commands._shared import _RESET_CLEANUP_TIMEOUT_S, logger

class SessionLifecycleCommandsMixin:
    """/reset (/new), /branch, /resume, /sessions, /title, /topic, /rollback, /undo, /retry, /diff handlers."""

    async def _handle_reset_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /new or /reset command."""
        source = event.source

        # Get existing session key
        session_key = self._session_key_for_source(source)
        self._invalidate_session_run_generation(session_key, reason="session_reset")
        # Evict the running-agent slot now that the generation is bumped. The
        # in-flight run's own guarded release (run_generation=old) will return
        # False and leave its dead agent behind; clearing here keeps the slot
        # from becoming a zombie that silently drops all later messages (#28686).
        # Idempotent, so the run's finally calling it again is harmless.
        self._release_running_agent_state(session_key)

        # Snapshot the old entry so on_session_finalize can report the
        # expiring session id before reset_session() rotates it.
        old_entry = self.session_store._entries.get(session_key)

        # Close tool resources on the old agent (terminal sandboxes, browser
        # daemons, background processes) before evicting from cache.
        # Guard with getattr because test fixtures may skip __init__.
        #
        # _cleanup_agent_resources is synchronous and can block for a long time
        # (agent.close() does subprocess teardown; shutdown_memory_provider()
        # may do network IO). This handler runs ON the event loop when a
        # Telegram/Discord/Slack confirm-button click resolves the slash-confirm
        # (see _request_slash_confirm), so an inline call wedges the whole loop
        # and the bot goes silent until restart (#35994). Offload it to a worker
        # thread (via the contextvar-preserving executor helper) with a bounded
        # timeout so the loop is never blocked.
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        if _cache_lock is not None:
            with _cache_lock:
                _cached = self._agent_cache.get(session_key)
                _old_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
            if _old_agent is not None:
                try:
                    await asyncio.wait_for(
                        self._run_in_executor_with_context(
                            self._cleanup_agent_resources, _old_agent
                        ),
                        timeout=_RESET_CLEANUP_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    # wait_for cancels the await, but the worker thread cannot be
                    # cancelled — a wedged teardown keeps running (or leaks) for
                    # the gateway's lifetime. The reset proceeds regardless.
                    logger.warning(
                        "Agent resource cleanup for session %s exceeded %ss during "
                        "/new reset; proceeding with reset (the worker thread is left "
                        "to finish on its own). (#35994)",
                        session_key, _RESET_CLEANUP_TIMEOUT_S,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Agent resource cleanup for session %s failed during /new "
                        "reset: %s (#35994)",
                        session_key, cleanup_exc,
                    )
        self._evict_cached_agent(session_key)

        # Conversation boundary: clear ALL conversation-scoped per-session
        # state (model/reasoning overrides, one-turn restores, model notes,
        # last-resolved cache, /queue overflow) + security state in one
        # funnel call. See _CONVERSATION_SCOPED_STATE in gateway/run.py.
        self._clear_conversation_scope(session_key, reason="session_reset")

        # The old conversation's in-flight async delegations end WITH it
        # (#55578): after the reset rotates the session id, their completions
        # would have no live owner — a dangling subagent can only burn tokens
        # and park an orphaned payload on the shared queue. Interrupt by the
        # expiring durable session id (delegations dispatched from gateway
        # chats are pinned to it via parent_session_id) and by the routing
        # key as a fallback for older records.
        try:
            from tools.async_delegation import interrupt_for_session

            interrupt_for_session(
                session_key=session_key,
                parent_session_id=str(getattr(old_entry, "session_id", "") or ""),
                reason="session_reset",
            )
        except Exception:
            pass

        try:
            from tools.env_passthrough import clear_env_passthrough
            clear_env_passthrough()
        except Exception:
            pass

        try:
            from tools.credential_files import clear_credential_files
            clear_credential_files()
        except Exception:
            pass

        # Reset the session
        new_entry = await self.async_session_store.reset_session(session_key)

        # (Conversation-scoped overrides + security state were already
        # cleared via _clear_conversation_scope above.)

        _old_sid = old_entry.session_id if old_entry else None

        # Fire plugin on_session_finalize hook (session boundary)
        try:
            from hermes_cli.lifecycle import finalize_session
            finalize_session(
                session_id=_old_sid,
                platform=source.platform.value if source.platform else "",
                reason="new_session",
                old_session_id=_old_sid,
                new_session_id=new_entry.session_id if new_entry else None,
            )
        except Exception:
            pass

        # Emit session:end hook (session is ending)
        await self.hooks.emit("session:end", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Resolve session config info to surface to the user, scoped to the
        # profile serving this source so a multiplexed /reset //new banner
        # reports the profile's model, not the base config's (#59003).
        try:
            session_info = await asyncio.to_thread(
                self._reset_notice_session_info, source
            )
        except Exception:
            session_info = ""

        if new_entry:
            header = await asyncio.to_thread(self._telegram_topic_new_header, source) or t("gateway.reset.header_default")
        else:
            # No existing session, just create one
            new_entry = await self.async_session_store.get_or_create_session(source, force_new=True)
            header = await asyncio.to_thread(self._telegram_topic_new_header, source) or t("gateway.reset.header_new")

        # Set session title if provided with /new <title>
        _title_arg = event.get_command_args().strip()
        _title_note = ""
        if _title_arg and self._session_db and new_entry:
            from hermes_state import SessionDB
            try:
                sanitized = SessionDB.sanitize_title(_title_arg)
            except ValueError as e:
                sanitized = None
                _title_note = t("gateway.reset.title_rejected", error=str(e))
            if sanitized:
                try:
                    await self._session_db.set_session_title(new_entry.session_id, sanitized)
                    header = t("gateway.reset.header_titled", title=sanitized)
                except ValueError as e:
                    _title_note = t("gateway.reset.title_error_untitled", error=str(e))
                except Exception:
                    pass
            elif not _title_note:
                # sanitize_title returned empty (whitespace-only / unprintable)
                _title_note = t("gateway.reset.title_empty_untitled")
        header = header + _title_note

        # When /new runs inside a Telegram DM topic lane, rewrite the
        # (chat_id, thread_id) → session_id binding so the next message
        # uses the freshly-created session. Without this, the binding
        # still points at the old session and the binding-lookup at the
        # top of _handle_message_with_agent would switch right back.
        if await asyncio.to_thread(self._is_telegram_topic_lane, source) and new_entry is not None:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, new_entry)
            except Exception:
                logger.debug("Failed to rebind Telegram topic after /new", exc_info=True)

        # Fire plugin on_session_reset hook (new session guaranteed to exist)
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _new_sid = new_entry.session_id if new_entry else None
            _invoke_hook(
                "on_session_reset",
                session_id=_new_sid,
                platform=source.platform.value if source.platform else "",
                reason="new_session",
                old_session_id=_old_sid,
                new_session_id=_new_sid,
            )
        except Exception:
            pass

        # Append a random tip to the reset message
        try:
            from hermes_cli.tips import get_random_tip
            _tip_line = t("gateway.reset.tip", tip=get_random_tip())
        except Exception:
            _tip_line = ""

        if session_info:
            return EphemeralReply(f"{header}\n\n{session_info}{_tip_line}")
        return EphemeralReply(f"{header}{_tip_line}")

    def _gateway_session_origin_for_id(self, session_id: str) -> Optional[SessionSource]:
        """Best-effort origin lookup for gateway session IDs."""
        lookup = getattr(type(self.session_store), "lookup_by_session_id", None)
        if callable(lookup):
            entry = lookup(self.session_store, session_id)
            return getattr(entry, "origin", None) if entry is not None else None

        # Test doubles and older stores may not expose the public lookup helper.
        # Keep the Matrix resume guard fail-closed if no origin can be resolved.
        entries = getattr(self.session_store, "_entries", {}) or {}
        for entry in entries.values():
            if getattr(entry, "session_id", None) == session_id:
                return getattr(entry, "origin", None)
        return None

    @staticmethod
    def _same_matrix_room(current: SessionSource, origin: Optional[SessionSource]) -> bool:
        return (
            origin is not None
            and origin.platform == Platform.MATRIX
            and current.platform == Platform.MATRIX
            and origin.chat_id == current.chat_id
            # thread_id is part of the session key (build_session_key appends it
            # for every chat type when present), and Matrix scopes the model's
            # turn to the current room/thread. A live session in another thread
            # of the SAME room is a DIFFERENT session, so a caller in thread A
            # must not resume/enumerate a target whose origin is in thread B.
            # Non-threaded rooms have empty thread_id on both sides ("" == ""),
            # so room-level sharing is preserved unchanged.
            and str(getattr(current, "thread_id", "") or "")
            == str(getattr(origin, "thread_id", "") or "")
        )

    def _same_origin_chat(self, current: SessionSource, origin: Optional[SessionSource]) -> bool:
        """Platform-agnostic counterpart to ``_same_matrix_room``.

        True when *origin* shares *current*'s platform and chat, and the same
        participant whenever the session key for this source is per-user. Group
        and thread sessions that ``build_session_key`` isolates per participant
        (the default ``group_sessions_per_user=True``) must also be scoped by
        participant here — otherwise a co-member could resume another member's
        live per-user group session (IDOR). Only an explicitly shared
        group/thread (``group_sessions_per_user=False`` /
        ``thread_sessions_per_user``) lets co-members share, mirroring the key
        contract via ``is_shared_multi_user_session``.
        """
        if origin is None or current is None:
            return False
        if origin.platform != current.platform:
            return False
        if origin.chat_id != current.chat_id:
            return False
        # thread_id is part of the session key for every chat type when present
        # (build_session_key appends it unconditionally), so a session in one
        # thread is a DIFFERENT session from another thread of the same parent
        # chat. is_shared_multi_user_session only decides participant sharing
        # WITHIN a thread, never across threads — require thread equality before
        # any sharing logic so a live origin in thread A cannot match a caller in
        # thread B of the same parent chat.
        if str(getattr(current, "thread_id", "") or "") != str(
            getattr(origin, "thread_id", "") or ""
        ):
            return False
        chat_type = (getattr(current, "chat_type", "") or "").lower()
        # DM-like chats are always per-user.
        if chat_type in {"dm", "direct", "private", ""}:
            # chat_id was already required equal above and, when present, IS the
            # DM session key — so an equal non-empty chat_id is sufficient.
            # build_session_key only falls back to the participant id
            # (``user_id_alt or user_id`` — Signal/Feishu key on user_id_alt)
            # when there is NO chat_id; mirror that and fail closed on a
            # missing/different participant so two no-chat_id DM origins are
            # never conflated (was: compared user_id only and allowed when
            # either side was missing).
            if str(getattr(current, "chat_id", "") or ""):
                return True
            cur_pid = str(current.user_id_alt or current.user_id or "")
            org_pid = str(origin.user_id_alt or origin.user_id or "")
            return bool(cur_pid) and cur_pid == org_pid
        # Non-DM: scope by participant whenever the session key for this source
        # is per-user. is_shared_multi_user_session mirrors build_session_key's
        # isolation rules exactly, so the guard stays in lock-step with the key.
        shared = is_shared_multi_user_session(
            current,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )
        if shared:
            return True
        # Per-user key: compare the participant id the key is actually built
        # from (user_id_alt or user_id — Signal/Feishu key on user_id_alt).
        cur_pid = current.user_id_alt or current.user_id
        org_pid = origin.user_id_alt or origin.user_id
        if cur_pid and org_pid:
            return cur_pid == org_pid
        # Per-user key but a participant id is missing on one side: cannot prove
        # the same owner — fail closed.
        return False

    def _resume_caller_is_admin(self, source: SessionSource) -> bool:
        """Whether *source* is an EXPLICITLY-configured admin allowed to make a
        cross-origin /resume or /sessions listing.

        Deliberately stricter than ``SlashAccessPolicy.is_admin()``: that returns
        True for every allowed caller when slash gating is DISABLED (so commands
        stay runnable by default), but cross-ORIGIN DATA ACCESS must require a
        real, configured admin. Otherwise the default (no admin list) config
        would treat every gateway caller as cross-origin-capable and re-open the
        enumeration IDOR.
        """
        try:
            from gateway.slash_access import policy_for_source
            policy = policy_for_source(self.config, source)
            uid = getattr(source, "user_id", None)
            return bool(policy.enabled and uid and policy.is_admin(uid))
        except Exception:
            return False

    async def _resume_target_allowed(
        self, source: SessionSource, target_id: str, allow_override: bool = False
    ) -> bool:
        """Whether *source* may resume the persisted session *target_id*.

        Generalizes the Matrix-only room guard to every adapter so a caller
        cannot bind their gateway session to another user's/room's persisted
        session id (IDOR). Uses the live origin when the target is active;
        otherwise falls back to the DB row's source + user_id (the sessions
        table has no chat_id). An identity-bearing caller is allowed only when
        the row PROVES the same owner; a row that lacks enough ownership data
        fails closed. An explicit admin ``--all`` override bypasses scoping.
        """
        if allow_override and self._resume_caller_is_admin(source):
            return True
        # Use the live origin only when it resolves to a real SessionSource; a
        # store that can't resolve it (or an unexpected lookup error) must not
        # silently allow/deny — fall through to the deterministic DB scoping.
        try:
            origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            origin = None
        if isinstance(origin, SessionSource):
            return self._same_origin_chat(source, origin)
        # Inactive/persisted-only: best-effort scope by DB row source + user.
        try:
            row = await self._session_db.get_session(target_id) or {}
        except Exception:
            return False
        caller_src = source.platform.value if source.platform else None
        row_src = row.get("source")
        if row_src and caller_src and str(row_src) != str(caller_src):
            return False  # different platform / source
        caller_uid = str(getattr(source, "user_id", "") or "")
        row_uid = str(row.get("user_id") or "")
        # Chat/thread origin recorded at session creation (see
        # SessionDB._insert_session_row). The sessions table historically stored
        # only source + user_id, so a same-user row could belong to a DIFFERENT
        # chat; comparing the persisted origin closes that gap. Legacy rows
        # created before origin capture have NULL here and therefore fail closed
        # (they cannot prove the caller's chat) — resume them via a live session
        # or an admin override.
        caller_chat = str(getattr(source, "chat_id", "") or "")
        row_chat = str(row.get("chat_id") or "")
        caller_thread = str(getattr(source, "thread_id", "") or "")
        row_thread = str(row.get("thread_id") or "")
        chat_type = (getattr(source, "chat_type", "") or "").lower()
        caller_is_dm = chat_type in {"dm", "direct", "private", ""}
        # build_session_key keys the participant on ``user_id_alt or user_id``
        # (Signal/Feishu carry the canonical participant in user_id_alt), but the
        # sessions table only ever stored user_id — it has no user_id_alt column.
        # So when the caller carries a user_id_alt, the row CANNOT prove the
        # canonical participant that the live session key is built from: two
        # members sharing one user_id but different user_id_alt map to DIFFERENT
        # session keys, yet the persisted row's user_id would match both. The
        # live-origin guard (_same_origin_chat) compares user_id_alt correctly;
        # the persisted fallback cannot, so any per-user comparison that would
        # otherwise rely on row_uid == caller_uid must fail closed here to stay
        # in lock-step with the key boundary (CWE-639). Shared group/thread
        # sessions are unaffected (they don't scope by participant at all), and
        # an admin --all override still bypasses this above.
        caller_keys_on_alt = bool(str(getattr(source, "user_id_alt", "") or ""))
        if caller_uid:
            # Identity-bearing caller: allow only when the row PROVES the same
            # owner AND the same platform/origin AND the same chat/thread. A row
            # with no/blank user_id cannot be proven to belong to this caller; a
            # row with no/blank source cannot be proven to share the caller's
            # platform (the row_src check above only rejects a *mismatching*
            # non-blank source, so a blank/legacy source would otherwise slip
            # through on user_id equality alone); and a row whose origin chat
            # (or thread) differs from the caller's belongs to a different
            # conversation. Any gap fails closed — an identified user must not
            # bind to an unowned, other-owned, other-chat, or unproven-origin
            # persisted session by id/title. (Legacy NULL-owner/blank-source/
            # NULL-chat rows are intentionally not resumable this way; use a
            # live session or an explicit admin override.)
            # Common origin proof for any identity-bearing caller: a non-blank
            # source that matches the caller's platform, and the same thread. A
            # blank/legacy source can't prove the platform; a different thread is
            # a different session (build_session_key appends thread_id).
            origin_ok = (
                bool(row_src) and bool(caller_src)
                and str(row_src) == str(caller_src)
                and row_thread == caller_thread
            )
            if not origin_ok:
                return False
            if caller_is_dm:
                # DMs are keyed on user_id; require the same owner. chat_id is
                # legitimately absent on both sides for a no-chat_id DM (scoped
                # by user_id), but a mismatching chat_id (when present) is still
                # rejected.
                #
                # A no-chat_id DM is keyed PURELY on the participant
                # (``user_id_alt or user_id``). If the caller keys on user_id_alt
                # the persisted row (user_id only) cannot prove that participant,
                # so fail closed. When chat_id is present on both sides it is the
                # DM key and equal chat_id is sufficient, so the alt gap doesn't
                # apply there.
                if caller_keys_on_alt and not (bool(row_chat) and bool(caller_chat)):
                    return False
                return (
                    bool(row_uid) and row_uid == caller_uid
                    and row_chat == caller_chat
                )
            # Non-DM (group/channel/forum/thread): build_session_key includes
            # chat_id, so a row (or caller) with NO chat provenance cannot prove
            # same-chat. Require both sides non-blank and equal — a legacy
            # NULL-chat row (or a caller missing its chat_id) fails closed even
            # when both normalize to "". (CWE-639)
            if not (bool(row_chat) and bool(caller_chat) and row_chat == caller_chat):
                return False
            # Within the same non-DM chat/thread, mirror build_session_key's
            # participant scoping: a SHARED group/thread session
            # (group_sessions_per_user=False, or a shared thread) is one session
            # for every participant, so the same-chat proof above is sufficient —
            # do NOT also require user-id equality (otherwise a co-member is
            # wrongly blocked from their own shared session). A per-user session
            # still requires the same owner.
            shared = is_shared_multi_user_session(
                source,
                group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
                thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
            )
            if shared:
                return True
            # Per-user non-DM: the session key includes the participant
            # (``user_id_alt or user_id``). If the caller keys on user_id_alt,
            # the persisted row (user_id only) cannot prove the canonical
            # participant, so fail closed rather than matching on user_id alone.
            if caller_keys_on_alt:
                return False
            return bool(row_uid) and row_uid == caller_uid
        # No caller identity: the persisted row carries only source + user_id
        # (the sessions table has no chat_id), so a same-platform row can belong
        # to a DIFFERENT chat or user. Same-platform alone is therefore NOT
        # ownership proof — an identity-less caller must not bind to, or
        # enumerate, a persisted session by id/title. Fail closed. A legitimate
        # same-chat resume of an ACTIVE session still works through the
        # live-origin branch above (which compares chat_id), and an operator can
        # use the admin --all override. (CWE-639: IDOR on session routing.)
        return False

    async def _resume_row_visible(
        self, source: SessionSource, row: dict, allow_all: bool
    ) -> bool:
        """Whether a titled-session listing *row* belongs to the caller's origin.

        Prevents cross-origin enumeration of session ids/previews via the
        numbered /resume list. Preserves the existing Matrix room-scoping
        semantics; scopes every other platform to the caller's own sessions
        unless an admin passes ``--all``.
        """
        sid = str(row.get("id") or "")
        if source.platform == Platform.MATRIX:
            # Cross-room enumeration is cross-ORIGIN data access: gate the
            # ``--all`` short-circuit behind a real configured admin, exactly
            # like the non-Matrix branch below. A non-admin Matrix ``--all``
            # falls back to same-room scoping rather than exposing every Matrix
            # titled session.
            if allow_all and self._resume_caller_is_admin(source):
                return True
            return self._same_matrix_room(source, self._gateway_session_origin_for_id(sid))
        if allow_all and self._resume_caller_is_admin(source):
            return True
        return await self._resume_target_allowed(source, sid, allow_override=False)

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        # Find the last *real* user message. Timeline bookkeeping rows carry
        # role=user + display_kind (model_switch / async_delegation_complete /
        # auto_continue / hidden); clients never count them as user turns.
        # Without this filter /retry rewrote the transcript around a marker
        # and re-sent opaque bookkeeping text (same class as the TUI ordinal).
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if msg.get("role") == "user" and not msg.get("display_kind"):
                last_user_msg = msg.get("content", "")
                last_user_idx = i
                break

        if not last_user_msg:
            return t("gateway.retry.no_previous")

        # Truncate history to before the last user message and persist
        truncated = history[:last_user_idx]
        await self.async_session_store.rewrite_transcript(session_entry.session_id, truncated)
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0

        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
            channel_prompt=event.channel_prompt,
        )

        # Let the normal message handler process it
        return await self._handle_message(retry_event)

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo [N] — back up N user turns (default 1), soft-deleting
        the truncated rows on disk and echoing the backed-up message text so
        the user can copy/edit and resend.

        Mirrors the CLI/TUI /undo: rewound rows stay in state.db (active=0)
        for audit and are hidden from re-prompts and search. The cached agent
        is evicted so the next message rebuilds context from the truncated
        (active-only) transcript — the gateway's equivalent of the CLI's
        in-place history surgery + memory-cache invalidation.
        """
        source = event.source

        # Parse optional turn count: "/undo" → 1, "/undo 3" → 3.
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = int(raw_args.split()[0])
            except (ValueError, IndexError):
                return t("gateway.undo.invalid_count", arg=raw_args.split()[0])
            if n < 1:
                n = 1

        session_entry = await self.async_session_store.get_or_create_session(source)
        result = await self.async_session_store.rewind_session(session_entry.session_id, n)

        if result is None:
            return t("gateway.undo.nothing")

        # Reset stored token count — transcript was truncated.
        session_entry.last_prompt_tokens = 0
        # Evict the cached agent so the next turn rebuilds from the active-only
        # transcript and memory providers refresh their per-session caches.
        try:
            session_key = build_session_key(source)
            self._evict_cached_agent(session_key)
        except Exception as e:
            logger.debug("undo: cached-agent eviction skipped: %s", e)

        target_text = result["target_text"]
        preview = target_text[:200] + "..." if len(target_text) > 200 else target_text
        return t(
            "gateway.undo.removed",
            turns=result["turns_undone"],
            count=result["rewound_count"],
            preview=preview,
        )

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        cp_kwargs = _checkpoint_agent_kwargs(_load_gateway_config())

        if not cp_kwargs["checkpoints_enabled"]:
            return t("gateway.rollback.not_enabled")

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_kwargs["checkpoint_max_snapshots"],
            max_total_size_mb=cp_kwargs["checkpoint_max_total_size_mb"],
            max_file_size_mb=cp_kwargs["checkpoint_max_file_size_mb"],
        )

        cwd = os.getenv("TERMINAL_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return t("gateway.rollback.none_found", cwd=cwd)

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return t("gateway.rollback.invalid_number", max=len(checkpoints))
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash)
        if result["success"]:
            return t(
                "gateway.rollback.restored",
                hash=result["restored_to"],
                reason=result["reason"],
            )
        return t("gateway.rollback.restore_failed", error=result["error"])

    async def _handle_diff_command(self, event: MessageEvent) -> str:
        """Handle /diff — show git changes in the working directory.

        ``/diff`` (default) shows unstaged + untracked changes, ``/diff
        staged`` the staged ones, ``/diff all`` everything since HEAD, and
        ``/diff session`` the cumulative checkpoint-baseline diff of what
        Hermes itself changed. ``--stat`` limits output to the summary.

        The diff body is truncated hard here (messaging surfaces are not a
        pager); platform senders additionally split/clamp long messages to
        per-platform limits, the same way tool-progress output is truncated
        in three layers before delivery.
        """
        args = event.get_command_args().strip()

        stat_only = False
        mode = "working"
        for arg in args.split():
            low = arg.lower()
            if low in ("--stat", "stat"):
                stat_only = True
            elif low in ("staged", "--staged", "cached", "--cached"):
                mode = "staged"
            elif low in ("all", "--all", "head"):
                mode = "all"
            elif low == "session":
                mode = "session"

        cwd = os.getenv("TERMINAL_CWD", str(Path.home()))

        if mode == "session":
            return await self._gateway_session_diff(cwd, stat_only)

        from tools.working_diff import collect_working_diff

        result = await asyncio.to_thread(collect_working_diff, cwd, mode)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            return t("gateway.diff.no_changes")

        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if untracked:
            shown = "\n".join(f"+ {rel}" for rel in untracked[:15])
            more = f"\n... and {len(untracked) - 15} more" if len(untracked) > 15 else ""
            out.append(f"**Untracked:**\n```\n{shown}{more}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    async def _gateway_session_diff(self, cwd: str, stat_only: bool) -> str:
        """Cumulative checkpoint-baseline diff for /diff session (gateway)."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager

        cp_kwargs = _checkpoint_agent_kwargs(_load_gateway_config())
        if not cp_kwargs["checkpoints_enabled"]:
            return t("gateway.diff.not_enabled")

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_kwargs["checkpoint_max_snapshots"],
            max_total_size_mb=cp_kwargs["checkpoint_max_total_size_mb"],
            max_file_size_mb=cp_kwargs["checkpoint_max_file_size_mb"],
        )

        result = await asyncio.to_thread(mgr.session_diff, cwd)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        if result.get("empty") or (not stat and not diff):
            return t("gateway.diff.no_changes")

        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    @staticmethod
    def _fenced_truncated_diff(diff: str, max_lines: int = 60,
                               max_chars: int = 3000) -> str:
        """Fence a diff body, truncating to messaging-friendly size."""
        diff_lines = diff.splitlines()
        truncated = False
        if len(diff_lines) > max_lines:
            diff = "\n".join(diff_lines[:max_lines])
            truncated = True
        if len(diff) > max_chars:
            diff = diff[:max_chars]
            truncated = True
        note = ""
        if truncated:
            note = (
                f"\n... (truncated — {len(diff_lines)} lines total; "
                "use /diff --stat for a summary)"
            )
        return f"```diff\n{diff}{note}\n```"

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return t("gateway.topic.not_telegram_dm")
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        # Authorization: /topic activates multi-session mode and mutates
        # SQLite side tables. Unauthorized senders (not in allowlist) must
        # not be able to do that. Gateway routes already authorize the
        # message before reaching here, but defense in depth.
        auth_fn = getattr(self, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                if not auth_fn(source):
                    return t("gateway.topic.unauthorized")
            except Exception:
                logger.debug("Topic auth check failed", exc_info=True)

        args = event.get_command_args().strip()

        # /topic help — inline usage without leaving the bot.
        if args.lower() in {"help", "?", "-h", "--help"}:
            return self._telegram_topic_help_text()

        # /topic off — clean disable path so users don't have to edit the DB.
        if args.lower() in {"off", "disable", "stop"}:
            return await self._disable_telegram_topic_mode_for_chat(source)

        if args:
            if not source.thread_id:
                return t("gateway.topic.restore_needs_topic")
            return await self._restore_telegram_topic_session(event, args)

        capabilities = await self._get_telegram_topic_capabilities(source)
        if capabilities.get("checked"):
            if capabilities.get("has_topics_enabled") is False:
                # Debounce the BotFather screenshot: don't re-send on every
                # /topic while threads are still disabled.
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_disabled")
            if capabilities.get("allows_users_to_create_topics") is False:
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t("gateway.topic.topics_user_disallowed")

        try:
            await self._session_db.enable_telegram_topic_mode(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                has_topics_enabled=capabilities.get("has_topics_enabled"),
                allows_users_to_create_topics=capabilities.get("allows_users_to_create_topics"),
            )
        except Exception as exc:
            logger.exception("Failed to enable Telegram topic mode")
            return t("gateway.topic.enable_failed", error=exc)

        if not source.thread_id:
            await self._ensure_telegram_system_topic(source)

        if source.thread_id:
            try:
                binding = await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                session_id = str(binding.get("session_id") or "")
                title = None
                try:
                    title = await self._session_db.get_session_title(session_id)
                except Exception:
                    title = None
                session_label = title or t("gateway.topic.untitled_session")
                return t(
                    "gateway.topic.bound_status",
                    label=session_label,
                    session_id=session_id,
                )
            return t("gateway.topic.thread_ready")

        return await self._telegram_topic_root_status_message(source)

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        # Ensure session exists in SQLite DB (it may only exist in session_store
        # if this is the first command in a new session)
        existing_title = await self._session_db.get_session_title(session_id)
        if existing_title is None:
            # Session doesn't exist in DB yet — create it
            try:
                await self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                    # Persist the messaging origin so a later /resume of this
                    # titled-but-now-inactive session can prove it belongs to the
                    # caller's chat/thread (IDOR scoping).
                    chat_id=source.chat_id,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                from hermes_state import SessionDB
                sanitized = SessionDB.sanitize_title(title_arg)
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
            if not sanitized:
                return t("gateway.title.empty_after_clean")
            # Set the title
            try:
                if await self._session_db.set_session_title(session_id, sanitized):
                    # Propagate the user-chosen title to the visible Telegram
                    # forum topic name too. Auto-generated titles already rename
                    # the topic; without this, /title only updated the DB title
                    # and the topic kept its auto-assigned name. No-ops off
                    # Telegram topic lanes and when auto-rename is disabled.
                    schedule_rename = getattr(
                        self, "_schedule_telegram_topic_title_rename", None
                    )
                    if callable(schedule_rename):
                        try:
                            await asyncio.to_thread(schedule_rename, source, session_id, sanitized)
                        except Exception:
                            logger.debug(
                                "Failed to rename Telegram topic from /title",
                                exc_info=True,
                            )
                    return t("gateway.title.set_to", title=sanitized)
                else:
                    return t("gateway.title.not_found")
            except ValueError as e:
                return t("gateway.shared.warn_passthrough", error=e)
        else:
            # Show the current title and session ID
            title = await self._session_db.get_session_title(session_id)
            if title:
                return t("gateway.title.current_with_title", session_id=session_id, title=title)
            else:
                return t("gateway.title.current_no_title", session_id=session_id)

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — list or switch to a previous session."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        source = await asyncio.to_thread(
            self._normalize_source_for_session_key, event.source
        )
        session_key = self._session_key_for_source(source)
        raw_args = event.get_command_args().strip()
        try:
            parts = shlex.split(raw_args)
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        allow_all = "--all" in parts
        allow_cross_room = "--cross-room" in parts
        name = " ".join(p for p in parts if p not in {"--all", "--cross-room"}).strip()

        # Strip common outer brackets/quotes users may type literally from the
        # usage hint (e.g. ``/resume <abc123>``). Mirrors the CLI behavior.
        if len(name) >= 2 and (
            (name[0] == "<" and name[-1] == ">")
            or (name[0] == "[" and name[-1] == "]")
            or (name[0] == '"' and name[-1] == '"')
            or (name[0] == "'" and name[-1] == "'")
        ):
            name = name[1:-1].strip()

        async def _list_titled_sessions() -> list[dict]:
            user_source = source.platform.value if source.platform else None
            widen = allow_all and self._resume_caller_is_admin(source)
            sessions = await self._session_db.list_sessions_rich(
                source=user_source,
                session_key=None if widen else session_key,
                limit=10,
            )
            return [s for s in sessions if s.get("title")][:10]

        if not name:
            # List recent titled sessions for this user/platform
            try:
                titled = await _list_titled_sessions()
                titled = [
                    s for s in titled
                    if await self._resume_row_visible(source, s, allow_all)
                ]
                if not titled:
                    if source.platform == Platform.MATRIX and not allow_all:
                        return t("gateway.resume.matrix_no_named_sessions")
                    return t("gateway.resume.no_named_sessions")
                lines = [t("gateway.resume.list_header")]
                for idx, s in enumerate(titled[:10], start=1):
                    title = s["title"]
                    if source.platform == Platform.MATRIX and allow_all:
                        origin = self._gateway_session_origin_for_id(str(s.get("id") or ""))
                        if origin:
                            title = f"{title} — {origin.chat_name or origin.chat_id}"
                    preview = s.get("preview", "")[:40]
                    preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
                    lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
                lines.append(t("gateway.resume.list_footer_numbered"))
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return t("gateway.resume.list_failed", error=e)

        # Resolve a numbered choice or a title to a session ID.
        if name.isdigit():
            try:
                titled = await _list_titled_sessions()
                titled = [
                    s for s in titled
                    if await self._resume_row_visible(source, s, allow_all)
                ]
            except Exception as e:
                logger.debug("Failed to list titled sessions for numeric resume: %s", e)
                return t("gateway.resume.list_failed", error=e)
            index = int(name)
            if index < 1 or index > len(titled):
                return t("gateway.resume.out_of_range", index=index)
            target = titled[index - 1]
            target_id = target.get("id")
            name = target.get("title") or name
        else:
            # Try direct session ID lookup first (so `/resume <session_id>`
            # works in the gateway, not just `/resume <title>`).
            session = await self._session_db.get_session(name)
            if session:
                target_id = session["id"]
            else:
                target_id = await self._session_db.resolve_session_by_title(name)
        if not target_id:
            return t("gateway.resume.not_found", name=name)
        # Compression creates child continuations that hold the live transcript.
        # Follow that chain so gateway /resume matches CLI behavior (#15000).
        try:
            target_id = await self._session_db.resolve_resume_session_id(target_id)
        except Exception as e:
            logger.debug("Failed to resolve resume continuation for %s: %s", target_id, e)

        if source.platform == Platform.MATRIX:
            target_origin = self._gateway_session_origin_for_id(target_id)
            if not self._same_matrix_room(source, target_origin) and not allow_cross_room:
                if target_origin is None:
                    return t("gateway.resume.matrix_blocked_no_origin", name=name)
                return t(
                    "gateway.resume.matrix_blocked_other_room",
                    room=target_origin.chat_name or target_origin.chat_id,
                    name=name,
                )
        elif not await self._resume_target_allowed(
            source, target_id, allow_override=(allow_all or allow_cross_room)
        ):
            # IDOR guard: a session id/title is a routing handle, not authority.
            # Bind /resume to the caller's own platform/user/chat on every
            # non-Matrix adapter so one user can't attach to another's
            # persisted transcript.
            return t("gateway.resume.blocked_not_owner", name=name)

        # Check if already on that session
        current_entry = await self.async_session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return t("gateway.resume.already_on", name=name)

        # Clear any running agent for this session key
        self._release_running_agent_state(session_key)

        # Switch the session entry to point at the old session
        new_entry = await self.async_session_store.switch_session(session_key, target_id)
        if not new_entry:
            return t("gateway.resume.switch_failed")

        # Conversation boundary: clear ALL conversation-scoped per-session
        # state (model/reasoning overrides #10702, one-turn restores, model
        # notes, last-resolved cache #58403, /queue overflow) + security
        # state in one funnel call. See _CONVERSATION_SCOPED_STATE in
        # gateway/run.py.
        self._clear_conversation_scope(session_key, reason="resume")

        # Evict any cached agent for this session so the next message
        # rebuilds with the correct session_id end-to-end — mirrors
        # /branch and /reset. Without this, the cached AIAgent (and its
        # memory provider, which cached `_session_id` during initialize())
        # keeps writing into the wrong session's record. See #6672.
        self._evict_cached_agent(session_key)

        # Get the title for confirmation
        title = await self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = await self.async_session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        if source.platform == Platform.MATRIX and allow_cross_room:
            return t(
                "gateway.resume.matrix_cross_room_success",
                title=title,
                room=source.chat_name or source.chat_id,
                msg_part=msg_part,
            )
        if not msg_count:
            return t("gateway.resume.resumed_no_count", title=title)
        if msg_count == 1:
            return t("gateway.resume.resumed_one", title=title, count=msg_count)
        return t("gateway.resume.resumed_many", title=title, count=msg_count)

    async def _handle_sessions_command(self, event: MessageEvent) -> str:
        """Handle /sessions — list previous sessions for gateway chats."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        from hermes_cli.session_listing import (
            format_gateway_session_listing,
            parse_session_listing_args,
            query_session_listing,
        )

        raw_args = event.get_command_args().strip()
        try:
            include_all, include_unnamed, target, search_query = (
                parse_session_listing_args(raw_args)
            )
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)

        if search_query == "":
            return "Usage: `/sessions search <query>`"

        if target:
            resume_event = dataclasses.replace(event, text=f"/resume {target}")
            return await self._handle_resume_command(resume_event)

        source = await asyncio.to_thread(
            self._normalize_source_for_session_key, event.source
        )
        session_key = self._session_key_for_source(source)

        # A cross-origin listing (`/sessions all`) is honored only for an
        # admin, mirroring the `/resume --all` override. `all` is just a parsed
        # user argument, so without this gate any caller could run
        # `/sessions all` and enumerate other origins' session ids / titles /
        # previews / sources — the enumeration half of the /resume IDOR.
        cross_origin = include_all and self._resume_caller_is_admin(source)
        current_entry = await self.async_session_store.get_or_create_session(source)
        rows = await asyncio.to_thread(
            query_session_listing,
            getattr(self._session_db, "_db", self._session_db),
            source=source.platform.value if source.platform else None,
            session_key=None if cross_origin else session_key,
            current_session_id=current_entry.session_id,
            include_all_sources=cross_origin,
            include_unnamed=include_unnamed,
            search_query=search_query,
            # Search filters at SQL level, so over-fetch before the visibility
            # cut: origin-invisible matches would otherwise consume the page.
            limit=50 if search_query else 10,
            exclude_sources=["tool"],
        )
        if not cross_origin:
            # Scope the listing to the caller's own origin on every adapter so
            # session ids/previews from other users/rooms aren't enumerable.
            rows = [
                row for row in rows
                if await self._resume_row_visible(source, row, allow_all=False)
            ]
        rows = rows[:10]
        if search_query:
            title = f"Sessions matching “{search_query}”"
        else:
            title = "Sessions" if include_unnamed else "Named Sessions"
        return format_gateway_session_listing(
            rows,
            include_source=cross_origin,
            title=title,
        )

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies conversation history to a new session so the user can explore
        a different approach without losing the original.
        Inspired by Claude Code's /branch command.
        """
        import uuid as _uuid

        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        source = event.source
        session_key = self._session_key_for_source(source)

        # Load the current session and its transcript
        current_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(current_entry.session_id)
        if not history:
            return t("gateway.branch.no_conversation")

        branch_name = event.get_command_args().strip()

        # Generate the new session ID
        from datetime import datetime as _dt
        now = _dt.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            current_title = await self._session_db.get_session_title(current_entry.session_id)
            base = current_title or "branch"
            branch_title = await self._session_db.get_next_title_in_lineage(base)

        parent_session_id = current_entry.session_id

        # Create the new session with parent link.
        # Persist a stable ``_branched_from`` marker in model_config so
        # list_sessions_rich() keeps the branch visible in /resume and
        # /sessions even after the parent is reopened and re-ended with a
        # different end_reason (e.g. tui_shutdown overwriting 'branched').
        try:
            await self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                model_config={"_branched_from": parent_session_id},
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return t("gateway.branch.create_failed", error=e)

        # Copy conversation history to the new session in bounded-chunk
        # transactions (see #23254): one txn per row was the removed
        # write-amplification pattern, and a history can be hundreds of rows.
        # Best-effort like the old loop — a failed copy still yields a
        # usable (partial) branch.
        try:
            await self._session_db.append_messages_batch(
                new_session_id,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        "tool_name": msg.get("tool_name") or msg.get("name"),
                        "tool_calls": msg.get("tool_calls"),
                        "tool_call_id": msg.get("tool_call_id"),
                        "finish_reason": msg.get("finish_reason"),
                        "reasoning": msg.get("reasoning"),
                        "reasoning_content": msg.get("reasoning_content"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "codex_reasoning_items": msg.get("codex_reasoning_items"),
                        "codex_message_items": msg.get("codex_message_items"),
                        # Keep the api_content sidecar so the branch's first turn
                        # replays the parent's exact wire bytes (warm provider
                        # prompt cache) instead of a full cold prefill.
                        "api_content": extract_api_content_sidecar(msg),
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in history
                ],
                chunk_rows=500,
            )
        except Exception:
            pass  # Best-effort copy

        # Set title
        try:
            await self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch the session store entry to the new session
        new_entry = await self.async_session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)
