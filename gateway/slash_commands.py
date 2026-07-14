"""Gateway slash-command handlers for GatewayRunner.

Extracted from ``gateway/run.py`` (god-file decomposition Phase 3b). These are
the in-session slash commands (/model, /reset, /usage, /compress, ...) the
gateway dispatches from ``_handle_message``. There are 42 of them (~3,200 LOC);
lifting them into a mixin that ``GatewayRunner`` inherits keeps every
``self._handle_*_command`` dispatch + test reference working via the MRO, while
removing the bulk from run.py.

Module-level run.py helpers a handler needs (``_hermes_home``,
``_load_gateway_config``, ``_resolve_gateway_model``, etc.) are imported lazily
inside the handler body — a deferred ``from gateway.run import ...`` resolves at
call time (run.py fully loaded by then), avoiding an import cycle.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import logging
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.i18n import t
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
)
from hermes_cli.config import atomic_config_write, cfg_get, clear_model_endpoint_credentials
from utils import (
    atomic_json_write,
    base_url_host_matches,
    is_truthy_value,
)

logger = logging.getLogger("gateway.run")

# Upper bound on the off-loop agent-resource cleanup during a /new or /reset
# (see _handle_reset_command). A stuck teardown must not block the event loop;
# past this the reset proceeds and the cleanup is left to finish (or leak) in
# its worker thread. (#35994)
_RESET_CLEANUP_TIMEOUT_S = 30.0


def _model_switch_skew_guard() -> Optional[str]:
    """Refuse a model switch when the gateway is running stale code.

    A long-lived gateway holds its modules in memory from boot. If the checkout
    changed underneath it (e.g. a manual ``git pull``), switching models can hit
    a first-time lazy import on a new code path and crash on a stale cached
    dependency — the cryptic ``cannot import name 'env_float' from 'utils'``.
    Detect the drift and tell the user to restart instead.

    Intentionally scoped to model switching — the known, highest-risk trigger.
    Any first-time lazy import on a stale process is technically exposed; we
    don't guard every import site, only this one.
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return t(
        "gateway.model.error_prefix",
        error=(
            f"This gateway is running code from {boot_rev} but the checkout on "
            f"disk is now {disk_rev}. Switching models would risk a stale-module "
            f"crash — restart the gateway to load the new code: hermes gateway restart"
        ),
    )


class GatewaySlashCommandsMixin:
    """In-session slash-command handlers for GatewayRunner."""

    async_session_store: AsyncSessionStore

    def _typed_command_prefix_for(self, platform) -> str:
        """Return the prefix users can always type to reach Hermes commands.

        Reads the adapter's ``typed_command_prefix`` capability flag
        (default "/"). Slack and Matrix return "!" because typed "/"
        commands are blocked in Slack threads / reserved by Matrix clients;
        their adapters rewrite "!command" to "/command" on receive.
        Instruction text built for those platforms must show the prefix
        that actually works when typed.
        """
        adapter = self.adapters.get(platform) if getattr(self, "adapters", None) else None
        return getattr(adapter, "typed_command_prefix", "/") if adapter is not None else "/"

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

        # Discard any /queue overflow for this session — /new is a
        # conversation-boundary operation, queued follow-ups from the
        # previous conversation must not bleed into the new one.
        _qe = getattr(self, "_queued_events", None)
        if _qe is not None:
            _qe.pop(session_key, None)

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

        # Clear any session-scoped model/reasoning overrides so the next agent
        # picks up configured defaults instead of previous session switches.
        self._session_model_overrides.pop(session_key, None)
        self._set_session_reasoning_override(session_key, None)
        if hasattr(self, "_pending_model_notes"):
            self._pending_model_notes.pop(session_key, None)

        # Clear the per-session last-resolved-model cache so the next turn
        # reads from current config instead of falling back to a stale model
        # after a config change (#58403).
        _lrm = getattr(self, "_last_resolved_model", None)
        if _lrm is not None:
            _lrm.pop(session_key, None)

        # Clear session-scoped dangerous-command approvals and /yolo state.
        # /new is a conversation-boundary operation — approval state from the
        # previous conversation must not survive the reset.
        self._clear_session_boundary_security_state(session_key)

        _old_sid = old_entry.session_id if old_entry else None

        # Fire plugin on_session_finalize hook (session boundary)
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_finalize",
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
            from hermes_cli.plugins import invoke_hook as _invoke_hook
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

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show active profile name and home directory."""
        from hermes_constants import display_hermes_home
        from hermes_cli.profiles import get_active_profile_name

        display = display_hermes_home()
        profile_name = get_active_profile_name()

        lines = [
            t("gateway.profile.header", profile=profile_name),
            t("gateway.profile.home", home=display),
        ]

        return "\n".join(lines)

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami — show the user's slash command access on this scope.

        Always works (it's in the always-allowed floor of slash_access).
        Reports: platform, scope (DM vs group), the user's tier
        (admin / user / unrestricted), and the slash commands they can
        actually run on this scope.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        source = event.source
        policy = _policy_for_source(self.config, source)
        platform = source.platform.value if source and source.platform else "?"
        chat_type = (source.chat_type if source else "") or "dm"
        scope = "DM" if chat_type.lower() in {"dm", "direct", "private", ""} else "group/channel"
        user_id = (source.user_id if source else None) or "?"

        if not policy.enabled:
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: unrestricted (no admin list configured for this scope)\n"
                f"Slash commands: all available"
            )

        if policy.is_admin(user_id):
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: **admin**\n"
                f"Slash commands: all available"
            )

        # Non-admin user. Show what's actually reachable.
        floor = ["help", "whoami"]  # mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS
        configured = sorted(policy.user_allowed_commands)
        # Combine + dedupe, preserve order: floor first, then operator additions.
        seen: set[str] = set()
        runnable: list[str] = []
        for c in floor + configured:
            if c not in seen:
                seen.add(c)
                runnable.append(c)
        runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
        return (
            f"**You** — {platform} ({scope})\n"
            f"User ID: `{user_id}`\n"
            f"Tier: user\n"
            f"Slash commands you can run: {runnable_str}"
        )

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban — delegate to the shared kanban CLI.

        Run the potentially-blocking DB work in a thread pool so the
        gateway event loop stays responsive.  Read operations (list,
        show, context, tail) are permitted while an agent is running;
        mutations are allowed too because the board is profile-agnostic
        and does not touch the running agent's state.

        For ``/kanban create`` invocations we also auto-subscribe the
        originating gateway source (platform + chat + thread) to the new
        task's terminal events, so the user hears back when the worker
        completes / blocks / auto-blocks / crashes without having to poll.
        """
        import asyncio
        import re
        import shlex
        from hermes_cli.kanban import run_slash

        text = (event.text or "").strip()
        # Strip the leading "/kanban" (with or without slash), leaving args.
        if text.startswith("/"):
            text = text.lstrip("/")
        if text.startswith("kanban"):
            text = text[len("kanban"):].lstrip()

        tokens = shlex.split(text) if text else []
        requested_board = None
        action = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--board":
                if i + 1 >= len(tokens):
                    break
                requested_board = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--board="):
                requested_board = tok.split("=", 1)[1]
                i += 1
                continue
            action = tok
            break

        is_create = action == "create"

        try:
            output = await asyncio.to_thread(run_slash, text)
        except Exception as exc:  # pragma: no cover - defensive
            return t("gateway.kanban.error_prefix", error=exc)

        # Auto-subscribe on create. Parse the task id from the CLI's standard
        # success line ("Created t_abcd  (ready, assignee=...)"). If the user
        # passed --json we don't subscribe; they're clearly scripting and
        # can call /kanban notify-subscribe explicitly.
        if is_create and output:
            m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
            if m:
                task_id = m.group(1)
                try:
                    source = event.source
                    platform = getattr(source, "platform", None)
                    platform_str = (
                        platform.value if hasattr(platform, "value") else str(platform or "")
                    ).lower()
                    chat_id = str(getattr(source, "chat_id", "") or "")
                    thread_id = str(getattr(source, "thread_id", "") or "")
                    user_id = str(getattr(source, "user_id", "") or "") or None
                    if platform_str and chat_id:
                        def _sub():
                            from hermes_cli import kanban_db as _kb
                            conn = _kb.connect(board=requested_board)
                            try:
                                _kb.add_notify_sub(
                                    conn, task_id=task_id,
                                    platform=platform_str, chat_id=chat_id,
                                    thread_id=thread_id or None,
                                    user_id=user_id,
                                    notifier_profile=getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name(),
                                )
                            finally:
                                conn.close()
                        await asyncio.to_thread(_sub)
                        output = (
                            output.rstrip()
                            + "\n"
                            + t("gateway.kanban.subscribed_suffix", task_id=task_id)
                        )
                except Exception as exc:
                    logger.warning("kanban create auto-subscribe failed: %s", exc)

        # Gateway messages have practical length caps; truncate long
        # listings to keep the UX reasonable.
        if len(output) > 3800:
            output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
        return output or t("gateway.kanban.no_output")

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _load_gateway_config, _resolve_gateway_model

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)

        connected_platforms = [p.value for p in self.adapters.keys()]

        # Check if there's an active agent. Keep the sentinel distinct: a
        # starting/pending run should not be treated as a fully usable agent for
        # model/context display, but it still occupies the session slot.
        session_key = session_entry.session_key
        agent = self._running_agents.get(session_key)
        is_running = agent is not None and agent is not _AGENT_PENDING_SENTINEL

        # Count pending /queue follow-ups (slot + overflow).
        adapter = self.adapters.get(source.platform) if source else None
        queue_depth = self._queue_depth(session_key, adapter=adapter)

        def _clean_str(value: Any) -> str:
            return value.strip() if isinstance(value, str) and value.strip() else ""

        def _int_value(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        title = None
        session_row: dict[str, Any] = {}
        # Pull token totals from the SQLite session DB rather than the
        # in-memory SessionStore.  The agent's per-turn token deltas are
        # persisted into sessions_db (run_agent.py), not into SessionEntry,
        # so session_entry.total_tokens is always 0.  SessionDB is the
        # single source of truth; reading it here keeps /status accurate
        # without duplicating token writes into two stores.
        db_total_tokens = 0
        if self._session_db:
            try:
                title = await self._session_db.get_session_title(session_entry.session_id)
            except Exception:
                title = None
            try:
                row = await self._session_db.get_session(session_entry.session_id)
                if isinstance(row, dict):
                    session_row = row
                    db_total_tokens = (
                        _int_value(row.get("input_tokens"))
                        + _int_value(row.get("output_tokens"))
                        + _int_value(row.get("cache_read_tokens"))
                        + _int_value(row.get("cache_write_tokens"))
                        + _int_value(row.get("reasoning_tokens"))
                    )
            except Exception:
                db_total_tokens = 0

        # Resolve model/context for cockpit-style status. Prefer the live or
        # cached agent because it carries the actual runtime route and context
        # compressor. Fall back to persisted SessionDB metadata plus the
        # SessionStore's last_prompt_tokens so /status remains useful between
        # turns without making billing/account calls.
        status_agent = agent if is_running else None
        if status_agent is None:
            cache_lock = getattr(self, "_agent_cache_lock", None)
            cache = getattr(self, "_agent_cache", None)
            if cache_lock is not None and cache is not None:
                try:
                    with cache_lock:
                        cached = cache.get(session_key)
                    if cached:
                        status_agent = cached[0]
                except Exception:
                    status_agent = None

        model_name = ""
        provider_name = ""
        base_url = ""
        context_used = 0
        context_total = 0
        if status_agent is not None and status_agent is not _AGENT_PENDING_SENTINEL:
            model_name = _clean_str(getattr(status_agent, "model", ""))
            provider_name = _clean_str(getattr(status_agent, "provider", ""))
            base_url = _clean_str(getattr(status_agent, "base_url", ""))
            ctx = getattr(status_agent, "context_compressor", None)
            if ctx is not None:
                context_used = _int_value(getattr(ctx, "last_prompt_tokens", 0))
                context_total = _int_value(getattr(ctx, "context_length", 0))

        model_name = model_name or _clean_str(session_row.get("model"))
        provider_name = provider_name or _clean_str(session_row.get("billing_provider"))
        base_url = base_url or _clean_str(session_row.get("billing_base_url"))
        context_used = context_used or _int_value(getattr(session_entry, "last_prompt_tokens", 0))

        user_config: dict[str, Any] = {}
        if not model_name or not provider_name or not context_total:
            try:
                user_config = _load_gateway_config()
            except Exception:
                user_config = {}
        if not model_name:
            model_name = _resolve_gateway_model(user_config)
        if not provider_name:
            model_cfg = user_config.get("model", {}) if isinstance(user_config, dict) else {}
            if isinstance(model_cfg, dict):
                provider_name = _clean_str(model_cfg.get("provider"))
        if not context_total:
            model_cfg = user_config.get("model", {}) if isinstance(user_config, dict) else {}
            configured_context = model_cfg.get("context_length") if isinstance(model_cfg, dict) else None
            if isinstance(configured_context, int) and configured_context > 0:
                context_total = configured_context

        model_line = ""
        if model_name:
            if provider_name:
                model_line = t("gateway.status.model_provider", model=model_name, provider=provider_name)
            else:
                model_line = t("gateway.status.model", model=model_name)

        context_line = ""
        if context_total:
            pct = min(100, round((context_used / context_total) * 100)) if context_total else 0
            context_line = t(
                "gateway.status.context",
                used=f"{context_used:,}",
                total=f"{context_total:,}",
                pct=f"{pct}",
            )
        elif context_used:
            context_line = t("gateway.status.context_used", used=f"{context_used:,}")

        lines = [
            t("gateway.status.header"),
            "",
            t("gateway.status.session_id", session_id=session_entry.session_id),
        ]
        if title:
            lines.append(t("gateway.status.title", title=title))
        lines.extend([
            t("gateway.status.created", timestamp=session_entry.created_at.strftime('%Y-%m-%d %H:%M')),
            t("gateway.status.last_activity", timestamp=session_entry.updated_at.strftime('%Y-%m-%d %H:%M')),
        ])
        if model_line:
            lines.append(model_line)
        if context_line:
            lines.append(context_line)
        lines.extend([
            t("gateway.status.tokens", tokens=f"{db_total_tokens:,}"),
            t("gateway.status.agent_running", state=t("gateway.status.state_yes") if is_running else t("gateway.status.state_no")),
        ])
        if queue_depth:
            lines.append(t("gateway.status.queued", count=queue_depth))
        if source.platform == Platform.MATRIX:
            adapter = self.adapters.get(Platform.MATRIX)
            scope = getattr(adapter, "_matrix_session_scope", os.getenv("MATRIX_SESSION_SCOPE", "auto"))
            thread = source.thread_id or "none"
            lines.extend([
                "",
                t("gateway.status.matrix_scope_header"),
                t("gateway.status.matrix_scope_room", room=source.chat_name or source.chat_id),
                t("gateway.status.matrix_scope_room_id", room_id=source.chat_id),
                t("gateway.status.matrix_scope_thread", thread_id=thread),
                t("gateway.status.matrix_scope_mode", scope=scope),
                t(
                    "gateway.status.matrix_scope_key",
                    session_key=self._redact_matrix_session_key(session_key),
                ),
            ])
        lines.extend([
            "",
            t("gateway.status.platforms", platforms=', '.join(connected_platforms)),
        ])

        return "\n".join(lines)

    @staticmethod
    def _redact_matrix_session_key(session_key: str) -> str:
        """Return a stable Matrix session-key fingerprint for shared room status."""
        text = str(session_key or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

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

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command - list active agents and running tasks."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        from tools.process_registry import format_uptime_short, process_registry

        now = time.time()
        current_session_key = self._session_key_for_source(event.source)

        running_agents: dict = getattr(self, "_running_agents", {}) or {}
        running_started: dict = getattr(self, "_running_agents_ts", {}) or {}

        agent_rows: list[dict] = []
        for session_key, agent in running_agents.items():
            started = float(running_started.get(session_key, now))
            elapsed = max(0, int(now - started))
            is_pending = agent is _AGENT_PENDING_SENTINEL
            agent_rows.append(
                {
                    "session_key": session_key,
                    "elapsed": elapsed,
                    "state": t("gateway.agents.state_starting") if is_pending else t("gateway.agents.state_running"),
                    "session_id": "" if is_pending else str(getattr(agent, "session_id", "") or ""),
                    "model": "" if is_pending else str(getattr(agent, "model", "") or ""),
                }
            )

        agent_rows.sort(key=lambda row: row["elapsed"], reverse=True)

        running_processes: list[dict] = []
        try:
            running_processes = [
                p for p in process_registry.list_sessions()
                if p.get("status") == "running"
            ]
        except Exception:
            running_processes = []

        background_tasks = [
            t for t in (getattr(self, "_background_tasks", set()) or set())
            if hasattr(t, "done") and not t.done()
        ]

        lines = [
            t("gateway.agents.header"),
            "",
            t("gateway.agents.active_agents", count=len(agent_rows)),
        ]

        if agent_rows:
            for idx, row in enumerate(agent_rows[:12], 1):
                current = t("gateway.agents.this_chat") if row["session_key"] == current_session_key else ""
                sid = f" · `{row['session_id']}`" if row["session_id"] else ""
                model = f" · `{row['model']}`" if row["model"] else ""
                lines.append(
                    f"{idx}. `{row['session_key']}` · {row['state']} · "
                    f"{format_uptime_short(row['elapsed'])}{sid}{model}{current}"
                )
            if len(agent_rows) > 12:
                lines.append(t("gateway.agents.more", count=len(agent_rows) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.running_processes", count=len(running_processes)),
            ]
        )
        if running_processes:
            for proc in running_processes[:12]:
                cmd = " ".join(str(proc.get("command", "")).split())
                if len(cmd) > 90:
                    cmd = cmd[:87] + "..."
                lines.append(
                    f"- `{proc.get('session_id', '?')}` · "
                    f"{format_uptime_short(int(proc.get('uptime_seconds', 0)))} · `{cmd}`"
                )
            if len(running_processes) > 12:
                lines.append(t("gateway.agents.more", count=len(running_processes) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.async_jobs", count=len(background_tasks)),
            ]
        )

        if not agent_rows and not running_processes and not background_tasks:
            lines.append("")
            lines.append(t("gateway.agents.none"))

        return "\n".join(lines)

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop command - interrupt a running agent.

        When an agent is truly hung (blocked thread that never checks
        _interrupt_requested), the early intercept in _handle_message()
        handles /stop before this method is reached.  This handler fires
        only through normal command dispatch (no running agent) or as a
        fallback.  Force-clean the session lock in all cases for safety.

        The session is preserved so the user can continue the conversation.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            # Force-clean the sentinel so the session is unlocked.
            await self._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command_pending",
            )
            logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
            return EphemeralReply(t("gateway.stop.stopped_pending"))
        if agent:
            # Force-clean the session lock so a truly hung agent doesn't
            # keep it locked forever.
            await self._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command_handler",
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        # No run under the caller's own session key.  In a per-user thread
        # (thread_sessions_per_user=True) each participant is isolated even
        # inside one shared thread, so a run another user started lives under
        # a different key.  Authorized users should still be able to /stop it
        # (#bernard-thread-stop).  Fall back to interrupting any running
        # agent(s) that share this thread, gated on authorization.
        sibling_keys = self._sibling_thread_run_keys(source, session_key)
        if sibling_keys and self._is_user_authorized(source):
            for sibling_key in sibling_keys:
                await self._interrupt_and_clear_session(
                    sibling_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_STOP,
                    invalidation_reason="stop_command_thread_sibling",
                )
            logger.info(
                "STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
                session_key,
                len(sibling_keys),
                ", ".join(sibling_keys),
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        return t("gateway.stop.no_active")

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — surface and
        manually control failed/paused gateway adapters.

        Examples:
            ``/platform list``           — show connected + failed/paused platforms
            ``/platform pause whatsapp`` — stop the reconnect watcher hammering whatsapp
            ``/platform resume whatsapp`` — re-queue a paused platform for retry
        """
        text = (getattr(event, "content", "") or "").strip()
        # Strip the leading "/platform" (or "/PLATFORM") token if present
        parts = text.split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""

        # Resolve platform name (case-insensitive, value match)
        def _resolve_platform(name: str):
            if not name:
                return None
            for p in Platform.__members__.values():
                if p.value.lower() == name:
                    return p
            return None

        if action == "list":
            lines = ["**Gateway platforms**"]
            connected = sorted(p.value for p in self.adapters.keys())
            if connected:
                lines.append("Connected: " + ", ".join(connected))
            else:
                lines.append("Connected: (none)")
            failed = getattr(self, "_failed_platforms", {}) or {}
            if failed:
                for p, info in failed.items():
                    if info.get("paused"):
                        reason = info.get("pause_reason") or "paused"
                        lines.append(
                            f"  · {p.value} — PAUSED ({reason}). "
                            f"Resume with `/platform resume {p.value}`."
                        )
                    else:
                        attempts = info.get("attempts", 0)
                        lines.append(
                            f"  · {p.value} — retrying (attempt {attempts})"
                        )
            else:
                lines.append("Failed/paused: (none)")
            return "\n".join(lines)

        if action in {"pause", "resume"}:
            if not target:
                return f"Usage: /platform {action} <name>"
            platform = _resolve_platform(target)
            if platform is None:
                return f"Unknown platform: {target}"
            failed = getattr(self, "_failed_platforms", {}) or {}
            if action == "pause":
                if platform not in failed:
                    return (
                        f"{platform.value} is not in the retry queue "
                        f"(it's either connected or not enabled)."
                    )
                if failed[platform].get("paused"):
                    return f"{platform.value} is already paused."
                self._pause_failed_platform(platform, reason="paused via /platform pause")
                return (
                    f"✓ {platform.value} paused. "
                    f"Resume with `/platform resume {platform.value}` or "
                    f"`hermes gateway restart` to reset."
                )
            # action == "resume"
            if platform not in failed:
                return (
                    f"{platform.value} is not in the retry queue — "
                    f"nothing to resume."
                )
            if not failed[platform].get("paused"):
                return (
                    f"{platform.value} is already retrying — "
                    f"no resume needed."
                )
            self._resume_paused_platform(platform)
            return f"✓ {platform.value} resumed — retrying on next watcher tick."

        return (
            "Usage: /platform <list|pause|resume> [name]\n"
            "  /platform list — show platform status\n"
            "  /platform pause <name> — stop retrying a failing platform\n"
            "  /platform resume <name> — re-queue a paused platform"
        )

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /restart command - drain active work, then restart the gateway."""
        from gateway.run import _hermes_home
        # Defensive idempotency check: if the previous gateway process
        # recorded this same /restart (same platform + update_id) and the new
        # process is seeing it *again*, this is a re-delivery caused by PTB's
        # graceful-shutdown `get_updates` ACK failing on the way out ("Error
        # while calling `get_updates` one more time to mark all fetched
        # updates. Suppressing error to ensure graceful shutdown. When
        # polling for updates is restarted, updates may be received twice."
        # in gateway.log).  Ignoring the stale redelivery prevents a
        # self-perpetuating restart loop where every fresh gateway
        # re-processes the same /restart command and immediately restarts
        # again.
        if self._is_stale_restart_redelivery(event):
            logger.info(
                "Ignoring redelivered /restart (platform=%s, update_id=%s) — "
                "already processed by a previous gateway instance.",
                event.source.platform.value if event.source and event.source.platform else "?",
                event.platform_update_id,
            )
            return ""

        if self._restart_requested or self._draining:
            count = self._running_agent_count()
            if count:
                return t("gateway.draining", count=count)
            return EphemeralReply(t("gateway.restart.in_progress"))

        # Save the requester's routing info so the new gateway process can
        # notify them once it comes back online.
        try:
            notify_data = {
                "platform": event.source.platform.value if event.source.platform else None,
                "chat_id": event.source.chat_id,
                "chat_type": event.source.chat_type,
            }
            if event.source.thread_id:
                notify_data["thread_id"] = event.source.thread_id
            if event.message_id:
                notify_data["message_id"] = event.message_id
            if event.source is not None:
                try:
                    self._restart_command_source = dataclasses.replace(
                        event.source,
                        message_id=str(event.message_id)
                        if event.message_id is not None
                        else event.source.message_id,
                    )
                except Exception:
                    self._restart_command_source = event.source
            atomic_json_write(
                _hermes_home / ".restart_notify.json",
                notify_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart notify file: %s", e)

        # Record the triggering platform + update_id in a dedicated dedup
        # marker.  Unlike .restart_notify.json (which gets unlinked once the
        # new gateway sends the "gateway restarted" notification), this
        # marker persists so the new gateway can still detect a delayed
        # /restart redelivery from Telegram.  Overwritten on every /restart.
        try:
            dedup_data = {
                "platform": event.source.platform.value if event.source.platform else None,
                "requested_at": time.time(),
            }
            if event.platform_update_id is not None:
                dedup_data["update_id"] = event.platform_update_id
            atomic_json_write(
                _hermes_home / ".restart_last_processed.json",
                dedup_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart dedup marker: %s", e)

        active_agents = self._running_agent_count()
        # When running under a service manager (systemd/launchd) or inside a
        # Docker/Podman container, use the service restart path: exit with
        # code 75 so the service manager / container restart policy restarts
        # us.  The detached subprocess approach (setsid + bash) doesn't work
        # under systemd (KillMode=mixed kills the cgroup) or Docker (tini
        # exits when the gateway dies, taking the detached helper with it).
        # systemd sets INVOCATION_ID; launchd sets XPC_SERVICE_NAME to the
        # job label.  Without the launchd check, macOS /restart takes the
        # detached path and exits 0, which KeepAlive.SuccessfulExit=false
        # treats as a deliberate stop — the gateway stays dead until next
        # login.  Interactive macOS shells inherit XPC_SERVICE_NAME=0, so
        # "0" must count as not-under-launchd.
        _under_service = bool(os.environ.get("INVOCATION_ID")) or os.environ.get(
            "XPC_SERVICE_NAME", "0"
        ) not in ("", "0")
        _in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        if _under_service or _in_container:
            self.request_restart(detached=False, via_service=True)
        else:
            self.request_restart(detached=True, via_service=False)
        if active_agents:
            return t("gateway.draining", count=active_agents)
        return EphemeralReply(t("gateway.restart.restarting"))

    async def _handle_version_command(self, event: MessageEvent) -> str:
        """Handle /version — show the running Hermes Agent version."""
        from hermes_cli.banner import format_banner_version_label

        return format_banner_version_label()

    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        from gateway.run import _telegramize_command_mentions
        from hermes_cli.commands import gateway_help_lines
        lines = [
            t("gateway.help.header"),
            *gateway_help_lines(),
        ]
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                lines.append(t("gateway.help.skill_header", count=len(skill_cmds)))
                # Show first 10, then point to /commands for the rest
                sorted_cmds = sorted(skill_cmds)
                for cmd in sorted_cmds[:10]:
                    lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
                if len(sorted_cmds) > 10:
                    lines.append(t("gateway.help.more_use_commands", count=len(sorted_cmds) - 10))
        except Exception:
            pass
        return _telegramize_command_mentions(
            "\n".join(lines),
            getattr(getattr(event, "source", None), "platform", None),
        )

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        from gateway.run import _telegramize_command_mentions
        from hermes_cli.commands import gateway_help_lines

        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                requested_page = int(raw_args)
            except ValueError:
                return t("gateway.commands.usage")
        else:
            requested_page = 1

        # Build combined entry list: built-in commands + skill commands
        entries = list(gateway_help_lines())
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                entries.append("")
                entries.append(t("gateway.commands.skill_header"))
                for cmd in sorted(skill_cmds):
                    desc = skill_cmds[cmd].get("description", "").strip() or t("gateway.commands.default_desc")
                    entries.append(f"`{cmd}` — {desc}")
        except Exception:
            pass

        if not entries:
            return t("gateway.commands.none")

        from gateway.config import Platform
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        total_pages = max(1, (len(entries) + page_size - 1) // page_size)
        page = max(1, min(requested_page, total_pages))
        start = (page - 1) * page_size
        page_entries = entries[start:start + page_size]

        lines = [
            t("gateway.commands.header", total=len(entries), page=page, total_pages=total_pages),
            "",
            *page_entries,
        ]
        if total_pages > 1:
            nav_parts = []
            if page > 1:
                nav_parts.append(t("gateway.commands.nav_prev", page=page - 1))
            if page < total_pages:
                nav_parts.append(t("gateway.commands.nav_next", page=page + 1))
            lines.extend(["", " | ".join(nav_parts)])
        if page != requested_page:
            lines.append(t("gateway.commands.out_of_range", requested=requested_page, page=page))
        return _telegramize_command_mentions(
            "\n".join(lines),
            getattr(getattr(event, "source", None), "platform", None),
        )

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model.

        Supports:
          /model                              — interactive picker (Telegram/Discord) or text list
          /model <name>                       — switch model (persists by default)
          /model <name> --session             — switch for this session only
          /model <name> --global              — switch and persist (explicit)
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model
        """
        from gateway.run import _hermes_home, _load_gateway_config
        import yaml
        from hermes_cli.model_switch import (
            switch_model as _switch_model, parse_model_flags,
            resolve_persist_behavior,
            list_authenticated_providers,
            list_picker_providers,
        )
        from hermes_cli.providers import get_label

        raw_args = event.get_command_args().strip()

        # Parse --provider, --global, --session, and --refresh flags
        (
            model_input,
            explicit_provider,
            is_global_flag,
            force_refresh,
            is_session,
        ) = parse_model_flags(raw_args)
        persist_global = resolve_persist_behavior(is_global_flag, is_session)

        # --refresh: bust the disk cache so the picker shows live data.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
            except Exception:
                pass

        # Read current model/provider from config
        current_model = ""
        current_provider = "openrouter"
        current_base_url = ""
        current_api_key = ""
        user_provs = None
        custom_provs = None
        config_path = _hermes_home / "config.yaml"
        try:
            cfg = _load_gateway_config()
            if cfg:
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_model = model_cfg.get("default", "")
                    current_provider = model_cfg.get("provider", current_provider)
                    current_base_url = model_cfg.get("base_url", "")
                user_provs = cfg.get("providers")
                try:
                    from hermes_cli.config import get_compatible_custom_providers
                    custom_provs = get_compatible_custom_providers(cfg)
                except Exception:
                    custom_provs = cfg.get("custom_providers")
        except Exception:
            pass

        # Check for session override
        source = event.source
        # Normalize the source the same way a normal message turn does
        # (Telegram DM topic recovery) before deriving the override key, so
        # the override is stored under the key the next message turn reads
        # (#30479).
        source = await asyncio.to_thread(self._normalize_source_for_session_key, source)
        session_key = self._session_key_for_source(source)
        override = self._session_model_overrides.get(session_key, {})
        if override:
            current_model = override.get("model", current_model)
            current_provider = override.get("provider", current_provider)
            current_base_url = override.get("base_url", current_base_url)
            current_api_key = override.get("api_key", current_api_key)

        # No args: show interactive picker (Telegram/Discord) or text list
        if not model_input and not explicit_provider:
            # Try interactive picker if the platform supports it
            adapter = self.adapters.get(source.platform)
            has_picker = (
                adapter is not None
                and getattr(type(adapter), "send_model_picker", None) is not None
            )

            if has_picker:
                try:
                    # Offload blocking provider-listing (can fall through to a
                    # synchronous urllib HTTP fetch on a stale cache) off the
                    # event loop so the gateway doesn't freeze. See #41289.
                    providers = await asyncio.to_thread(
                        list_picker_providers,
                        current_provider=current_provider,
                        current_base_url=current_base_url,
                        current_model=current_model,
                        user_providers=user_provs,
                        custom_providers=custom_provs,
                        max_models=50,
                        include_moa=True,
                    )
                except Exception:
                    providers = []

                if providers:
                    # Build a callback closure for when the user picks a model.
                    # Captures self + locals needed for the switch logic.
                    _self = self
                    _session_key = session_key
                    _cur_model = current_model
                    _cur_provider = current_provider
                    _cur_base_url = current_base_url
                    _cur_api_key = current_api_key

                    async def _on_model_selected(
                        _chat_id: str, model_id: str, provider_slug: str
                    ) -> str:
                        """Perform the model switch and return confirmation text."""
                        skew_error = _model_switch_skew_guard()
                        if skew_error:
                            return skew_error
                        # Offload the switch off the event loop — switch_model()
                        # can fall through to a synchronous models.dev HTTP fetch
                        # (requests.get, 15s timeout) on a cold/expired cache,
                        # which freezes the gateway otherwise. See #20525, #41289.
                        result = await asyncio.to_thread(
                            _switch_model,
                            raw_input=model_id,
                            current_provider=_cur_provider,
                            current_model=_cur_model,
                            current_base_url=_cur_base_url,
                            current_api_key=_cur_api_key,
                            is_global=persist_global,
                            explicit_provider=provider_slug,
                            user_providers=user_provs,
                            custom_providers=custom_provs,
                        )
                        if not result.success:
                            return t("gateway.model.error_prefix", error=result.error_message)

                        try:
                            from hermes_cli.context_switch_guard import (
                                enrich_model_switch_warnings_for_gateway,
                            )

                            enrich_model_switch_warnings_for_gateway(
                                result,
                                _self,
                                session_key=_session_key,
                                source=event.source,
                                custom_providers=custom_provs,
                                load_gateway_config=_load_gateway_config,
                            )
                        except Exception as exc:
                            logger.debug("preflight-compression switch warning failed: %s", exc)

                        # Update cached agent in-place
                        cached_entry = None
                        _cache_lock = getattr(_self, "_agent_cache_lock", None)
                        _cache = getattr(_self, "_agent_cache", None)
                        if _cache_lock and _cache is not None:
                            with _cache_lock:
                                cached_entry = _cache.get(_session_key)
                        if cached_entry and cached_entry[0] is not None:
                            try:
                                cached_entry[0].switch_model(
                                    new_model=result.new_model,
                                    new_provider=result.target_provider,
                                    api_key=result.api_key,
                                    base_url=result.base_url,
                                    api_mode=result.api_mode,
                                )
                            except Exception as exc:
                                # The in-place swap rolled the agent back to the
                                # OLD working model/client and re-raised.  Abort
                                # the rest of the commit: do NOT persist the
                                # failed model to the DB, do NOT set a session
                                # override pointing at the broken model, and do
                                # NOT evict the working cached agent.  Otherwise
                                # the next message rebuilds a dead agent from the
                                # broken override and the conversation is lost
                                # (#50163).  A failed switch must be a no-op.
                                logger.warning(
                                    "Picker model switch failed for cached agent: %s", exc
                                )
                                return t(
                                    "gateway.model.error_prefix",
                                    error=(
                                        f"Model switch to {result.new_model} failed ({exc}); "
                                        f"staying on {_cur_model}."
                                    ),
                                )

                        # Persist the new model to the session DB so the
                        # dashboard shows the updated model (#34850).
                        _sess_db = getattr(_self, "_session_db", None)
                        if _sess_db is not None:
                            try:
                                _sess_entry = await _self.async_session_store.get_or_create_session(
                                    event.source
                                )
                                await _sess_db.update_session_model(
                                    _sess_entry.session_id, result.new_model
                                )
                            except Exception as exc:
                                logger.debug(
                                    "Failed to persist model switch to DB: %s", exc
                                )

                        # Store model note + session override
                        if not hasattr(_self, "_pending_model_notes"):
                            _self._pending_model_notes = {}
                        _self._pending_model_notes[_session_key] = (
                            f"[Note: model was just switched from {_cur_model} to {result.new_model} "
                            f"via {result.provider_label or result.target_provider}. "
                            f"Adjust your self-identification accordingly.]"
                        )
                        _self._session_model_overrides[_session_key] = {
                            "model": result.new_model,
                            "provider": result.target_provider,
                            "api_key": result.api_key,
                            "base_url": result.base_url,
                            "api_mode": result.api_mode,
                        }

                        # Write-through the non-secret parts to the session
                        # store so the picked model survives a gateway restart
                        # (api_key is never persisted).
                        try:
                            await _self.async_session_store.set_model_override(
                                _session_key,
                                _self._session_model_overrides[_session_key],
                            )
                        except Exception:
                            logger.debug(
                                "Failed to persist session model override",
                                exc_info=True,
                            )

                        # Evict cached agent so the next turn creates a fresh
                        # agent from the override rather than relying on the
                        # stale cache signature to trigger a rebuild.
                        _self._evict_cached_agent(_session_key)

                        # Persist to config (default) unless --session opted out,
                        # mirroring the text /model command path above so a picked
                        # model survives across sessions like a typed one (#49066).
                        if persist_global:
                            try:
                                from hermes_cli.config import persist_model_switch_config

                                persist_model_switch_config(
                                    new_model=result.new_model,
                                    target_provider=result.target_provider,
                                    base_url=result.base_url,
                                    api_mode=result.api_mode,
                                )
                            except Exception as e:
                                logger.warning("Failed to persist model switch: %s", e)

                        # Build confirmation text
                        plabel = result.provider_label or result.target_provider
                        lines = [t("gateway.model.switched", model=result.new_model)]
                        lines.append(t("gateway.model.provider_label", provider=plabel))
                        mi = result.model_info
                        from hermes_cli.model_switch import resolve_display_context_length
                        _sw_config_ctx = None
                        try:
                            _sw_cfg = _load_gateway_config()
                            _sw_model_cfg = _sw_cfg.get("model", {})
                            if isinstance(_sw_model_cfg, dict):
                                _sw_raw = _sw_model_cfg.get("context_length")
                                if _sw_raw is not None:
                                    _sw_config_ctx = int(_sw_raw)
                        except Exception:
                            pass
                        ctx = resolve_display_context_length(
                            result.new_model,
                            result.target_provider,
                            base_url=result.base_url or current_base_url or "",
                            api_key=result.api_key or current_api_key or "",
                            model_info=mi,
                            custom_providers=custom_provs,
                            config_context_length=_sw_config_ctx,
                        )
                        if ctx:
                            lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
                        if mi:
                            if mi.max_output:
                                lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
                            lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))
                        if result.warning_message:
                            lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))
                        if persist_global:
                            lines.append(t("gateway.model.saved_global"))
                        else:
                            lines.append(t("gateway.model.session_only_hint"))
                        return "\n".join(lines)

                    metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                    result = await adapter.send_model_picker(
                        chat_id=source.chat_id,
                        providers=providers,
                        current_model=current_model,
                        current_provider=current_provider,
                        session_key=session_key,
                        on_model_selected=_on_model_selected,
                        metadata=metadata,
                    )
                    if result.success:
                        return None  # Picker sent — adapter handles the response

            # Fallback: text list (for platforms without picker or if picker failed)
            provider_label = get_label(current_provider)
            lines = [t("gateway.model.current_label", model=current_model or "unknown", provider=provider_label), ""]

            try:
                # Offload blocking provider-listing off the event loop so the
                # gateway doesn't freeze on a stale-cache HTTP fetch. See #41289.
                providers = await asyncio.to_thread(
                    list_authenticated_providers,
                    current_provider=current_provider,
                    current_base_url=current_base_url,
                    current_model=current_model,
                    user_providers=user_provs,
                    custom_providers=custom_provs,
                    max_models=5,
                )
                for p in providers:
                    tag = t("gateway.model.current_tag") if p["is_current"] else ""
                    lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
                    if p["models"]:
                        model_strs = ", ".join(f"`{m}`" for m in p["models"])
                        extra = t("gateway.model.more_models_suffix", count=p["total_models"] - len(p["models"])) if p["total_models"] > len(p["models"]) else ""
                        lines.append(f"  {model_strs}{extra}")
                    elif p.get("api_url"):
                        lines.append(f"  `{p['api_url']}`")
                    lines.append("")
            except Exception:
                pass

            lines.append(t("gateway.model.usage_switch_model"))
            lines.append(t("gateway.model.usage_switch_provider"))
            lines.append(t("gateway.model.usage_persist"))
            return "\n".join(lines)

        # Perform the switch
        skew_error = _model_switch_skew_guard()
        if skew_error:
            return skew_error
        # Offload the switch off the event loop — switch_model() can fall
        # through to a synchronous models.dev HTTP fetch (requests.get, 15s
        # timeout) on a cold/expired cache, which freezes the gateway
        # otherwise. See #20525, #41289.
        result = await asyncio.to_thread(
            _switch_model,
            raw_input=model_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key=current_api_key,
            is_global=persist_global,
            explicit_provider=explicit_provider,
            user_providers=user_provs,
            custom_providers=custom_provs,
        )

        if not result.success:
            return t("gateway.model.error_prefix", error=result.error_message)

        try:
            from hermes_cli.context_switch_guard import (
                enrich_model_switch_warnings_for_gateway,
            )

            enrich_model_switch_warnings_for_gateway(
                result,
                self,
                session_key=session_key,
                source=source,
                custom_providers=custom_provs,
                load_gateway_config=_load_gateway_config,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)

        async def _finish_switch() -> str:
            """Apply the resolved switch (agent, session, config) and build the reply."""
            # If there's a cached agent, update it in-place
            cached_entry = None
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached_entry = _cache.get(session_key)

            if cached_entry and cached_entry[0] is not None:
                try:
                    cached_entry[0].switch_model(
                        new_model=result.new_model,
                        new_provider=result.target_provider,
                        api_key=result.api_key,
                        base_url=result.base_url,
                        api_mode=result.api_mode,
                    )
                except Exception as exc:
                    # In-place swap rolled the agent back to the OLD working
                    # model/client and re-raised.  Abort the commit: skip DB
                    # persist, session override, cache eviction, and config
                    # write so a failed switch is a no-op rather than a dead
                    # conversation (#50163).  Without this early return the
                    # next message rebuilds a broken agent from the override.
                    logger.warning("In-place model switch failed for cached agent: %s", exc)
                    return t(
                        "gateway.model.error_prefix",
                        error=(
                            f"Model switch to {result.new_model} failed ({exc}); "
                            f"staying on {current_model}."
                        ),
                    )

            # Persist the new model to the session DB so the dashboard
            # shows the updated model (#34850).
            _sess_db = getattr(self, "_session_db", None)
            if _sess_db is not None:
                try:
                    _sess_entry = await self.async_session_store.get_or_create_session(source)
                    # If this session was auto-reset, consume the flag so the
                    # next regular message's cleanup does not wipe the model
                    # override just stored below (Closes #48031).
                    if getattr(_sess_entry, "was_auto_reset", False):
                        _sess_entry.was_auto_reset = False
                    await _sess_db.update_session_model(
                        _sess_entry.session_id, result.new_model
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to persist model switch to DB: %s", exc
                    )

            # Store a note to prepend to the next user message so the model
            # knows about the switch (avoids system messages mid-history).
            if not hasattr(self, "_pending_model_notes"):
                self._pending_model_notes = {}
            self._pending_model_notes[session_key] = (
                f"[Note: model was just switched from {current_model} to {result.new_model} "
                f"via {result.provider_label or result.target_provider}. "
                f"Adjust your self-identification accordingly.]"
            )

            # Store session override so next agent creation uses the new model
            self._session_model_overrides[session_key] = {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            }

            # Write-through the non-secret parts (model/provider/base_url) to
            # the session store so the override survives a gateway restart.
            # api_key/api_mode are never persisted — they are re-resolved via
            # runtime provider resolution on rehydration.
            try:
                await self.async_session_store.set_model_override(
                    session_key,
                    self._session_model_overrides[session_key],
                )
            except Exception:
                logger.debug(
                    "Failed to persist session model override", exc_info=True
                )

            # Evict cached agent so the next turn creates a fresh agent from the
            # override rather than relying on cache signature mismatch detection.
            self._evict_cached_agent(session_key)

            # Persist to config (default) unless --session opted out
            if persist_global:
                try:
                    from hermes_cli.config import persist_model_switch_config

                    persist_model_switch_config(
                        new_model=result.new_model,
                        target_provider=result.target_provider,
                        base_url=result.base_url,
                        api_mode=result.api_mode,
                    )
                except Exception as e:
                    logger.warning("Failed to persist model switch: %s", e)

            # Build confirmation message with full metadata
            provider_label = result.provider_label or result.target_provider
            lines = [t("gateway.model.switched", model=result.new_model)]
            lines.append(t("gateway.model.provider_label", provider=provider_label))

            # Context: always resolve via the provider-aware chain so Codex OAuth,
            # Copilot, and Nous-enforced caps win over the raw models.dev entry.
            mi = result.model_info
            from hermes_cli.model_switch import resolve_display_context_length
            _sw2_config_ctx = None
            try:
                _sw2_cfg = _load_gateway_config()
                _sw2_model_cfg = _sw2_cfg.get("model", {})
                if isinstance(_sw2_model_cfg, dict):
                    _sw2_raw = _sw2_model_cfg.get("context_length")
                    if _sw2_raw is not None:
                        _sw2_config_ctx = int(_sw2_raw)
            except Exception:
                pass
            ctx = resolve_display_context_length(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=mi,
                custom_providers=custom_provs,
                config_context_length=_sw2_config_ctx,
            )
            if ctx:
                lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
            if mi:
                if mi.max_output:
                    lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
                lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))

            # Cache notice
            cache_enabled = (
                (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
                or result.api_mode == "anthropic_messages"
            )
            if cache_enabled:
                lines.append(t("gateway.model.prompt_caching_enabled"))

            if result.warning_message:
                lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))

            if persist_global:
                lines.append(t("gateway.model.saved_global"))
            else:
                lines.append(t("gateway.model.session_only_hint"))

            return "\n".join(lines)

        # Expensive-model confirmation gate (typed /model <name> path).
        # The pickers (Telegram/Discord inline keyboards, TUI, dashboard)
        # already confirm via their own UI affordances; this covers the
        # direct text command, which previously bypassed the guard.
        # expensive_model_warning() may hit models.dev or a /models endpoint
        # on a cache miss, so run it off the event loop.
        _cost_warning = None
        try:
            from hermes_cli.model_cost_guard import expensive_model_warning

            _cost_warning = await asyncio.to_thread(
                expensive_model_warning,
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=result.model_info,
            )
        except Exception:
            _cost_warning = None
        if _cost_warning is not None:
            async def _on_cost_confirm(choice: str) -> str:
                if choice == "cancel":
                    return (
                        f"🟡 Model switch cancelled. Current model unchanged "
                        f"({current_model or 'unknown'})."
                    )
                # "once" and "always" both proceed — there is no persistent
                # opt-out for the cost guard (each expensive switch should be
                # an explicit decision).
                return await _finish_switch()

            _p = self._typed_command_prefix_for(event.source.platform)
            return await self._request_slash_confirm(
                event=event,
                command="model",
                title="Expensive Model Warning",
                message=(
                    f"⚠️ **Expensive Model Warning**\n\n{_cost_warning.message}\n\n"
                    f"_Text fallback: reply `{_p}approve` to switch or `{_p}cancel` to keep "
                    "the current model._"
                ),
                handler=_on_cost_confirm,
            )

        return await _finish_switch()

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime command in the gateway.

        Same surface as the CLI handler in cli.py:
            /codex-runtime                  — show current state
            /codex-runtime auto             — Hermes default runtime
            /codex-runtime codex_app_server — codex subprocess runtime
            /codex-runtime on / off         — synonyms

        On change, the cached agent for this session is evicted so the next
        message creates a fresh AIAgent with the new api_mode wired in
        (avoids prompt-cache invalidation mid-session)."""
        from hermes_cli import codex_runtime_switch as crs

        raw_args = event.get_command_args().strip() if event else ""
        new_value, errors = crs.parse_args(raw_args)
        if errors:
            return "❌ " + "\n❌ ".join(errors)

        # Load + persist via the same helpers used for /model and /yolo
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            return f"❌ Could not load config: {exc}"
        cfg = load_config()

        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )

        # On a real change, evict the cached agent so the new runtime takes
        # effect on the next message rather than waiting for cache TTL.
        if result.success and new_value is not None and result.requires_new_session:
            try:
                session_key = self._session_key_for_source(event.source)
                self._evict_cached_agent(session_key)
            except Exception:
                logger.debug("could not evict cached agent after codex-runtime change",
                             exc_info=True)

        prefix = "✓" if result.success else "✗"
        return f"{prefix} {result.message}"

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        from gateway.run import _hermes_home, _load_gateway_config
        from hermes_constants import display_hermes_home

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / 'config.yaml'

        try:
            config = _load_gateway_config()
            personalities = cfg_get(config, "agent", "personalities", default={})
        except Exception:
            config = {}
            personalities = {}

        if not personalities:
            return t("gateway.personality.none_configured", path=display_hermes_home())

        if not args:
            lines = [t("gateway.personality.header")]
            lines.append(t("gateway.personality.none_option"))
            for name, prompt in personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(t("gateway.personality.item", name=name, preview=preview))
            lines.append(t("gateway.personality.usage"))
            return "\n".join(lines)

        def _resolve_prompt(value):
            if isinstance(value, dict):
                parts = [value.get("system_prompt", "")]
                if value.get("tone"):
                    parts.append(f'Tone: {value["tone"]}')
                if value.get("style"):
                    parts.append(f'Style: {value["style"]}')
                return "\n".join(p for p in parts if p)
            return str(value)

        if args in {"none", "default", "neutral"}:
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = ""
                atomic_config_write(config_path, config)
            except Exception as e:
                return t("gateway.personality.save_failed", error=str(e))
            self._ephemeral_system_prompt = ""
            return t("gateway.personality.cleared")
        elif args in personalities:
            new_prompt = _resolve_prompt(personalities[args])

            # Write to config.yaml, same pattern as CLI save_config_value.
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = new_prompt
                atomic_config_write(config_path, config)
            except Exception as e:
                return t("gateway.personality.save_failed", error=str(e))

            # Update in-memory so it takes effect on the very next message.
            self._ephemeral_system_prompt = new_prompt

            return t("gateway.personality.set_to", name=args)

        available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
        return t("gateway.personality.unknown", name=args, available=available)

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_msg = history[i].get("content", "")
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

    async def _handle_goal_command(self, event: "MessageEvent") -> str:
        """Handle /goal for gateway platforms.

        Subcommands: ``/goal`` / ``/goal status`` / ``/goal pause`` /
        ``/goal resume`` / ``/goal clear``. Any other text becomes the
        new goal.

        Setting a new goal queues the goal text as the next turn so the
        agent starts working on it immediately — the post-turn
        continuation hook then takes over from there.
        """
        args = (event.get_command_args() or "").strip()
        lower = args.lower()

        mgr, session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")

        if not args or lower == "status":
            return mgr.status_line()

        # /goal show → print the active goal's completion contract
        if lower == "show":
            return f"{mgr.status_line()}\n{mgr.render_contract()}"

        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                return t("gateway.goal.no_goal_set")
            try:
                adapter = self.adapters.get(event.source.platform) if event.source else None
                _quick_key = self._session_key_for_source(event.source) if event.source else None
                if adapter and _quick_key:
                    self._clear_goal_pending_continuations(_quick_key, adapter)
            except Exception as exc:
                logger.debug("goal pause: pending continuation cleanup failed: %s", exc)
            return t("gateway.goal.paused", goal=state.goal)

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return t("gateway.goal.no_resume")
            return t("gateway.goal.resumed", goal=state.goal)

        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            try:
                adapter = self.adapters.get(event.source.platform) if event.source else None
                _quick_key = self._session_key_for_source(event.source) if event.source else None
                if adapter and _quick_key:
                    self._clear_goal_pending_continuations(_quick_key, adapter)
            except Exception as exc:
                logger.debug("goal clear: pending continuation cleanup failed: %s", exc)
            return t("gateway.goal_cleared") if had else t("gateway.no_active_goal")

        # /goal wait <pid> [reason] — park the loop on a background process.
        if lower == "wait" or lower.startswith("wait "):
            wait_arg = args[len("wait"):].strip()
            if not wait_arg:
                return "Usage: /goal wait <pid> [reason]"
            wtokens = wait_arg.split(None, 1)
            try:
                pid = int(wtokens[0])
            except ValueError:
                return "/goal wait: <pid> must be an integer process id."
            reason = wtokens[1].strip() if len(wtokens) > 1 else ""
            try:
                mgr.wait_on(pid, reason=reason)
            except (RuntimeError, ValueError) as exc:
                return f"/goal wait: {exc}"
            rtxt = f" ({reason})" if reason else ""
            return f"⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits."

        # /goal unwait — clear the wait barrier.
        if lower == "unwait":
            if mgr.stop_waiting():
                return "▶ Wait barrier cleared — goal loop resumes."
            return "No wait barrier set."

        # /goal draft <objective> → draft a structured completion contract,
        # then set it. The aux LLM call is sync; run it off the event loop.
        draft_contract_obj = None
        if lower.startswith("draft"):
            objective = args[len("draft"):].strip()
            if not objective:
                return "Usage: /goal draft <objective in plain language>"
            try:
                import asyncio
                from hermes_cli.goals import draft_contract

                draft_contract_obj = await asyncio.get_running_loop().run_in_executor(
                    None, draft_contract, objective
                )
            except Exception as exc:
                logger.debug("goal draft failed: %s", exc)
                draft_contract_obj = None
            args = objective  # the goal text is the objective
            contract = draft_contract_obj
        else:
            # Inline `field: value` lines parse into a completion contract;
            # the remaining prose is the goal headline. Plain free-form goals
            # (no such lines) behave exactly as before.
            from hermes_cli.goals import parse_contract

            headline, parsed = parse_contract(args)
            args = headline or args
            contract = parsed if not parsed.is_empty() else None

        # Otherwise — treat the remaining text as the new goal.
        try:
            state = mgr.set(args, contract=contract)
        except ValueError as exc:
            return t("gateway.goal.invalid", error=str(exc))

        # Queue the goal text as an immediate first turn so the agent
        # starts making progress. The post-turn hook takes over after.
        adapter = self.adapters.get(event.source.platform) if event.source else None
        _quick_key = self._session_key_for_source(event.source) if event.source else None
        if adapter and _quick_key:
            try:
                kickoff_event = MessageEvent(
                    text=state.goal,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                )
                self._enqueue_fifo(_quick_key, kickoff_event, adapter)
            except Exception as exc:
                logger.debug("goal kickoff enqueue failed: %s", exc)

        base = t("gateway.goal.set", budget=state.max_turns, goal=state.goal)
        if state.has_contract():
            return f"{base}\nCompletion contract:\n{state.contract.render_block()}"
        if lower.startswith("draft"):
            # Drafting was requested but the aux model couldn't produce one.
            return f"{base}\n(Couldn't draft a contract — running as a free-form goal.)"
        return base

    async def _handle_subgoal_command(self, event: "MessageEvent") -> str:
        """Handle /subgoal for gateway platforms (mirror of CLI handler).

        Subgoals are extra criteria appended to the active goal mid-loop.
        They modify state read at the next turn boundary, so this is safe
        to invoke while the agent is running.
        """
        args = (event.get_command_args() or "").strip()
        mgr, _session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")
        if not mgr.has_goal():
            return "No active goal. Set one with /goal <text>."

        # No args → list current subgoals.
        if not args:
            return f"{mgr.status_line()}\n{mgr.render_subgoals()}"

        tokens = args.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if verb == "remove":
            if not rest:
                return "Usage: /subgoal remove <n>"
            try:
                idx = int(rest.split()[0])
            except ValueError:
                return "/subgoal remove: <n> must be an integer (1-based index)."
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                return f"/subgoal remove: {exc}"
            return f"✓ Removed subgoal {idx}: {removed}"

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                return f"/subgoal clear: {exc}"
            if prev:
                return f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}."
            return "No subgoals to clear."

        try:
            text = mgr.add_subgoal(args)
        except (ValueError, RuntimeError) as exc:
            return f"/subgoal: {exc}"
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return f"✓ Added subgoal {idx}: {text}"

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

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        from gateway.run import _home_target_env_var, _home_thread_env_var
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id

        env_key = _home_target_env_var(platform_name)
        thread_env_key = _home_thread_env_var(platform_name)
        thread_id = source.thread_id

        # Save to .env so it persists across restarts
        try:
            from hermes_cli.config import save_env_value
            save_env_value(env_key, str(chat_id))
            # Keep thread/topic routing explicit and clear stale values when
            # /sethome is run from the parent chat instead of a thread.
            save_env_value(thread_env_key, str(thread_id or ""))
        except Exception as e:
            return t("gateway.set_home.save_failed", error=e)

        # Keep the running gateway config in sync too. The pre-restart
        # notification path reads self.config before the process reloads env.
        if source.platform:
            platform_config = self.config.platforms.setdefault(
                source.platform,
                PlatformConfig(enabled=True),
            )
            platform_config.home_channel = HomeChannel(
                platform=source.platform,
                chat_id=str(chat_id),
                name=chat_name,
                thread_id=str(thread_id) if thread_id else None,
            )

        return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id
        platform = event.source.platform
        voice_key = self._voice_key(platform, chat_id)

        adapter = self.adapters.get(platform)

        if args in {"on", "enable"}:
            self._voice_mode[voice_key] = "voice_only"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
            return t("gateway.voice.enabled_voice_only")
        elif args in {"off", "disable"}:
            self._voice_mode[voice_key] = "off"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
            return t("gateway.voice.disabled_text")
        elif args == "tts":
            self._voice_mode[voice_key] = "all"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
            return t("gateway.voice.tts_enabled")
        elif args in {"channel", "join"}:
            return await self._handle_voice_channel_join(event)
        elif args == "leave":
            return await self._handle_voice_channel_leave(event)
        elif args == "status":
            mode = self._voice_mode.get(voice_key, "off")
            labels = {
                "off": t("gateway.voice.label_off"),
                "voice_only": t("gateway.voice.label_voice_only"),
                "all": t("gateway.voice.label_all"),
            }
            # Append voice channel info if connected
            adapter = self.adapters.get(event.source.platform)
            guild_id = self._get_guild_id(event)
            if guild_id and hasattr(adapter, "get_voice_channel_info"):
                info = adapter.get_voice_channel_info(guild_id)
                if info:
                    lines = [
                        t("gateway.voice.status_mode", label=labels.get(mode, mode)),
                        t("gateway.voice.status_channel", channel=info['channel_name']),
                        t("gateway.voice.status_participants", count=info['member_count']),
                    ]
                    for m in info["members"]:
                        status = t("gateway.voice.speaking") if m.get("is_speaking") else ""
                        lines.append(t("gateway.voice.status_member", name=m['display_name'], status=status))
                    return "\n".join(lines)
            return t("gateway.voice.status_mode", label=labels.get(mode, mode))
        else:
            # Toggle: off → on, on/all → off
            current = self._voice_mode.get(voice_key, "off")
            if current == "off":
                self._voice_mode[voice_key] = "voice_only"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)
                toggle_line = t("gateway.voice.enabled_short")
            else:
                self._voice_mode[voice_key] = "off"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
                toggle_line = t("gateway.voice.disabled_short")
            # Bare /voice still toggles, but append an explainer so users
            # discover the on/off/tts/status subcommands (and, on Discord,
            # live voice-channel join/leave). The toggle result is shown
            # first via the {toggle} placeholder.
            supports_voice_channels = adapter is not None and hasattr(
                adapter, "join_voice_channel"
            )
            channels = (
                t("gateway.voice.help_channels") if supports_voice_channels else ""
            )
            return t("gateway.voice.help", toggle=toggle_line, channels=channels)

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from gateway.run import _hermes_home
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        # Read checkpoint config from config.yaml
        cp_cfg = {}
        try:
            import yaml as _y
            _cfg_path = _hermes_home / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, encoding="utf-8") as _f:
                    _data = _y.safe_load(_f) or {}
                cp_cfg = _data.get("checkpoints", {})
                if isinstance(cp_cfg, bool):
                    cp_cfg = {"enabled": cp_cfg}
        except Exception:
            pass

        if not cp_cfg.get("enabled", False):
            return t("gateway.rollback.not_enabled")

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_cfg.get("max_snapshots", 50),
            max_total_size_mb=cp_cfg.get("max_total_size_mb", 500),
            max_file_size_mb=cp_cfg.get("max_file_size_mb", 10),
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

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return t("gateway.background.usage")

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        event_message_id = self._reply_anchor_for_event(event)

        # Forward image/audio attachments so the background agent can see them.
        media_urls = list(event.media_urls) if event.media_urls else []
        media_types = list(event.media_types) if event.media_types else []

        # Fire-and-forget the background task
        _task = asyncio.create_task(
            self._run_background_task(
                prompt,
                source,
                task_id,
                event_message_id=event_message_id,
                media_urls=media_urls,
                media_types=media_types,
            )
        )
        self._background_tasks.add(_task)
        _task.add_done_callback(self._background_tasks.discard)

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return t("gateway.background.started", preview=preview, task_id=task_id)

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning                       Show current effort level and display state
            /reasoning <level>               Set reasoning effort for this session only
            /reasoning <level> --global      Persist reasoning effort to config.yaml
            /reasoning reset                 Clear this session's reasoning override
            /reasoning show|on               Show model reasoning in responses
            /reasoning hide|off              Hide model reasoning from responses
        """
        from gateway.run import _hermes_home, _platform_config_key
        import yaml

        raw_args = event.get_command_args().strip()
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        config_path = _hermes_home / "config.yaml"
        # Normalize the source (Telegram DM topic recovery) before deriving
        # the override key so storage matches the key the next message turn
        # reads — same fix as /model (#30479).
        _reasoning_source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(_reasoning_source)
        self._show_reasoning = self._load_show_reasoning()
        # Use the session's effective model (session /model override wins over
        # config default) so per-model reasoning_overrides display correctly.
        _session_model = str(
            ((getattr(self, "_session_model_overrides", {}) or {}).get(session_key) or {}).get("model") or ""
        )
        self._reasoning_config = self._resolve_session_reasoning_config(
            source=event.source,
            session_key=session_key,
            model=_session_model,
        )

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                atomic_config_write(config_path, user_config)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not raw_args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = t("gateway.reasoning.level_default")
            elif rc.get("enabled") is False:
                level = t("gateway.reasoning.level_disabled")
            else:
                level = rc.get("effort", "medium")
            display_state = (
                t("gateway.reasoning.display_on")
                if self._show_reasoning
                else t("gateway.reasoning.display_off")
            )
            has_session_override = session_key in (getattr(self, "_session_reasoning_overrides", {}) or {})
            scope = (
                t("gateway.reasoning.scope_session")
                if has_session_override
                else t("gateway.reasoning.scope_global")
            )
            return t(
                "gateway.reasoning.status",
                level=level,
                scope=scope,
                display=display_state,
            )

        # Display toggle (per-platform)
        platform_key = _platform_config_key(event.source.platform)
        if args in {"show", "on"}:
            self._show_reasoning = True
            _save_config_key(f"display.platforms.{platform_key}.show_reasoning", True)
            return t("gateway.reasoning.display_set_on", platform=platform_key)

        if args in {"hide", "off"}:
            self._show_reasoning = False
            _save_config_key(f"display.platforms.{platform_key}.show_reasoning", False)
            return t("gateway.reasoning.display_set_off", platform=platform_key)

        # Effort level change
        effort = args.strip()
        if effort == "reset":
            if persist_global:
                return t("gateway.reasoning.reset_global_unsupported")
            self._set_session_reasoning_override(session_key, None)
            self._reasoning_config = self._load_reasoning_config()
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.reset_done")
        if effort == "none":
            parsed = {"enabled": False}
        elif effort in {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            parsed = {"enabled": True, "effort": effort}
        else:
            return t(
                "gateway.reasoning.unknown_arg",
                arg=effort or raw_args.lower(),
            )

        self._reasoning_config = parsed
        if persist_global:
            if _save_config_key("agent.reasoning_effort", effort):
                self._set_session_reasoning_override(session_key, None)
                self._evict_cached_agent(session_key)
                return t("gateway.reasoning.set_global", effort=effort)
            self._set_session_reasoning_override(session_key, parsed)
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.set_global_save_failed", effort=effort)

        self._set_session_reasoning_override(session_key, parsed)
        self._evict_cached_agent(session_key)
        return t("gateway.reasoning.set_session", effort=effort)

    async def _handle_memory_command(self, event: MessageEvent) -> str:
        """Handle /memory — review pending memory writes + toggle the approval gate.

        Memory entries are small enough to review inline in a chat bubble, so
        the full pending/approve/reject/approval flow works on every platform.
        Gate changes persist to config.yaml and evict the cached agent so the
        new setting takes effect on the next message.
        """
        from gateway.run import _hermes_home
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        from tools.memory_tool import load_on_disk_store

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []
        session_key = self._session_key_for_source(event.source)
        config_path = _hermes_home / "config.yaml"

        def _set_approval(enabled: bool):
            import yaml
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            user_config.setdefault("memory", {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            # New setting must take effect next message → drop cached agent.
            self._evict_cached_agent(session_key)

        # Apply approved writes against a fresh on-disk store (the gateway has
        # no long-lived agent; the store persists to the same MEMORY/USER.md).
        # load_on_disk_store() honors the user's configured char limits.
        store = load_on_disk_store()

        out = handle_pending_subcommand(
            wa.MEMORY, args, memory_store=store, set_mode_fn=_set_approval,
        )
        if out is None:
            out = ("Unknown /memory subcommand. Use: pending, approve <id>, "
                   "reject <id>, approval <on|off>.")
        return out

    async def _handle_skills_command(self, event: MessageEvent) -> str:
        """Handle /skills on the gateway — pending skill-write review only.

        The full skills hub (search/browse/install) stays CLI-only; this
        handler covers the write-approval review surface (pending / approve /
        reject / diff / approval) so a skill staged from a gateway session can
        be reviewed from that same session. Gated by ``skills.write_approval``
        via the CommandDef's ``gateway_config_gate``; also answers when staged
        writes still exist after the gate was turned off (so they are never
        stranded).

        ``diff`` output is truncated for chat bubbles — the full diff lives in
        the pending JSON file under ``~/.hermes/pending/skills/``. (Note this is
        the write-approval ``diff <id>``; the CLI also has an unrelated
        ``hermes skills diff <name>`` that diffs a bundled skill vs stock.)
        """
        from gateway.run import _hermes_home
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []
        session_key = self._session_key_for_source(event.source)
        config_path = _hermes_home / "config.yaml"

        gate_on = wa.write_approval_enabled(wa.SKILLS)
        wants_toggle = bool(args) and args[0].lower() in {"approval", "mode"}
        if not gate_on and not wants_toggle and wa.pending_count(wa.SKILLS) == 0:
            return ("Skill write approval is off (skills.write_approval). "
                    "Enable it with /skills approval on, then review staged "
                    "writes here with /skills pending.")

        def _set_approval(enabled: bool):
            import yaml
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            user_config.setdefault("skills", {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            # New setting must take effect next message → drop cached agent.
            self._evict_cached_agent(session_key)

        out = handle_pending_subcommand(
            wa.SKILLS, args, set_mode_fn=_set_approval,
        )
        if out is None:
            return ("Unknown /skills subcommand on this platform. Use: pending, "
                    "approve <id>, reject <id>, diff <id>, approval <on|off>. "
                    "(Search/install are CLI-only.)")

        # Chat bubbles can't hold a full skill diff — truncate and point at
        # the real review surface. (Note: `hermes skills diff <name>` is a
        # *different* command — it diffs a bundled skill against its stock
        # version — so we point at the pending JSON file, not that command.)
        if args and args[0].lower() == "diff" and len(out) > 3000:
            pending_id = args[1] if len(args) > 1 else "<id>"
            out = (out[:3000]
                   + "\n… (truncated — full diff in "
                     f"~/.hermes/pending/skills/{pending_id}.json)")
        return out

    async def _handle_fast_command(self, event: MessageEvent) -> str:
        """Handle /fast — mirror the CLI Priority Processing toggle in gateway chats."""
        from gateway.run import _hermes_home, _load_gateway_config, _resolve_gateway_model
        import yaml
        from hermes_cli.models import model_supports_fast_mode

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._service_tier = self._load_service_tier()

        user_config = _load_gateway_config()
        model = _resolve_gateway_model(user_config)
        if not model_supports_fast_mode(model):
            return t("gateway.fast.not_supported")

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                atomic_config_write(config_path, user_config)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not args or args == "status":
            status = t("gateway.fast.status_fast") if self._service_tier == "priority" else t("gateway.fast.status_normal")
            return t("gateway.fast.status", mode=status)

        if args in {"fast", "on"}:
            self._service_tier = "priority"
            saved_value = "fast"
            label = t("gateway.fast.label_fast")
        elif args in {"normal", "off"}:
            self._service_tier = None
            saved_value = "normal"
            label = t("gateway.fast.label_normal")
        else:
            return t("gateway.fast.unknown_arg", arg=args)

        if _save_config_key("agent.service_tier", saved_value):
            return t("gateway.fast.saved", label=label)
        return t("gateway.fast.session_only", label=label)

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        else:
            enable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` in config.yaml (default off).
        When enabled, cycles the tool progress mode through off → new → all →
        verbose → off for the *current platform*.  The setting is saved to
        ``display.platforms.<platform>.tool_progress`` so each channel can
        have its own verbosity level independently.
        """
        from gateway.run import _hermes_home, _load_gateway_config, _platform_config_key

        config_path = _hermes_home / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- check config gate ------------------------------------------------
        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(
                cfg_get(user_config, "display", "tool_progress_command"),
                default=False,
            )
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return t("gateway.verbose.not_enabled")

        # --- cycle mode (per-platform) ----------------------------------------
        cycle = ["off", "new", "all", "verbose", "log"]
        descriptions = {
            "off": t("gateway.verbose.mode_off"),
            "new": t("gateway.verbose.mode_new"),
            "all": t("gateway.verbose.mode_all"),
            "verbose": t("gateway.verbose.mode_verbose"),
            "log": t("gateway.verbose.mode_log"),
        }

        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to display.platforms.<platform>.tool_progress
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if "platforms" not in display or not isinstance(display.get("platforms"), dict):
                display["platforms"] = {}
            if platform_key not in display["platforms"] or not isinstance(display["platforms"].get(platform_key), dict):
                display["platforms"][platform_key] = {}
            display["platforms"][platform_key]["tool_progress"] = new_mode
            atomic_config_write(config_path, user_config)
            return (
                f"{descriptions[new_mode]}\n"
                + t("gateway.verbose.saved_suffix", platform=platform_key)
            )
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command — toggle the runtime-metadata footer.

        Usage:
            /footer           → toggle on/off
            /footer on        → enable globally
            /footer off       → disable globally
            /footer status    → show current state + fields

        The footer is saved to ``display.runtime_footer.enabled`` (global).
        Per-platform overrides under ``display.platforms.<platform>.runtime_footer``
        are respected but not modified here — edit config.yaml directly for
        per-platform control.
        """
        from gateway.run import _hermes_home, _load_gateway_config, _platform_config_key, _resolve_gateway_model
        from gateway.runtime_footer import resolve_footer_config

        config_path = _hermes_home / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- parse argument -------------------------------------------------
        arg = ""
        try:
            text = (getattr(event, "message", None) or "").strip()
            if text.startswith("/"):
                parts = text.split(None, 1)
                if len(parts) > 1:
                    arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        # --- load config ----------------------------------------------------
        try:
            user_config: dict = _load_gateway_config()
        except Exception as e:
            return t("gateway.config_read_failed", error=e)

        effective = resolve_footer_config(user_config, platform_key)

        if arg in {"status", "?"}:
            state = t("gateway.footer.state_on") if effective["enabled"] else t("gateway.footer.state_off")
            fields = ", ".join(effective.get("fields") or [])
            return t(
                "gateway.footer.status",
                state=state,
                fields=fields,
                platform=platform_key,
            )

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not effective["enabled"]
        else:
            return t("gateway.footer.usage")

        # --- write global flag ---------------------------------------------
        try:
            if not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if not isinstance(display.get("runtime_footer"), dict):
                display["runtime_footer"] = {}
            display["runtime_footer"]["enabled"] = new_state
            atomic_config_write(config_path, user_config)
        except Exception as e:
            logger.warning("Failed to save runtime_footer.enabled: %s", e)
            return t("gateway.config_save_failed", error=e)

        state = t("gateway.footer.state_on") if new_state else t("gateway.footer.state_off")
        example = ""
        if new_state:
            # Show a preview using current agent state if available.
            from gateway.runtime_footer import format_runtime_footer
            preview = format_runtime_footer(
                model=_resolve_gateway_model(user_config) or None,
                context_tokens=0,
                context_length=None,
                fields=effective.get("fields") or ["model", "context_pct", "cwd"],
            )
            if preview:
                example = t("gateway.footer.example_line", preview=preview)
        return t("gateway.footer.saved", state=state, example=example)

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context.

        Accepts an optional focus topic: ``/compress <focus>`` guides the
        summariser to preserve information related to *focus* while being
        more aggressive about discarding everything else.

        Also accepts the boundary-aware form ``/compress here [N]``:
        summarize everything except the most recent ``N`` exchanges
        (default 2), kept verbatim. Inspired by Claude Code's Rewind
        "Summarize up to here" action (v2.1.139, May 2026,
        https://code.claude.com/docs/en/whats-new/2026-w20).
        """
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")

        # Parse args: either a focus topic (full compress) or the
        # boundary-aware "here [N]" form (partial compress).
        from hermes_cli.partial_compress import (
            extract_compress_flags,
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
            summarize_compress_preview,
        )
        _raw_args = (event.get_command_args() or "").strip()
        # Strip --preview/--dry-run/--aggressive before positional parsing
        # so the flags coexist with 'here [N]' / focus-topic forms.
        _raw_args, _preview, _aggressive = extract_compress_flags(_raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

        _agg_note = ""
        if _aggressive:
            # LLM-free hard truncation is not supported on this surface —
            # it would need its own transcript-persistence branch outside
            # the guarded _compress_context rotation machinery (#44794).
            _agg_note = t("gateway.compress.aggressive_unsupported")
            if not _preview:
                return _agg_note

        if _preview:
            # Report what WOULD be compressed — no agent, no writes.
            from agent.model_metadata import estimate_request_tokens_rough
            _pv_msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            approx_tokens = estimate_request_tokens_rough(_pv_msgs)
            report = summarize_compress_preview(
                _pv_msgs, partial, keep_last, focus_topic, approx_tokens
            )
            lines = [f"🗜️ {line}" for line in report["lines"]]
            if _aggressive:
                lines.append(_agg_note)
            return "\n".join(lines)

        try:
            from run_agent import AIAgent
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            session_key = self._session_key_for_source(source)
            # Preserve the same platform + stable gateway session identity that a
            # normal gateway turn passes (gateway/run.py main turn), so external
            # context engines bind this temporary compression agent to the
            # original platform conversation instead of falling back to an
            # unbound/default "cli" host source — see #50422. _platform_config_key
            # maps LOCAL->"cli" exactly like the live turn, avoiding a new
            # "local" vs "cli" mismatch.
            from gateway.run import _platform_config_key
            platform_key = (
                _platform_config_key(source.platform) if source.platform else None
            )
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
            if not runtime_kwargs.get("api_key"):
                return t("gateway.compress.no_provider")

            # Pass the FULL transcript (tool results included) — same
            # rationale as the session-hygiene auto-compress in
            # gateway/run.py (#3854): filtering to user/assistant-only
            # starves the compressor's tool-result pruning and can trip the
            # protect-first/last early-return on short filtered histories.
            msgs = [
                m for m in history
                if m.get("role") in {"user", "assistant", "tool"}
            ]

            # Boundary-aware split: only the head is summarized; the most
            # recent `keep_last` exchanges are preserved verbatim. The
            # split snaps the tail to a user-turn start so the rejoined
            # transcript keeps role alternation valid.
            tail: list = []
            head = msgs
            if partial:
                head, tail = split_history_for_partial_compress(msgs, keep_last)
                if not tail:
                    # Degenerate split — fall back to full compression.
                    partial = False
                    head = msgs

            # Bind the temporary compression agent to the originating source's
            # platform + stable gateway session key. These are *authoritative*
            # identity invariants (derived from `source`), so assign them into
            # runtime_kwargs directly rather than via setdefault: a value already
            # present there from the resolver would be a placeholder/stale
            # identity and must not win. Assigning (vs passing a second explicit
            # kwarg) also keeps each key single-valued, avoiding a "got multiple
            # values for keyword argument" TypeError. platform is only set when
            # known: for a source without platform metadata we leave it unset so
            # AIAgent's default (platform=None -> source "cli") applies, exactly
            # the prior behavior. _resolve_session_agent_runtime does not set
            # either key today, so in practice this just adds them.
            if platform_key is not None:
                runtime_kwargs["platform"] = platform_key
            runtime_kwargs["gateway_session_key"] = session_key
            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=4,
                quiet_mode=True,
                skip_memory=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
                session_db=getattr(self._session_db, "_db", self._session_db),
            )
            try:
                tmp_agent._print_fn = lambda *a, **kw: None
                # Prevent close() from ending the newly rotated session —
                # the gateway session entry now points at the new id and
                # must remain open for the next user turn.
                tmp_agent._end_session_on_close = False

                # Estimate with system prompt + tool schemas included so the
                # figure reflects real request pressure, not a transcript-only
                # underestimate (#6217). Must be computed after tmp_agent is
                # built so _cached_system_prompt/tools are populated.
                _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
                _tools = getattr(tmp_agent, "tools", None) or None
                approx_tokens = estimate_request_tokens_rough(
                    msgs, system_prompt=_sys_prompt, tools=_tools
                )

                compressor = tmp_agent.context_compressor
                if not compressor.has_content_to_compress(head):
                    return t("gateway.compress.nothing_to_do")

                loop = asyncio.get_running_loop()
                compressed, _ = await loop.run_in_executor(
                    None,
                    lambda: tmp_agent._compress_context(head, "", approx_tokens=approx_tokens, focus_topic=focus_topic, force=True)
                )

                # Re-append the verbatim tail after the compressed head,
                # guarding the seam against illegal role adjacency.
                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)

                # _compress_context either rotated (legacy: ended the old
                # session, created a continuation id — write compressed messages
                # into the NEW session so the original stays searchable) or
                # compacted in place (compression.in_place / #38763: same id,
                # transcript replaced with the compacted set).
                new_session_id = tmp_agent.session_id
                rotated = new_session_id != session_entry.session_id
                _in_place = bool(getattr(tmp_agent, "_last_compaction_in_place", False))

                # Persist the compressed transcript BEFORE repointing the live
                # session onto the new session_id. Order matters: if we
                # repointed first and the canonical DB write then failed (lock
                # contention under concurrent writes, ENOSPC, a disk/IO error),
                # the session entry would already reference a brand-new, empty
                # session_id while the handler still reported success — the
                # user's active conversation would silently vanish from view.
                # Writing first, and treating a write failure as fatal, keeps
                # the old history reachable (on rotation the entry still points
                # at it; in place the original transcript is untouched) and lets
                # the outer handler surface a "compress failed" banner instead.
                #
                # Only rewrite the transcript when rotation produced a NEW
                # session id.  In-place compaction does NOT need a rewrite:
                # archive_and_compact() has already soft-archived the previous
                # active rows and inserted the compacted messages as the new
                # active set inside _compress_context().  Calling
                # rewrite_transcript() after in-place compaction would invoke
                # replace_messages(active_only=False) which DELETEs ALL rows —
                # including the archived turns that archive_and_compact()
                # deliberately preserved (silent data loss, #61145).
                #
                # The third case: _compress_context could NOT rotate AND was
                # not in-place (e.g. legacy mode but _session_db unavailable /
                # the DB split raised) — there session_id is unchanged for a
                # FAILURE reason, and rewrite_transcript() would DELETE the
                # original messages and replace them with only the compressed
                # summary (permanent data loss #44794, #39704).
                if rotated:
                    if not await self.async_session_store.rewrite_transcript(
                        new_session_id, compressed
                    ):
                        raise RuntimeError(
                            f"failed to persist compressed transcript for "
                            f"session {new_session_id}"
                        )
                    session_entry.session_id = new_session_id
                    await self.async_session_store._save()
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compress-command",
                    )
                elif _in_place:
                    # archive_and_compact() already persisted the compacted
                    # transcript inside _compress_context — nothing to do.
                    pass
                else:
                    logger.warning(
                        "Manual /compress: session rotation did not occur "
                        "(session_id unchanged) and in-place mode is off — "
                        "preserving original transcript instead of overwriting "
                        "it (#44794)."
                    )
                # Reset stored token count — transcript changed, old value is stale
                await self.async_session_store.update_session(
                    session_entry.session_key, last_prompt_tokens=0
                )
                new_tokens = estimate_request_tokens_rough(
                    compressed, system_prompt=_sys_prompt, tools=_tools
                )
                summary = summarize_manual_compression(
                    msgs,
                    compressed,
                    approx_tokens,
                    new_tokens,
                )
                # Detect summary-generation failure so we can surface a
                # visible warning to the user even on the manual /compress
                # path (otherwise the failure is silently logged).
                # _last_compress_aborted means the aux LLM returned no
                # usable summary and the compressor preserved messages
                # unchanged (no drop, no placeholder).  force=True was
                # passed above so any active cooldown is bypassed.
                _summary_aborted = bool(getattr(compressor, "_last_compress_aborted", False))
                _summary_err = getattr(compressor, "_last_summary_error", None)
                # Separately: did the user's CONFIGURED aux model fail
                # and we recovered via main?  Surface that as an info
                # note so they can fix their config.
                _aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
                _aux_fail_err = getattr(compressor, "_last_aux_model_failure_error", None)
            finally:
                # Evict cached agent so next turn rebuilds system prompt
                # from current files (SOUL.md, memory, etc.).
                self._evict_cached_agent(session_key)
                self._cleanup_agent_resources(tmp_agent)
            lines = [f"🗜️ {summary['headline']}"]
            if focus_topic:
                lines.append(t("gateway.compress.focus_line", topic=focus_topic))
            lines.append(summary["token_line"])
            if summary["note"]:
                lines.append(summary["note"])
            if _summary_aborted:
                lines.append(
                    t(
                        "gateway.compress.aborted",
                        error=(_summary_err or "unknown error"),
                    )
                )
            elif _aux_fail_model:
                lines.append(
                    t(
                        "gateway.compress.aux_failed",
                        model=_aux_fail_model,
                        error=(_aux_fail_err or "unknown error"),
                    )
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return t("gateway.compress.failed", error=e)

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

        source = event.source
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
            sessions = await self._session_db.list_sessions_rich(source=user_source, limit=10)
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
        self._clear_session_boundary_security_state(session_key)

        # Clear session-scoped model/reasoning overrides so the resumed
        # conversation picks up configured defaults instead of a /model
        # switch made in the previous session under the same chat
        # session_key. /resume is a conversation boundary just like /new
        # (which clears these too); without this, a stale override leaks
        # across the switch. See #10702.
        _overrides = getattr(self, "_session_model_overrides", None)
        if isinstance(_overrides, dict):
            _overrides.pop(session_key, None)
        self._set_session_reasoning_override(session_key, None)
        _pending_notes = getattr(self, "_pending_model_notes", None)
        if isinstance(_pending_notes, dict):
            _pending_notes.pop(session_key, None)
        # Clear per-session model cache too, for the same reason — the
        # resumed conversation must resolve from current config, not a
        # stale value cached under this session_key before the switch
        # (mirrors /new and the compression-exhausted auto-reset, #58403).
        _lrm = getattr(self, "_last_resolved_model", None)
        if isinstance(_lrm, dict):
            _lrm.pop(session_key, None)

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

        source = event.source
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

        # Copy conversation history to the new session
        for msg in history:
            try:
                await self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning"),
                    reasoning_content=msg.get("reasoning_content"),
                    reasoning_details=msg.get("reasoning_details"),
                    codex_reasoning_items=msg.get("codex_reasoning_items"),
                    codex_message_items=msg.get("codex_message_items"),
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

    async def _handle_credits_command(self, event: MessageEvent) -> str:
        """Handle /credits -- show Nous credit balance and the top-up handoff.

        Renders the balance block + identity line + a tappable top-up URL that
        opens the portal billing page with the modal open. The terminal does NOT
        confirm, poll, or track payment (billing phase 2a) — checkout happens in
        the browser and the next /credits shows the new balance. The tappable URL
        is the affordance: it works on every platform (button-capable or plain
        text like SMS/email). Fetched off the event loop; fail-open.
        """
        from agent.account_usage import build_credits_view

        try:
            view = await asyncio.to_thread(build_credits_view, markdown=True)
        except Exception:
            view = None

        if view is None or not view.logged_in:
            return t("gateway.credits.not_logged_in")

        lines: list[str] = ["💳 **Nous credits**"]
        for line in view.balance_lines:
            if line.lstrip().startswith("📈"):
                continue  # drop the helper's header; we print our own
            lines.append(line)
        if view.identity_line:
            lines.append("")
            lines.append(view.identity_line)
        if view.topup_url:
            lines.append("")
            lines.append(f"Top up: {view.topup_url}")
            lines.append("Complete your top-up in the browser — credits will appear in /credits shortly.")
        return "\n".join(lines)

    def _context_breakdown_lines(self, agent, source) -> list[str]:
        """Render the per-category context breakdown for /usage.

        Estimated (chars/4) — same engine the desktop popover uses. Returns an
        empty list and never raises on failure so /usage stays robust.
        """
        try:
            from agent.context_breakdown import compute_session_context_breakdown

            history: list[dict] = []
            try:
                entry = self.session_store.get_or_create_session(source)
                history = self.session_store.load_transcript(entry.session_id) or []
            except Exception:
                history = []

            payload = compute_session_context_breakdown(agent, history)
            categories = payload.get("categories") or []
            if not categories:
                return []

            total = payload.get("estimated_total") or 0
            out = [t("gateway.usage.breakdown_header")]
            for cat in categories:
                tokens = int(cat.get("tokens") or 0)
                if tokens <= 0:
                    continue
                cat_id = str(cat.get("id") or "")
                label = t(f"gateway.usage.breakdown_cat_{cat_id}")
                # Missing key → t() echoes the key back; fall back to the
                # English label the engine already provides.
                if label.endswith(f"breakdown_cat_{cat_id}"):
                    label = str(cat.get("label") or cat_id)
                pct = round(tokens / total * 100) if total else 0
                out.append(
                    t("gateway.usage.breakdown_line", label=label, count=f"{tokens:,}", pct=pct)
                )
            return out if len(out) > 1 else []
        except Exception:
            return []

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns)
        so that rate limits, cost estimates, and detailed token breakdowns are
        available whenever the user asks, not only while the agent is running.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        source = event.source
        session_key = self._session_key_for_source(source)

        # `/usage reset [--force]` — redeem one banked Codex rate-limit reset
        # credit. Parsed before the display path so it never mixes with the
        # stats rendering below.
        raw_args = event.get_command_args().strip()
        args = [a.lower() for a in raw_args.split()] if raw_args else []
        wants_reset = bool(args) and args[0] == "reset"
        if args and not wants_reset:
            return t("gateway.usage.unknown_subcommand", args=raw_args)

        # Try running agent first (mid-turn), then cached agent (between turns)
        agent = self._running_agents.get(session_key)
        if not agent or agent is _AGENT_PENDING_SENTINEL:
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached = _cache.get(session_key)
                    if cached:
                        agent = cached[0]

        # Resolve provider/base_url/api_key for the account-usage fetch.
        # Prefer the live agent; fall back to persisted billing data on the
        # SessionDB row so `/usage` still returns account info between turns
        # when no agent is resident.
        provider = getattr(agent, "provider", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        base_url = getattr(agent, "base_url", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        api_key = getattr(agent, "api_key", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        if not provider and getattr(self, "_session_db", None) is not None:
            try:
                _entry_for_billing = await self.async_session_store.get_or_create_session(source)
                persisted = await self._session_db.get_session(_entry_for_billing.session_id) or {}
            except Exception:
                persisted = {}
            provider = provider or persisted.get("billing_provider")
            base_url = base_url or persisted.get("billing_base_url")

        if wants_reset:
            normalized_provider = str(provider or "").strip().lower()
            if normalized_provider != "openai-codex":
                return t("gateway.usage.reset_wrong_provider")
            force = "--force" in args[1:]
            from agent.account_usage import redeem_codex_reset_credit

            result = await asyncio.to_thread(
                redeem_codex_reset_credit,
                base_url=base_url,
                api_key=api_key,
                force=force,
            )
            return result.message

        # Fetch account usage off the event loop so slow provider APIs don't
        # block the gateway. Failures are non-fatal -- account_lines stays [].
        account_lines: list[str] = []
        credits_lines: list[str] = []
        if provider:
            try:
                account_snapshot = await asyncio.to_thread(
                    fetch_account_usage,
                    provider,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                account_snapshot = None
            if account_snapshot:
                account_lines = render_account_usage_lines(account_snapshot, markdown=True)

        # ── Nous credits magnitudes + monthly-grant % gauge ─────────────
        # Shared with the CLI / TUI /usage block via nous_credits_lines(): a single
        # auth-gate + portal-fetch + render path (which also honors the dev fixture).
        # Run off the event loop. The helper gates on "a Nous account is logged in"
        # — NOT the inference provider and NOT nested under `if provider:` — so a
        # Nous-credentialled user running inference elsewhere (or with none resident)
        # still sees their balance. NO recovery trigger: messaging binds no notice
        # consumer, so /usage only displays. Fail-open: never break /usage.
        try:
            from agent.account_usage import nous_credits_lines

            credits_lines = await asyncio.to_thread(nous_credits_lines, markdown=True)
        except Exception:
            credits_lines = []  # fail-open: never break /usage

        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = []

            # Rate limits (when available from provider headers)
            rl_state = agent.get_rate_limit_state()
            if rl_state and rl_state.has_data:
                from agent.rate_limit_tracker import format_rate_limit_compact
                lines.append(t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)))
                lines.append("")

            # Session token usage — detailed breakdown matching CLI
            input_tokens = getattr(agent, "session_input_tokens", 0) or 0
            output_tokens = getattr(agent, "session_output_tokens", 0) or 0

            lines.append(t("gateway.usage.header_session"))
            lines.append(t("gateway.usage.label_model", model=agent.model))
            lines.append(t("gateway.usage.label_input_tokens", count=f"{input_tokens:,}"))
            lines.append(t("gateway.usage.label_output_tokens", count=f"{output_tokens:,}"))
            lines.append(t("gateway.usage.label_total", count=f"{agent.session_total_tokens:,}"))
            lines.append(t("gateway.usage.label_api_calls", count=agent.session_api_calls))

            # Context window and compressions
            ctx = agent.context_compressor
            _lpt = ctx.last_prompt_tokens if ctx.last_prompt_tokens > 0 else 0
            if _lpt:
                pct = min(100, _lpt / ctx.context_length * 100) if ctx.context_length else 0
                lines.append(t("gateway.usage.label_context", used=f"{_lpt:,}", total=f"{ctx.context_length:,}", pct=f"{pct:.0f}"))
            if ctx.compression_count:
                lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))

            # Per-category context breakdown (estimated — chars/4 heuristic).
            # Same engine the desktop popover uses (PR #54907). The system
            # prompt / tools / skills / memory slices read off the live agent;
            # the conversation slice is estimated from the session transcript.
            breakdown_lines = await asyncio.to_thread(
                self._context_breakdown_lines, agent, source
            )
            if breakdown_lines:
                lines.append("")
                lines.extend(breakdown_lines)

            if account_lines:
                lines.append("")
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)

            return "\n".join(lines)

        # No agent at all -- check session history for a rough count
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            lines = [
                t("gateway.usage.header_session_info"),
                t("gateway.usage.label_messages", count=len(msgs)),
                t("gateway.usage.label_estimated_context", count=f"{approx:,}"),
                t("gateway.usage.detailed_after_first"),
            ]
            if account_lines:
                lines.append("")
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)
            return "\n".join(lines)
        if account_lines or credits_lines:
            # account-only, credits-only, or both — joined with a blank divider.
            parts = list(account_lines)
            if credits_lines:
                if parts:
                    parts.append("")
                parts.extend(credits_lines)
            return "\n".join(parts)
        return t("gateway.usage.no_data")

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        args = event.get_command_args().strip()

        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', args)

        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return t("gateway.insights.invalid_days", value=parts[i + 1])
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            loop = asyncio.get_running_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reload-mcp — reconnect MCP servers and rebuild the cached agent.

        Reloading MCP tools invalidates the provider prompt cache for the
        active session (tool schemas are baked into the system prompt).  The
        next message re-sends full input tokens, which is expensive on
        long-context or high-reasoning models.

        To surface that cost, the command routes through the slash-confirm
        primitive: users get an Approve Once / Always Approve / Cancel
        prompt before the reload actually runs.  "Always Approve" persists
        ``approvals.mcp_reload_confirm: false`` so the prompt is silenced
        for subsequent reloads in any session.

        Users can also skip the confirm by flipping the config key directly.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # Read the gate fresh from disk so a prior "always" click takes
        # effect on the next invocation without restarting the gateway.
        user_config = self._read_user_config()
        approvals = user_config.get("approvals") if isinstance(user_config, dict) else None
        confirm_required = True
        if isinstance(approvals, dict):
            confirm_required = bool(approvals.get("mcp_reload_confirm", True))

        if not confirm_required:
            return await self._execute_mcp_reload(event)

        # Route through slash-confirm.  The primitive sends the prompt and
        # stores the resume handler; the button/text response triggers
        # ``_resolve_slash_confirm`` which invokes the handler with the
        # chosen outcome.
        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return t("gateway.reload_mcp.cancelled")
            if choice == "always":
                # Persist the opt-out and run the reload.
                try:
                    from cli import save_config_value
                    save_config_value("approvals.mcp_reload_confirm", False)
                    logger.info(
                        "User opted out of /reload-mcp confirmation (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)
            # once / always → run the reload
            result = await self._execute_mcp_reload(event)
            if choice == "always":
                return f"{result}\n\n" + t("gateway.reload_mcp.always_followup")
            return result

        prompt_message = t("gateway.reload_mcp.confirm_prompt")
        return await self._request_slash_confirm(
            event=event,
            command="reload-mcp",
            title="/reload-mcp",
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload-skills — rescan skills dir, queue a note for next turn.

        Skills don't need to be in the system prompt for the model to use
        them (they're invoked via ``/skill-name``, ``skills_list``, or
        ``skill_view`` at runtime), so this does NOT clear the prompt cache
        — prefix caching stays intact.

        If any skills were added or removed, a one-shot note is queued on
        ``self._pending_skills_reload_notes[session_key]``. The gateway
        prepends it to the NEXT user message in this session (see the
        consumer at ~L11025 in ``_run_agent_turn``), then clears it. Nothing
        is written to the session transcript out-of-band, so message
        alternation is preserved.
        """
        loop = asyncio.get_running_loop()
        try:
            from agent.skill_commands import reload_skills

            result = await loop.run_in_executor(None, reload_skills)
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])  # [{"name", "description"}, ...]
            total = result.get("total", 0)

            # Let each connected adapter refresh any platform-side state
            # that cached the skill list at startup. Today that's the
            # Discord /skill autocomplete (registered once per connect);
            # without this call, new skills stay invisible in the
            # dropdown and deleted skills error out when clicked. Other
            # adapters that don't override refresh_skill_group (Telegram's
            # BotCommand menu, Slack subcommand map, etc.) are silently
            # skipped — the in-process reload above is enough for them.
            for adapter in list(self.adapters.values()):
                refresh = getattr(adapter, "refresh_skill_group", None)
                if not callable(refresh):
                    continue
                try:
                    maybe = refresh()
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception as exc:
                    logger.warning(
                        "Adapter %s refresh_skill_group raised: %s",
                        getattr(adapter, "name", adapter), exc,
                    )

            lines = [t("gateway.reload_skills.header")]
            if not added and not removed:
                lines.append(t("gateway.reload_skills.no_new"))
                lines.append(t("gateway.reload_skills.total", count=total))
                return "\n".join(lines)

            def _fmt_line(item: dict) -> str:
                nm = item.get("name", "")
                desc = item.get("description", "")
                if desc:
                    return t("gateway.reload_skills.item_with_desc", name=nm, desc=desc)
                return t("gateway.reload_skills.item_no_desc", name=nm)

            if added:
                lines.append(t("gateway.reload_skills.added_header"))
                for item in added:
                    lines.append(_fmt_line(item))
            if removed:
                lines.append(t("gateway.reload_skills.removed_header"))
                for item in removed:
                    lines.append(_fmt_line(item))
            lines.append(t("gateway.reload_skills.total", count=total))

            # Queue the one-shot note for the next user turn in this session.
            # Format matches how the system prompt renders pre-existing
            # skills (``    - name: description``) so the model reads the
            # diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            if added:
                sections.append("")
                sections.append("Added Skills:")
                for item in added:
                    sections.append(_fmt_line(item))
            if removed:
                sections.append("")
                sections.append("Removed Skills:")
                for item in removed:
                    sections.append(_fmt_line(item))
            sections.append("")
            sections.append("Use skills_list to see the updated catalog.]")
            note = "\n".join(sections)

            session_key = self._session_key_for_source(event.source)
            if not hasattr(self, "_pending_skills_reload_notes"):
                self._pending_skills_reload_notes = {}
            if session_key:
                self._pending_skills_reload_notes[session_key] = note

            return "\n".join(lines)

        except Exception as e:
            logger.warning("Skills reload failed: %s", e)
            return t("gateway.reload_skills.failed", error=e)

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles — list installed skill bundles.

        Mirrors the CLI ``/bundles`` handler. Returns a single text
        message suitable for any gateway adapter; bundles are loaded by
        invoking the bundle's own ``/<slug>`` command, not by this one.
        """
        try:
            from agent.skill_bundles import list_bundles, _bundles_dir
        except Exception as exc:
            logger.warning("Bundles command unavailable: %s", exc)
            return f"Bundles subsystem unavailable: {exc}"

        bundles = list_bundles()
        if not bundles:
            return (
                "No skill bundles installed.\n"
                "Create one on the host with:\n"
                "  `hermes bundles create <name> --skill <s1> --skill <s2>`\n"
                f"Directory: `{_bundles_dir()}`"
            )

        lines = [f"**Skill Bundles** ({len(bundles)} installed):", ""]
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            lines.append(
                f"• `/{info['slug']}` — {desc} _({skill_count} skills)_"
            )
            for s in info.get("skills", []):
                lines.append(f"    · {s}")
        lines.append("")
        lines.append("Invoke a bundle with `/<slug>` to load all its skills.")
        return "\n".join(lines)

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        The agent thread(s) are blocked inside tools/approval.py waiting for
        the user to respond.  This handler signals the event so the agent
        resumes and the terminal_tool executes the command inline — the same
        flow as the CLI's synchronous input() approval.

        Supports multiple concurrent approvals (parallel subagents,
        execute_code).  ``/approve`` resolves the oldest pending command;
        ``/approve all`` resolves every pending command at once.

        Usage:
            /approve              — approve oldest pending command once
            /approve all          — approve ALL pending commands at once
            /approve session      — approve oldest + remember for session
            /approve all session  — approve all + remember for session
            /approve always       — approve oldest + remember permanently
            /approve all always   — approve all + remember permanently
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.approval_expired")
            return t("gateway.approve.no_pending")

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in {"always", "permanent", "permanently"} for a in remaining):
            choice = "always"
        elif any(a in {"session", "ses"} for a in remaining):
            choice = "session"
        else:
            choice = "once"

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return t("gateway.approve.no_pending")

        # Resume typing indicator — agent is about to continue processing.
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        plural = "plural" if count > 1 else "singular"
        return t(f"gateway.approve.{choice}_{plural}", count=count)

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        ``/deny <reason>`` (or ``/deny all <reason>``) attaches a one-line
        reason that is relayed back to the agent so it can adapt instead of
        only hearing "denied". Ported from qwibitai/nanoclaw#2832.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.deny.stale")
            return t("gateway.deny.no_pending")

        # Parse args: a leading "all" token denies every pending command;
        # anything after it (or the whole arg string when "all" is absent) is
        # captured verbatim as the optional deny reason relayed to the agent.
        raw_args = event.get_command_args().strip()
        tokens = raw_args.split()
        resolve_all = bool(tokens) and tokens[0].lower() == "all"
        if resolve_all:
            reason = raw_args[len(tokens[0]):].strip()
        else:
            reason = raw_args
        # Cap to a sane one-liner; the agent only needs a short hint.
        if reason:
            reason = reason[:280].strip()

        count = resolve_gateway_approval(
            session_key, "deny", resolve_all=resolve_all,
            reason=reason or None,
        )
        if not count:
            return t("gateway.deny.no_pending")

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info(
            "User denied %d dangerous command(s) via /deny%s",
            count, " (with reason)" if reason else "",
        )
        if reason:
            if count > 1:
                return t("gateway.deny.denied_reason_plural", count=count, reason=reason)
            return t("gateway.deny.denied_reason_singular", reason=reason)
        if count > 1:
            return t("gateway.deny.denied_plural", count=count)
        return t("gateway.deny.denied_singular")

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Gateway uploads ONLY the summary report (system info + log tails),
        NOT full log files, to protect conversation privacy.  Users who need
        full log uploads should use ``hermes debug share`` from the CLI.
        """
        import asyncio
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report,
            upload_to_pastebin, _schedule_auto_delete,
            _GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
        )

        loop = asyncio.get_running_loop()

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            _best_effort_sweep_expired_pastes()
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)

            urls = {}
            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)

            # Schedule auto-deletion after 6 hours
            _schedule_auto_delete(list(urls.values()))

            lines = [_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            lines.append("")
            lines.append(t("gateway.debug.auto_delete"))
            lines.append(t("gateway.debug.full_logs_hint"))
            lines.append(t("gateway.debug.share_hint"))
            return "\n".join(lines)

        return await loop.run_in_executor(None, _collect_and_upload)

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` in a detached session (via ``setsid``) so it
        survives the gateway restart that ``hermes update`` may trigger. Marker
        files are written so either the current gateway process or the next one
        can notify the user when the update finishes.
        """
        from gateway.run import _hermes_home, _resolve_hermes_bin
        import json
        import shutil
        import subprocess
        from datetime import datetime
        from hermes_cli.config import is_managed, format_managed_message

        # Block non-messaging platforms (API server, webhooks, ACP)
        platform = event.source.platform
        _allowed = self._UPDATE_ALLOWED_PLATFORMS
        # Plugin platforms with allow_update_command=True are also allowed
        if platform not in _allowed:
            try:
                from gateway.platform_registry import platform_registry
                entry = platform_registry.get(platform.value)
                if not entry or not entry.allow_update_command:
                    return t("gateway.update.platform_not_messaging")
            except Exception:
                return t("gateway.update.platform_not_messaging")

        if is_managed():
            return f"✗ {format_managed_message('update Hermes Agent')}"

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return t("gateway.update.not_git_repo")

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return t("gateway.update.hermes_cmd_not_found")

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        session_key = self._session_key_for_source(event.source)
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "chat_type": event.source.chat_type,
            "user_id": event.source.user_id,
            "session_key": session_key,
            "timestamp": datetime.now().isoformat(),
        }
        if event.source.thread_id:
            pending["thread_id"] = event.source.thread_id
        if event.message_id:
            pending["message_id"] = event.message_id
        _tmp_pending = pending_path.with_suffix(".tmp")
        _tmp_pending.write_text(json.dumps(pending))
        _tmp_pending.replace(pending_path)
        exit_code_path.unlink(missing_ok=True)

        # Spawn `hermes update --gateway` detached so it survives gateway restart.
        # --gateway enables file-based IPC for interactive prompts (stash
        # restore, config migration) so the gateway can forward them to the
        # user instead of silently skipping them.
        # Use setsid for portable session detach (works under system services
        # where systemd-run --user fails due to missing D-Bus session).
        # PYTHONUNBUFFERED ensures output is flushed line-by-line so the
        # gateway can stream it to the messenger in near-real-time.
        # Spawn `hermes update --gateway` detached so it survives gateway restart.
        # --gateway enables file-based IPC for interactive prompts (stash
        # restore, config migration) so the gateway can forward them to the
        # user instead of silently skipping them.
        # Use setsid for portable session detach (works under system services
        # where systemd-run --user fails due to missing D-Bus session).
        # PYTHONUNBUFFERED ensures output is flushed line-by-line so the
        # gateway can stream it to the messenger in near-real-time.
        #
        # Windows: no bash/setsid chain.  Run `hermes update --gateway`
        # directly via sys.executable; redirect stdout/stderr to the same
        # output files via Popen file handles; write the exit code in a
        # follow-up write.  A tiny Python watcher would be cleaner but
        # we're already inside gateway/run.py's update path which is async,
        # so the simplest correct thing is: launch an inline Python helper
        # that runs the command and writes both outputs.
        try:
            if sys.platform == "win32":
                import textwrap
                from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

                # hermes_cmd is a list of argv parts we can pass directly
                # (no shell-quoting needed).
                helper = textwrap.dedent(
                    """
                    import os, subprocess, sys
                    output_path = sys.argv[1]
                    exit_code_path = sys.argv[2]
                    cmd = sys.argv[3:]
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    with open(output_path, "wb") as f:
                        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                        rc = proc.wait(timeout=3600)
                    with open(exit_code_path, "w") as f:
                        f.write(str(rc))
                    """
                ).strip()
                subprocess.Popen(
                    [
                        sys.executable, "-c", helper,
                        str(output_path), str(exit_code_path),
                        *hermes_cmd, "update", "--gateway",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **windows_detach_popen_kwargs(),
                )
            else:
                hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
                update_cmd = (
                    f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
                    f" > {shlex.quote(str(output_path))} 2>&1; "
                    # Avoid `status=$?`: `status` is a read-only special parameter
                    # in zsh, and this command string is copied/reused in macOS/zsh
                    # operator wrappers. Keep the template zsh-safe even though this
                    # specific subprocess currently runs under bash.
                    f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(exit_code_path))}"
                )
                setsid_bin = shutil.which("setsid")
                if setsid_bin:
                    # Preferred: setsid creates a new session, fully detached
                    subprocess.Popen(
                        [setsid_bin, "bash", "-c", update_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                else:
                    # Fallback: start_new_session=True calls os.setsid() in child
                    subprocess.Popen(
                        ["bash", "-c", update_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return t("gateway.update.start_failed", error=e)

        self._schedule_update_notification_watch()
        return t("gateway.update.starting")
