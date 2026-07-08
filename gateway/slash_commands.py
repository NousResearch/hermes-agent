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
from gateway.session import SessionSource, build_session_key
from hermes_cli.config import cfg_get, clear_model_endpoint_credentials
from utils import (
    atomic_json_write,
    atomic_yaml_write,
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
        new_entry = self.session_store.reset_session(session_key)

        # Clear any session-scoped model/reasoning overrides so the next agent
        # picks up configured defaults instead of previous session switches.
        self._set_session_model_override(session_key, None)
        self._set_session_reasoning_override(session_key, None)
        if hasattr(self, "_pending_model_notes"):
            self._pending_model_notes.pop(session_key, None)

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

        # Resolve session config info to surface to the user
        try:
            session_info = self._format_session_info()
        except Exception:
            session_info = ""

        if new_entry:
            header = await asyncio.to_thread(self._telegram_topic_new_header, source) or t("gateway.reset.header_default")
        else:
            # No existing session, just create one
            new_entry = self.session_store.get_or_create_session(source, force_new=True)
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
        session_entry = self.session_store.get_or_create_session(source)

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
        )

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
        session_entry = self.session_store.get_or_create_session(source)
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
                                _sess_entry = _self.session_store.get_or_create_session(
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
                        _self._set_session_model_override(_session_key, {
                            "model": result.new_model,
                            "provider": result.target_provider,
                            "api_key": result.api_key,
                            "base_url": result.base_url,
                            "api_mode": result.api_mode,
                        })

                        # Announce the deliberate switch to the conversation (P2).
                        await _self._announce_switch(
                            event.source,
                            "Model",
                            f"{_cur_provider}/{_cur_model}",
                            f"{result.target_provider}/{result.new_model}",
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
                                if config_path.exists():
                                    with open(config_path, encoding="utf-8") as f:
                                        _persist_cfg = yaml.safe_load(f) or {}
                                else:
                                    _persist_cfg = {}
                                _raw_model = _persist_cfg.get("model")
                                if isinstance(_raw_model, dict):
                                    _persist_model_cfg = _raw_model
                                elif isinstance(_raw_model, str) and _raw_model.strip():
                                    _persist_model_cfg = {"default": _raw_model.strip()}
                                    _persist_cfg["model"] = _persist_model_cfg
                                else:
                                    _persist_model_cfg = {}
                                    _persist_cfg["model"] = _persist_model_cfg
                                _persist_model_cfg["default"] = result.new_model
                                _persist_model_cfg["provider"] = result.target_provider
                                if result.base_url:
                                    _persist_model_cfg["base_url"] = result.base_url
                                if str(result.target_provider or "").strip().lower() != "custom":
                                    clear_model_endpoint_credentials(_persist_model_cfg, clear_base_url=True)
                                from hermes_cli.config import save_config
                                save_config(_persist_cfg)
                            except Exception as e:
                                logger.warning("Failed to persist model switch: %s", e)

                        # Build confirmation text
                        plabel = result.provider_label or result.target_provider
                        lines = [t("gateway.model.switched", model=result.new_model)]
                        lines.append(t("gateway.model.provider_label", provider=plabel))
                        try:
                            _reasoning_label = self._reasoning_effort_label(
                                self._resolve_session_reasoning_config(source=source)
                            )
                            if _reasoning_label:
                                lines.append(t("gateway.model.reasoning_label", effort=_reasoning_label))
                        except Exception:
                            pass
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
                    _sess_entry = self.session_store.get_or_create_session(source)
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
            # (single door — also persists the config-backed identity, RC-2/P3b).
            self._set_session_model_override(session_key, {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            })

            # Announce the deliberate switch to the conversation (P2). Compares the
            # (provider, model, api_mode) route so a same-slug/different-endpoint
            # switch still announces; silent on a true no-op. Best-effort.
            await self._announce_switch(
                event.source,
                "Model",
                f"{current_provider}/{current_model}",
                f"{result.target_provider}/{result.new_model}",
            )

            # Evict cached agent so the next turn creates a fresh agent from the
            # override rather than relying on cache signature mismatch detection.
            self._evict_cached_agent(session_key)

            # Persist to config (default) unless --session opted out
            if persist_global:
                try:
                    if config_path.exists():
                        with open(config_path, encoding="utf-8") as f:
                            cfg = yaml.safe_load(f) or {}
                    else:
                        cfg = {}
                    # Coerce scalar/None ``model:`` into a dict before mutation —
                    # otherwise ``cfg.setdefault("model", {})`` returns the existing
                    # scalar and the next assignment raises
                    # ``TypeError: 'str' object does not support item assignment``.
                    # Reproduces when ``config.yaml`` has ``model: <name>`` (flat
                    # string) instead of the proper nested ``model: {default: ...}``.
                    raw_model = cfg.get("model")
                    if isinstance(raw_model, dict):
                        model_cfg = raw_model
                    elif isinstance(raw_model, str) and raw_model.strip():
                        model_cfg = {"default": raw_model.strip()}
                        cfg["model"] = model_cfg
                    else:
                        model_cfg = {}
                        cfg["model"] = model_cfg
                    model_cfg["default"] = result.new_model
                    model_cfg["provider"] = result.target_provider
                    if result.base_url:
                        model_cfg["base_url"] = result.base_url
                    if str(result.target_provider or "").strip().lower() != "custom":
                        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)
                    from hermes_cli.config import save_config
                    save_config(cfg)
                except Exception as e:
                    logger.warning("Failed to persist model switch: %s", e)

            # Build confirmation message with full metadata
            provider_label = result.provider_label or result.target_provider
            lines = [t("gateway.model.switched", model=result.new_model)]
            lines.append(t("gateway.model.provider_label", provider=provider_label))

            # Reasoning effort in effect after the switch. /model does NOT clear
            # a /reasoning session override, so resolve the session-aware value
            # (falls back to config.yaml) rather than the global default.
            try:
                _reasoning_label = self._reasoning_effort_label(
                    self._resolve_session_reasoning_config(source=source)
                )
                if _reasoning_label:
                    lines.append(t("gateway.model.reasoning_label", effort=_reasoning_label))
            except Exception:
                pass

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
                atomic_yaml_write(config_path, config)
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
                atomic_yaml_write(config_path, config)
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
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
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
        self.session_store.rewrite_transcript(session_entry.session_id, truncated)
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

        mgr, session_entry = self._get_goal_manager_for_event(event)
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
        mgr, _session_entry = self._get_goal_manager_for_event(event)
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
        """Handle /undo [N] by delegating to the shared half-turn undo core."""
        source = event.source

        # Parse optional half-turn count: "/undo" → 1, "/undo 3" → 3.
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = int(raw_args.split()[0])
            except (ValueError, IndexError):
                return t("gateway.undo.invalid_count", arg=raw_args.split()[0])
            if n < 1:
                n = 1

        session_entry = self.session_store.get_or_create_session(source)
        result = self.session_store.rewind_session(session_entry.session_id, n)

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

        return t(
            "gateway.undo.removed",
            turns=n,
            count=len(result.get("rewound_ids") or []),
        ) + self._undo_tail_suffix(session_entry.session_id)

    def _undo_tail_suffix(self, session_id: str) -> str:
        """Render a one-line '↦ now at …' confirmation of the active tail.

        Lets a user confirm at a glance WHERE the thread landed after /undo or
        /redo — which message is now last — without scrolling. Best-effort:
        never let a preview failure break the command's primary reply.
        """
        try:
            import hermes_undo

            hermes_undo._session_db = self.session_store._db
            info = hermes_undo.tail_preview(session_id)
        except Exception as e:
            logger.debug("undo/redo tail preview skipped: %s", e)
            return ""
        # The read itself failed (transient DB error) — the primary undo/redo
        # already succeeded, so omit the suffix rather than misreport "empty".
        if info.get("error"):
            return ""
        if info.get("empty"):
            return "\n" + t("gateway.undo.now_empty")
        # Bound the role to the set that has a translated party label; an
        # unexpected role (system/developer/legacy function) would otherwise
        # ask t() for a missing key and render the raw key path to the user.
        role = info.get("role") or "message"
        if role not in {"user", "assistant", "tool", "message"}:
            role = "message"
        who = t(f"gateway.undo.party.{role}")
        preview = info.get("preview")
        if preview:
            return "\n" + t("gateway.undo.now_at", who=who, preview=preview)
        return "\n" + t("gateway.undo.now_at_notext", who=who)

    async def _handle_redo_command(self, event: MessageEvent) -> str:
        """Handle /redo [N] by delegating to the shared redo core."""
        source = event.source
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = int(raw_args.split()[0])
            except (ValueError, IndexError):
                return t("gateway.redo.invalid_count", arg=raw_args.split()[0])

        session_entry = self.session_store.get_or_create_session(source)
        result = self.session_store.restore_session(session_entry.session_id, n)
        if result is None:
            return t("gateway.redo.nothing")

        reactivated = int(result.get("reactivated_count") or 0)
        if reactivated <= 0:
            message = str(result.get("message") or "")
            if "restart" in message:
                return t("gateway.redo.restart_lost")
            return t("gateway.redo.nothing")

        session_entry.last_prompt_tokens = 0
        try:
            session_key = build_session_key(source)
            self._evict_cached_agent(session_key)
        except Exception as e:
            logger.debug("redo: cached-agent eviction skipped: %s", e)

        return t(
            "gateway.redo.restored",
            ops=n,
            count=reactivated,
        ) + self._undo_tail_suffix(session_entry.session_id)

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
        self._reasoning_config = self._resolve_session_reasoning_config(
            source=event.source,
            session_key=session_key,
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
                atomic_yaml_write(config_path, user_config)
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
        # Capture the resolved effective effort BEFORE applying so the announce
        # compares old vs new correctly (P2). Uses the config-fallback-inclusive
        # label, not the footer's "" sentinel.
        _old_effort = self._resolved_effort_label(
            source=event.source, session_key=session_key,
        )
        if effort == "reset":
            if persist_global:
                return t("gateway.reasoning.reset_global_unsupported")
            self._set_session_reasoning_override(session_key, None)
            self._reasoning_config = self._load_reasoning_config()
            self._evict_cached_agent(session_key)
            _new_effort = self._resolved_effort_label(
                source=event.source, session_key=session_key,
            )
            await self._announce_switch(event.source, "Reasoning", _old_effort, _new_effort)
            return t("gateway.reasoning.reset_done")
        if effort == "none":
            parsed = {"enabled": False}
        elif effort in {"minimal", "low", "medium", "high", "xhigh"}:
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
                _new_effort = self._resolved_effort_label(
                    source=event.source, session_key=session_key,
                )
                await self._announce_switch(event.source, "Reasoning", _old_effort, _new_effort)
                return t("gateway.reasoning.set_global", effort=effort)
            self._set_session_reasoning_override(session_key, parsed)
            self._evict_cached_agent(session_key)
            _new_effort = self._resolved_effort_label(
                source=event.source, session_key=session_key,
            )
            await self._announce_switch(event.source, "Reasoning", _old_effort, _new_effort)
            return t("gateway.reasoning.set_global_save_failed", effort=effort)

        self._set_session_reasoning_override(session_key, parsed)
        self._evict_cached_agent(session_key)
        _new_effort = self._resolved_effort_label(
            source=event.source, session_key=session_key,
        )
        await self._announce_switch(event.source, "Reasoning", _old_effort, _new_effort)
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
            atomic_yaml_write(config_path, user_config)
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
            atomic_yaml_write(config_path, user_config)
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
                atomic_yaml_write(config_path, user_config)
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
        cycle = ["off", "new", "all", "verbose"]
        descriptions = {
            "off": t("gateway.verbose.mode_off"),
            "new": t("gateway.verbose.mode_new"),
            "all": t("gateway.verbose.mode_all"),
            "verbose": t("gateway.verbose.mode_verbose"),
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
            atomic_yaml_write(config_path, user_config)
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
            atomic_yaml_write(config_path, user_config)
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
                fields=effective.get("fields") or ["provider_model", "context_full", "cwd"],
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
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")

        # Real, provider-measured context size from the last live API call
        # (the same number /usage reports). Captured BEFORE the post-compress
        # reset further down zeroes session_entry.last_prompt_tokens. This is
        # the only tokenizer-truth figure available here; the estimates below
        # are char/4 heuristics. Showing both side-by-side stops the
        # apples-to-oranges confusion where a tool-heavy real context (e.g.
        # 290K) gets compared against a transcript-only estimate.
        real_before_tokens = int(getattr(session_entry, "last_prompt_tokens", 0) or 0)

        # Parse args: either a focus topic (full compress) or the
        # boundary-aware "here [N]" form (partial compress).
        from hermes_cli.partial_compress import (
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
        )
        _raw_args = (event.get_command_args() or "").strip()
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

        try:
            from run_agent import AIAgent
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                estimate_request_tokens_rough,
            )

            session_key = self._session_key_for_source(source)
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
            if not runtime_kwargs.get("api_key"):
                return t("gateway.compress.no_provider")

            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]

            # The rows EXCLUDED from the chat-only compression input: tool
            # results, system rows, contentless (tool-call-only) turns. When
            # the transcript rewrite below happens, these stored rows are
            # dropped — usually the bulk of a tool-heavy session. Measure
            # them so the feedback can reconcile both axes instead of
            # claiming "No changes" over a six-figure token cut (#F1,
            # 2026-07-02 spec: compress-feedback-honesty).
            non_chat_rows = [
                m
                for m in history
                if not (m.get("role") in {"user", "assistant"} and m.get("content"))
            ]
            non_chat_count = len(non_chat_rows)

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

                # Two independent measurements, surfaced as two lines so the
                # user can see both "how big is the conversation" and "how big
                # is the actual request":
                #
                #  • Chat size  — estimate_messages_tokens_rough over the
                #    user/hermes turns only (`msgs`). Excludes system prompt,
                #    tool schemas, and tool results. This is the original
                #    "Approx request size" figure Ace wanted to retain.
                #  • Full request size — estimate_request_tokens_rough over the
                #    FULL transcript (`history`) plus system prompt + tool
                #    schemas. This is what the model is actually sent and lines
                #    up with the real provider count from /usage.
                #
                # The compressor itself is fed the full-request estimate so its
                # pressure logic reflects reality (#6217).
                #
                # F3 (2026-07-02 honesty spec): resolve the REAL fixed overhead
                # (resident agent's system prompt + full tool schemas) ONCE,
                # up front, and use it for BOTH the before and after estimates.
                # The temp agent here is memory-only (empty system prompt,
                # tools=[memory]), so measuring "before" with its overhead
                # under-reports by the entire fixed cost while the line claims
                # "includes chat, system, tools".
                _tmp_sys = getattr(tmp_agent, "_cached_system_prompt", "") or ""
                _tmp_tools = getattr(tmp_agent, "tools", None) or None
                _real_sys, _real_tools = self._resolve_fixed_overhead(session_key)
                _full_after_has_fixed = bool(_real_sys or _real_tools)
                _sys_prompt = _real_sys if _full_after_has_fixed else _tmp_sys
                _tools = _real_tools if _full_after_has_fixed else _tmp_tools
                approx_tokens = estimate_request_tokens_rough(
                    history, system_prompt=_sys_prompt, tools=_tools
                )
                chat_before_tokens = estimate_messages_tokens_rough(msgs)

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
                _old_session_id = session_entry.session_id
                _in_place = bool(getattr(tmp_agent, "_last_compaction_in_place", False))
                if rotated:
                    session_entry.session_id = new_session_id
                    self.session_store._save()
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compress-command",
                    )

                # Rewrite the transcript when EITHER rotation produced a new id
                # OR in-place compaction succeeded. The danger this guards
                # against is the THIRD case: _compress_context could NOT rotate
                # AND was not in-place (e.g. legacy mode but _session_db
                # unavailable / the DB split raised) — there session_id is
                # unchanged for a FAILURE reason, and rewrite_transcript() would
                # DELETE the original messages and replace them with only the
                # compressed summary (permanent data loss #44794, #39704). In
                # in-place mode the unchanged id is SUCCESS, so the rewrite is
                # exactly right (and is the durable write when the throwaway
                # /compress agent has no _session_db of its own).
                if rotated or _in_place:
                    self.session_store.rewrite_transcript(
                        new_session_id, compressed
                    )
                else:
                    logger.warning(
                        "Manual /compress: session rotation did not occur "
                        "(session_id unchanged) and in-place mode is off — "
                        "preserving original transcript instead of overwriting "
                        "it (#44794)."
                    )
                # Whether the STORED transcript actually changed. Downstream
                # feedback must be computed on this basis — when no rewrite
                # happened, the next request resends the ORIGINAL transcript
                # (tool rows included), so an "after" measured over the
                # chat-only `compressed` list would claim a shrink that never
                # occurred (#F2).
                _rewritten = bool(rotated or _in_place)
                if _rewritten:
                    # Reset stored token count — transcript changed, old value
                    # is stale. On a preserved transcript the real provider-
                    # measured count is still valid; zeroing it would destroy
                    # the only tokenizer-truth figure for the next /compress
                    # or /usage (#F4).
                    self.session_store.update_session(
                        session_entry.session_key, last_prompt_tokens=0
                    )
                # After-compression estimates.
                #  • chat_after  — user/hermes turns only (headline + Chat size).
                #  • full_after  — PRE-FLIGHT estimate of the NEXT request: the
                #    REAL fixed overhead (resident agent's system prompt + full
                #    tool schemas, resolved below) + the post-compress transcript.
                #    The old code measured this off the memory-only temp agent,
                #    whose _cached_system_prompt is empty and tools = [memory] —
                #    so it under-reported by the entire ~30k fixed overhead
                #    (e.g. "56,965 -> 3,436"). Now it reflects what's actually
                #    sent next turn.
                #    Basis: the rows the next request will actually carry —
                #    `compressed` after a real rewrite, the ORIGINAL `history`
                #    when the store was preserved (#F2).
                _after_basis = compressed if _rewritten else history
                chat_after_tokens = (
                    estimate_messages_tokens_rough(compressed)
                    if _rewritten
                    else chat_before_tokens
                )
                non_chat_tokens = (
                    estimate_messages_tokens_rough(non_chat_rows)
                    if non_chat_rows
                    else 0
                )
                # Fixed overhead already resolved up front (F3): _sys_prompt /
                # _tools hold the resident agent's real system prompt + tool
                # schemas when available (_full_after_has_fixed=True), else the
                # temp agent's memory-only fallback — the same basis used for
                # `approx_tokens`, so before/after are finally comparable.
                # No-rewrite path skips the estimate entirely: the reply says
                # "unchanged" and never reads an after figure, and estimating
                # over the full tool-heavy history is not free.
                full_after_tokens = (
                    estimate_request_tokens_rough(
                        _after_basis, system_prompt=_sys_prompt, tools=_tools
                    )
                    if _rewritten
                    else approx_tokens
                )
                # Both-axes feedback: chat delta AND the stored tool/system
                # rows the rewrite dropped, so the headline reconciles with
                # the token math (#F1/#F5).
                summary = summarize_manual_compression(
                    msgs,
                    compressed,
                    chat_before_tokens,
                    chat_after_tokens,
                    non_chat_count=non_chat_count,
                    non_chat_tokens=non_chat_tokens,
                    transcript_rewritten=_rewritten,
                    full_before_count=len(history),
                )
                # Granular reconciling breakdown (the same CompactionStats +
                # renderer the auto-compaction announce uses): Messages /
                # Context / per-bucket "Removed from live context" lines with
                # the tool-vs-other sub-split. Built ONLY when the rewrite
                # actually happened; degrades to the enhanced two-line form on
                # ANY failure — a reconcile bug must never break /compress or
                # ship wrong math (same contract as the hygiene announce).
                _granular = None
                _granular_has_wire = False
                if _rewritten and len(compressed) != len(history):
                    try:
                        from agent.compaction_stats import build_hygiene_stats
                        from agent.conversation_compression import (
                            _format_granular_announce,
                        )
                        from agent.provider_model_util import format_provider_model

                        _engine_name = getattr(compressor, "name", None)
                        _stats = build_hygiene_stats(
                            raw_history=history,
                            eligible_msgs=msgs,
                            compressed=compressed,
                            estimator=estimate_messages_tokens_rough,
                            engine_is_lcm=(_engine_name == "lcm"),
                        )
                        _s_ok, _s_why = _stats.validate()
                        if _s_ok:
                            _prov = (
                                runtime_kwargs.get("provider")
                                if isinstance(runtime_kwargs, dict)
                                else None
                            )
                            _model_part = (
                                format_provider_model(_prov, model) if model else ""
                            )
                            if _engine_name == "lcm":
                                _model_part = (
                                    f"{_model_part} · engine: lcm"
                                    if _model_part
                                    else "engine: lcm"
                                )
                            _granular = _format_granular_announce(
                                f"🗜️ {summary['headline']}",
                                _stats,
                                _model_part,
                                False,
                                None,
                                None,
                                basis="stored",
                                # Wire-first (Ace 2026-07-02): when the REAL
                                # provider-measured before-count exists, the
                                # block's Context line carries the wire story
                                # (measured before → next-request estimate)
                                # and the trailing Full-request line below is
                                # skipped as a duplicate.
                                wire_before=real_before_tokens,
                                wire_after=full_after_tokens,
                            )
                            _granular_has_wire = (
                                real_before_tokens > 0 and full_after_tokens > 0
                            )
                            # Honest, store-correct recovery pointer.
                            if _engine_name == "lcm":
                                _granular += (
                                    "\n↩ Nothing lost — original messages preserved "
                                    "in lcm.db (recover with lcm_grep / lcm_expand)"
                                )
                            elif rotated:
                                _granular += (
                                    f"\n↩ previous transcript preserved: "
                                    f"{_old_session_id} (searchable via session_search)"
                                )
                        else:
                            logger.warning(
                                "Manual /compress granular stats reconcile "
                                "failed (%s) — using two-line form",
                                _s_why,
                            )
                    except Exception:
                        logger.warning(
                            "Manual /compress granular stats build failed "
                            "(non-fatal, using two-line form)",
                            exc_info=True,
                        )
                        _granular = None
                        _granular_has_wire = False
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
            # Headline + per-axis lines. When the granular reconciling
            # breakdown built successfully (rewrite happened, stats validated),
            # it REPLACES the headline/chat/dropped lines with the full
            # Messages/Context/Removed-buckets block — the same renderer the
            # auto-compaction announce uses. The Full-request line below still
            # appends (its scope is chat+system+tools, complementary to the
            # granular block's transcript-only Context line).
            if _granular:
                lines = [_granular]
                if focus_topic:
                    lines.append(t("gateway.compress.focus_line", topic=focus_topic))
            else:
                lines = [f"🗜️ {summary['headline']}"]
                if focus_topic:
                    lines.append(t("gateway.compress.focus_line", topic=focus_topic))

                # Line 1 — Chat size. Enhanced mode (tool-heavy stored transcript)
                # renders the reconciling per-axis lines from the summary helper:
                # the chat line + the dropped tool/system rows line. Classic mode
                # keeps the original chat_size locale lines.
                if summary.get("enhanced"):
                    if summary.get("chat_line"):
                        lines.append(summary["chat_line"])
                    if summary.get("dropped_line"):
                        lines.append(summary["dropped_line"])
                elif summary["noop"] and chat_after_tokens == chat_before_tokens:
                    lines.append(
                        t("gateway.compress.chat_size_unchanged", before=chat_before_tokens)
                    )
                else:
                    lines.append(
                        t(
                            "gateway.compress.chat_size",
                            before=chat_before_tokens,
                            after=chat_after_tokens,
                        )
                    )

            # Line 2 — Full request size: what the model is actually sent
            # (chat + system + tools + tool results). The "before" is the REAL
            # provider-measured count from the last live request when we have
            # one (no ~ prefix); otherwise it falls back to the char-based
            # full-request estimate (~ prefix). The "after" is always an
            # estimate — the real post-compression count doesn't exist until
            # the next live API call.
            #
            # Wire-first: when the granular block already rendered the wire
            # story on its Context line (measured before → next-request
            # estimate), this line would be a duplicate — skip it.
            #
            # When the stored transcript was NOT rewritten, the next request
            # resends the same context — say "unchanged" instead of printing a
            # before → after pair whose delta is pure estimator noise (#F2).
            if not _rewritten:
                _fr_before = (
                    f"{real_before_tokens:,}"
                    if real_before_tokens > 0
                    else f"~{approx_tokens:,}"
                )
                lines.append(
                    t("gateway.compress.full_request_unchanged", before=_fr_before)
                )
            elif _granular_has_wire:
                pass  # wire story already on the granular Context line
            elif real_before_tokens > 0:
                lines.append(
                    t(
                        "gateway.compress.full_request_real",
                        before=real_before_tokens,
                        after=full_after_tokens,
                    )
                )
            else:
                lines.append(
                    t(
                        "gateway.compress.full_request_est",
                        before=approx_tokens,
                        after=full_after_tokens,
                    )
                )
            # Honesty note: when no resident agent was available to supply the
            # real fixed overhead, the "after" omits system prompt + tool
            # schemas (~tens of k tokens). Say so rather than under-report.
            # Only relevant when an "after" estimate was actually shown.
            if _rewritten and not _full_after_has_fixed:
                lines.append(t("gateway.compress.full_request_no_fixed"))

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
        session_entry = self.session_store.get_or_create_session(source)
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
                if source.platform == Platform.MATRIX and not allow_all:
                    scoped = []
                    for s in titled:
                        origin = self._gateway_session_origin_for_id(str(s.get("id") or ""))
                        if self._same_matrix_room(source, origin):
                            scoped.append(s)
                    titled = scoped
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
                if source.platform == Platform.MATRIX and not allow_all:
                    scoped = []
                    for s in titled:
                        origin = self._gateway_session_origin_for_id(str(s.get("id") or ""))
                        if self._same_matrix_room(source, origin):
                            scoped.append(s)
                    titled = scoped
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

        # Check if already on that session
        current_entry = self.session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return t("gateway.resume.already_on", name=name)

        # Clear any running agent for this session key
        self._release_running_agent_state(session_key)

        # Switch the session entry to point at the old session
        new_entry = self.session_store.switch_session(session_key, target_id)
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

        # Evict any cached agent for this session so the next message
        # rebuilds with the correct session_id end-to-end — mirrors
        # /branch and /reset. Without this, the cached AIAgent (and its
        # memory provider, which cached `_session_id` during initialize())
        # keeps writing into the wrong session's record. See #6672.
        self._evict_cached_agent(session_key)

        # Get the title for confirmation
        title = await self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = self.session_store.load_transcript(target_id)
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
            include_all, include_unnamed, target = parse_session_listing_args(raw_args)
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)

        if target:
            resume_event = dataclasses.replace(event, text=f"/resume {target}")
            return await self._handle_resume_command(resume_event)

        current_entry = self.session_store.get_or_create_session(source)
        rows = await asyncio.to_thread(
            query_session_listing,
            getattr(self._session_db, "_db", self._session_db),
            source=source.platform.value if source.platform else None,
            current_session_id=current_entry.session_id,
            include_all_sources=include_all,
            include_unnamed=include_unnamed,
            limit=10,
            exclude_sources=["tool"],
        )
        if source.platform == Platform.MATRIX and not include_all:
            rows = [
                row for row in rows
                if self._same_matrix_room(
                    source, self._gateway_session_origin_for_id(str(row.get("id") or ""))
                )
            ]
        return format_gateway_session_listing(
            rows,
            include_source=include_all,
            title="Sessions" if include_unnamed else "Named Sessions",
        )

    async def _try_discord_branch_thread(
        self,
        source: "SessionSource",
        branch_title: str,
        new_session_id: str,
        msg_count: int,
    ) -> Optional[str]:
        """Discord path for /branch: spawn a thread bound to the branch session.

        On Discord, /branch spawns a NEW thread under the parent text channel
        and binds the copied (branch) session to that thread's session key —
        leaving the parent channel on its own session (the parent conversation
        continues in place). Inside a thread already, spawns a SIBLING thread
        under the same parent channel (Discord can't nest threads).

        Returns the new thread id on success. Returns ``None`` to signal the
        caller to fall back to the classic in-place branch (non-Discord, DMs,
        a thread with no resolvable parent, or thread creation failed).
        """
        if source.platform != Platform.DISCORD:
            return None
        adapter = self.adapters.get(Platform.DISCORD) if getattr(self, "adapters", None) else None
        if adapter is None:
            return None
        # Threads only exist in server channels, not DMs.
        if source.chat_type == "dm":
            return None

        # Resolve the parent text channel to host the new thread. In a thread
        # already → sibling under the same parent; in a channel → this channel.
        if source.chat_type == "thread":
            parent_channel_id = source.parent_chat_id
            if not parent_channel_id:
                # No resolvable parent — fall back to classic in-place branch.
                return None
        else:
            parent_channel_id = source.chat_id
        if not parent_channel_id:
            return None

        try:
            new_thread_id = await adapter.create_handoff_thread(
                str(parent_channel_id), branch_title,
            )
        except Exception as exc:
            logger.debug("branch: create_handoff_thread raised: %s", exc, exc_info=True)
            new_thread_id = None
        if not new_thread_id:
            # Thread creation not permitted / failed — classic in-place branch.
            return None

        # Build the thread's session source. For a Discord thread the adapter
        # keys chat_id to the thread id itself (effective_channel), thread_id to
        # the thread id, and parent_chat_id to the hosting channel — mirror that
        # so the session key we bind here matches the one a real user message in
        # the thread will produce (thread sessions are user-shared by default).
        dest_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(new_thread_id),
            chat_type="thread",
            user_id="system:branch",
            user_name="Branch",
            thread_id=str(new_thread_id),
            parent_chat_id=str(parent_channel_id),
        )
        thread_session_key = self._session_key_for_source(dest_source)

        # Make sure a session_store entry exists for the thread key, then
        # re-point it at the copied branch session (mirrors _process_handoff).
        self.session_store.get_or_create_session(dest_source)
        switched = self.session_store.switch_session(thread_session_key, new_session_id)
        if switched is None:
            logger.warning(
                "branch: could not bind thread key %s -> %s",
                thread_session_key, new_session_id,
            )
            return None
        self._clear_session_boundary_security_state(thread_session_key)
        self._evict_cached_agent(thread_session_key)
        try:
            self._release_running_agent_state(thread_session_key)
        except Exception:
            pass

        # Post an intro into the new thread so the branch has a visible anchor.
        # Include a link back to the PARENT conversation (where /branch ran) so
        # the child<->parent relationship is navigable from BOTH sides (the
        # parent channel already got a link INTO this thread).
        try:
            parent_ref = source.thread_id or source.chat_id
            parent_mention = f"<#{parent_ref}>" if parent_ref else t("gateway.branch.thread_parent_fallback")
            intro = t(
                "gateway.branch.thread_intro",
                title=branch_title,
                count=msg_count,
                parent=parent_mention,
            )
            await adapter.send(str(new_thread_id), intro)
        except Exception as exc:
            logger.debug("branch: thread intro send failed: %s", exc)

        return str(new_thread_id)

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
        current_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(current_entry.session_id)
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

        # Branch-point: how many messages the branch inherits from the parent.
        # Everything AFTER this index in the branch transcript is NEW exploration
        # that happened only in the branch — that's the DELTA /merge folds back,
        # so it doesn't re-summarize history the parent already has.
        branch_point_len = len(history)

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
                model_config={"_branched_from": parent_session_id, "_branch_point_len": branch_point_len},
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

        msg_count = len([m for m in history if m.get("role") == "user"])

        # Discord: spawn a NEW thread bound to the branch session and leave the
        # parent channel on its own session (the parent conversation continues
        # in place). Every other platform keeps the classic in-place switch.
        new_thread_id = await self._try_discord_branch_thread(
            source, branch_title, new_session_id, msg_count,
        )
        if new_thread_id:
            thread_mention = f"<#{new_thread_id}>"
            key = (
                "gateway.branch.thread_branched_one"
                if msg_count == 1
                else "gateway.branch.thread_branched_many"
            )
            return t(
                key,
                title=branch_title,
                count=msg_count,
                thread=thread_mention,
                parent=parent_session_id,
                new=new_session_id,
            )

        # Classic in-place branch: switch the current session key to the copy.
        new_entry = self.session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)

    def _build_merge_summarizer(self):
        """Construct a minimal ContextCompressor for /merge's summary call.

        Reuses the auxiliary ``task="compression"`` path (cheap model resolved
        from config) — ``main_runtime`` is only a fallback. Returns None if the
        runtime provider can't be resolved (surfaced as a no-summary merge).
        """
        try:
            from agent.context_compressor import ContextCompressor
            from hermes_cli.runtime_provider import (
                resolve_runtime_provider,
                _get_model_config,
            )
            runtime = resolve_runtime_provider()
            model_cfg = _get_model_config() or {}
            model = (
                (model_cfg.get("default") if isinstance(model_cfg, dict) else None)
                or runtime.get("model")
                or "gpt-4o-mini"
            )
            return ContextCompressor(
                model=model,
                quiet_mode=True,
                base_url=runtime.get("base_url", "") or "",
                api_key=runtime.get("api_key", "") or "",
                provider=runtime.get("provider", "") or "",
                api_mode=runtime.get("api_mode", "") or "",
            )
        except Exception as exc:
            logger.warning("merge: could not build summarizer: %s", exc)
            return None

    def _purge_old_merge_records(self, merges_dir, max_age_days: int = 30) -> None:
        """Delete merge-record .md files older than ``max_age_days``.

        Merge records are ephemeral operational artifacts (the summary's content
        also lives permanently in the target session's transcript). Self-purge on
        every write so the dir never accumulates — no separate cron needed.
        """
        import time
        try:
            cutoff = time.time() - max_age_days * 86400
            for p in merges_dir.glob("*.md"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("merge: purge sweep skipped: %s", exc)

    def _write_merge_record(self, source_title, source_id, target_title,
                            target_id, summary, platform):
        """Write a durable merge-record .md and return its path (or None).

        Location: ``<HERMES_HOME>/merges/<ts>-<source-slug>.md``. Self-purges
        records older than 30 days first. Best-effort — never raises.
        """
        import re as _re
        from datetime import datetime as _dt
        try:
            from hermes_constants import get_hermes_home
            base = get_hermes_home()
        except Exception:
            base = os.path.expanduser("~/.hermes")
        try:
            merges_dir = Path(base) / "merges"
            merges_dir.mkdir(parents=True, exist_ok=True)
            self._purge_old_merge_records(merges_dir)
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            slug = _re.sub(r"[^a-z0-9]+", "-", (source_title or "session").lower()).strip("-")[:40] or "session"
            path = merges_dir / f"{ts}-{slug}.md"
            body = (
                f"# Merge record — {source_title}\n\n"
                f"- When: {_dt.now().isoformat(timespec='seconds')}\n"
                f"- Platform: {platform}\n"
                f"- Source session: {source_title} (`{source_id}`)\n"
                f"- Target session: {target_title} (`{target_id}`)\n\n"
                f"## Summary folded into the target\n\n{summary}\n"
            )
            path.write_text(body, encoding="utf-8")
            return path
        except Exception as exc:
            logger.debug("merge: could not write merge record: %s", exc)
            return None

    async def _list_mergeable_sessions(self, source) -> str:
        """Return the numbered list of the caller's titled sessions (merge targets).

        Mirrors /resume's no-arg listing so /merge with no arg (and not in a
        branched thread) surfaces what names are mergeable instead of a dead no-op.
        """
        try:
            user_source = source.platform.value if source.platform else None
            sessions = await self._session_db.list_sessions_rich(source=user_source, limit=10)
            titled = [s for s in sessions if s.get("title")][:10]
            titled = [
                s for s in titled
                if await self._resume_row_visible(source, s, False)
            ]
            # Don't offer the caller's OWN current session as a merge target.
            try:
                cur = self.session_store.get_or_create_session(source)
                titled = [s for s in titled if s.get("id") != cur.session_id]
            except Exception:
                pass
            if not titled:
                return t("gateway.merge.no_named_sessions")
            lines = [t("gateway.merge.list_header")]
            for idx, s in enumerate(titled[:10], start=1):
                title = s["title"]
                preview = s.get("preview", "")[:40]
                preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
                lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
            lines.append(t("gateway.merge.list_footer"))
            return "\n".join(lines)
        except Exception as e:
            logger.debug("merge: failed to list titled sessions: %s", e)
            return t("gateway.resume.list_failed", error=e)

    async def _resolve_merge_target(self, source, name):
        """Resolve a /merge <name> arg to a target session id + title.

        Mirrors /resume: numeric → index into the caller's titled list; else a
        direct session-id lookup, then a title lookup. Returns (target_id, title)
        or (None, error_string).
        """
        if name.isdigit():
            try:
                user_source = source.platform.value if source.platform else None
                sessions = await self._session_db.list_sessions_rich(source=user_source, limit=10)
                titled = [s for s in sessions if s.get("title")][:10]
                titled = [
                    s for s in titled
                    if await self._resume_row_visible(source, s, False)
                ]
                try:
                    cur = self.session_store.get_or_create_session(source)
                    titled = [s for s in titled if s.get("id") != cur.session_id]
                except Exception:
                    pass
            except Exception as e:
                return None, t("gateway.resume.list_failed", error=e)
            index = int(name)
            if index < 1 or index > len(titled):
                return None, t("gateway.resume.out_of_range", index=index)
            target = titled[index - 1]
            return target.get("id"), (target.get("title") or name)

        # Non-numeric: direct id lookup first, then title.
        session = await self._session_db.get_session(name)
        if session:
            return session["id"], (session.get("title") or name)
        target_id = await self._session_db.resolve_session_by_title(name)
        if not target_id:
            return None, t("gateway.merge.not_found", name=name)
        title = await self._session_db.get_session_title(target_id) or name
        return target_id, title

    async def _merged_targets_for(self, source_session_id):
        """Return the set of target session ids this source has already merged into.

        Recorded in the source session's ``model_config._merged_into`` list.
        Best-effort — returns an empty set on any read error.
        """
        try:
            row = await self._session_db.get_session(source_session_id)
        except Exception:
            return set()
        if not row:
            return set()
        mc = row.get("model_config")
        if isinstance(mc, str) and mc:
            try:
                import json as _json
                mc = _json.loads(mc)
            except Exception:
                return set()
        if not isinstance(mc, dict):
            return set()
        merged = mc.get("_merged_into")
        if isinstance(merged, list):
            return {str(x) for x in merged}
        return set()

    async def _merge_already_done(self, source_session_id, target_id) -> bool:
        """Whether this (source → target) merge has already been recorded."""
        return str(target_id) in await self._merged_targets_for(source_session_id)

    def _merge_claim_lock(self, source_session_id, target_id):
        """Return a process-local asyncio.Lock for this (source → target) claim.

        Serializes the re-check → record → append critical section so two
        overlapping /merge calls for the same pair can't both append. Lazily
        created and cached; the gateway runs one event loop so a plain dict of
        locks is sufficient (no cross-process concurrency on a single session).
        """
        import asyncio as _asyncio
        locks = getattr(self, "_merge_claim_locks", None)
        if locks is None:
            locks = {}
            self._merge_claim_locks = locks
        key = f"{source_session_id}->{target_id}"
        lock = locks.get(key)
        if lock is None:
            lock = _asyncio.Lock()
            locks[key] = lock
        return lock

    async def _record_merge_done(self, source_session_id, target_id) -> bool:
        """Append ``target_id`` to the source's ``model_config._merged_into`` list.

        Idempotency ledger for /merge — read-modify-write of the source session's
        model_config. Returns ``True`` only when the marker is CONFIRMED durably
        recorded (re-read after write); ``False`` on any failure. The caller must
        NOT append the fold when this returns False, otherwise a marker-less fold
        could be duplicated by a later retry (Greptile P1).
        """
        try:
            row = await self._session_db.get_session(source_session_id)
            mc = (row or {}).get("model_config")
            if isinstance(mc, str) and mc:
                import json as _json
                try:
                    mc = _json.loads(mc)
                except Exception:
                    mc = {}
            if not isinstance(mc, dict):
                mc = {}
            merged = mc.get("_merged_into")
            if not isinstance(merged, list):
                merged = []
            if str(target_id) not in {str(x) for x in merged}:
                merged.append(str(target_id))
            mc["_merged_into"] = merged
            import json as _json
            await self._session_db.update_session_meta(source_session_id, _json.dumps(mc))
        except Exception as exc:
            logger.warning("merge: could not record merge ledger: %s", exc)
            return False
        # Confirm the marker actually persisted before we trust it — a silently
        # failed/locked write must NOT let the fold proceed marker-less.
        try:
            return str(target_id) in await self._merged_targets_for(source_session_id)
        except Exception:
            return False

    async def _unrecord_merge_done(self, source_session_id, target_id) -> None:
        """Remove ``target_id`` from the source's ledger (rollback on failed fold)."""
        try:
            row = await self._session_db.get_session(source_session_id)
            mc = (row or {}).get("model_config")
            if isinstance(mc, str) and mc:
                import json as _json
                try:
                    mc = _json.loads(mc)
                except Exception:
                    return
            if not isinstance(mc, dict):
                return
            merged = mc.get("_merged_into")
            if not isinstance(merged, list):
                return
            new = [x for x in merged if str(x) != str(target_id)]
            if new != merged:
                mc["_merged_into"] = new
                import json as _json
                await self._session_db.update_session_meta(source_session_id, _json.dumps(mc))
        except Exception as exc:
            logger.debug("merge: could not roll back merge ledger: %s", exc)

    async def _handle_merge_command(self, event: MessageEvent) -> str:
        """Handle /merge [name] — fold a summary of THIS session into a target.

        - ``/merge <name>`` (any platform): summarize the current session and fold
          the summary into the session titled/ided ``<name>``. Owner-guarded
          exactly like /resume — you can only merge into sessions you own.
        - ``/merge`` (no arg) inside a branched Discord thread: target = the
          thread's parent session, then archive+lock the thread.
        - ``/merge`` (no arg) elsewhere: list the caller's recent titled sessions
          (mergeable targets), do nothing.

        The current session is READ-ONLY here — it is summarized, never ended or
        switched; the user keeps talking in it.
        """
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

        source = event.source
        name = event.get_command_args().strip()
        # Strip surrounding quotes/brackets like /resume does.
        if len(name) >= 2 and (
            (name[0] == "[" and name[-1] == "]")
            or (name[0] == '"' and name[-1] == '"')
            or (name[0] == "'" and name[-1] == "'")
        ):
            name = name[1:-1].strip()

        current_entry = self.session_store.get_or_create_session(source)
        source_session_id = current_entry.session_id

        # --- Resolve the TARGET session and whether this is the thread form. ---
        is_thread_form = False
        thread_adapter = None
        target_id = None
        target_title = None
        branch_point_len = None  # thread form: index into source_history where new exploration begins

        if not name:
            # No arg. In a branched Discord thread → target = parent, archive after.
            branch_parent = None
            if source.platform == Platform.DISCORD and source.chat_type == "thread" and source.thread_id:
                try:
                    row = await self._session_db.get_session(source_session_id)
                except Exception:
                    row = None
                if row:
                    branch_parent = row.get("parent_session_id")
                    mc = row.get("model_config")
                    if isinstance(mc, str) and mc:
                        try:
                            import json as _json
                            mc = _json.loads(mc)
                        except Exception:
                            mc = {}
                    if isinstance(mc, dict):
                        if not branch_parent:
                            branch_parent = mc.get("_branched_from")
                        # Delta boundary: only summarize turns AFTER the branch
                        # point so we don't re-fold history the parent already has.
                        _bpl = mc.get("_branch_point_len")
                        if isinstance(_bpl, int) and _bpl >= 0:
                            branch_point_len = _bpl
            if branch_parent:
                is_thread_form = True
                thread_adapter = self.adapters.get(Platform.DISCORD) if getattr(self, "adapters", None) else None
                target_id = branch_parent
                target_title = await self._session_db.get_session_title(target_id) or "parent"
            else:
                # Not a branched thread → list mergeable sessions.
                return await self._list_mergeable_sessions(source)
        else:
            target_id, resolved = await self._resolve_merge_target(source, name)
            if not target_id:
                return resolved  # error/not-found string
            target_title = resolved

        # Don't merge a session into itself.
        if target_id == source_session_id:
            return t("gateway.merge.same_session")

        # --- Owner guard: a merge WRITES into the target (bigger than /resume's
        # read), so refuse cross-owner targets fail-closed, same as /resume. ---
        if not is_thread_form:
            try:
                allowed = await self._resume_target_allowed(source, target_id)
            except Exception:
                allowed = False
            if not allowed:
                return t("gateway.merge.blocked_not_owner", name=target_title)

        # --- Idempotency guard (Greptile P1): refuse a duplicate fold of the
        # SAME source into the SAME target. A retry, double-submit, or a
        # reopened/re-invoked thread would otherwise append the branch summary
        # to the parent twice and skew later agent behavior. We record each
        # completed (source → target) merge in the source's model_config
        # `_merged_into` list; a repeat (source, target) is a no-op. Merging the
        # same source into a DIFFERENT target later is still allowed. ---
        already = await self._merge_already_done(source_session_id, target_id)
        if already:
            return t("gateway.merge.already_merged", title=target_title)

        # --- Summarize the CURRENT session (read-only). ---
        source_history = self.session_store.load_transcript(source_session_id)
        if not source_history:
            return t("gateway.merge.no_conversation")
        source_title = await self._session_db.get_session_title(source_session_id) or "session"

        # Thread form: only summarize the DELTA — the turns that happened in the
        # branch AFTER the branch point. A branched thread is a full copy of the
        # parent up to the branch, plus the new exploration; folding the whole
        # copy back re-summarizes context the parent already has. The named form
        # has no shared prefix, so it summarizes the whole source session.
        summarize_history = source_history
        if is_thread_form and isinstance(branch_point_len, int) and branch_point_len >= 0:
            # The delta is authoritative once we have a valid branch point: even
            # branch_point_len == len(source_history) (zero new turns) must yield
            # an EMPTY delta → no_new_turns, NOT a re-summary of the whole copy.
            summarize_history = source_history[branch_point_len:]
            if not summarize_history:
                return t("gateway.merge.no_new_turns")
        if not summarize_history:
            return t("gateway.merge.no_new_turns")

        summarizer = self._build_merge_summarizer()
        summary = None
        if summarizer is not None:
            try:
                summary = await asyncio.to_thread(
                    summarizer._generate_summary,
                    summarize_history,
                    f"merging session '{source_title}' into '{target_title}'",
                )
            except Exception as exc:
                logger.warning("merge: summary generation failed: %s", exc)
                summary = None
        if summary:
            summary = summary.strip()
        if not summary:
            return t("gateway.merge.no_summary")

        # --- Layer 1: fold ONE labeled user-role message into the TARGET. ---
        # Make the claim ATOMIC (Greptile P1): hold a per-(source→target)
        # asyncio lock across re-check → record → append so two overlapping
        # /merge calls can't both pass the duplicate check and both append. The
        # gateway is single-event-loop, so this lock fully serializes the claim.
        lock = self._merge_claim_lock(source_session_id, target_id)
        async with lock:
            # Authoritative re-check INSIDE the lock — the earlier pre-summary
            # check is only a cheap fast-path; this is the one that gates the write.
            if await self._merge_already_done(source_session_id, target_id):
                return t("gateway.merge.already_merged", title=target_title)
            recorded = await self._record_merge_done(source_session_id, target_id)
            if not recorded:
                return t("gateway.merge.ledger_failed")
            fold_key = "gateway.merge.fold_body_thread" if is_thread_form else "gateway.merge.fold_body"
            fold_text = t(fold_key, title=source_title, summary=summary)
            try:
                await self._session_db.append_message(
                    session_id=target_id,
                    role="user",
                    content=fold_text,
                )
            except Exception as exc:
                logger.error("merge: failed to append fold to target %s: %s", target_id, exc)
                # Roll back the ledger entry — the fold never landed, so a retry
                # must be allowed to try again rather than being wrongly refused.
                await self._unrecord_merge_done(source_session_id, target_id)
                return t("gateway.merge.fold_failed", error=exc)

        # --- Layer 2: durable .md record (self-purging). ---
        record_path = self._write_merge_record(
            source_title, source_session_id, target_title, target_id,
            summary, source.platform.value if source.platform else "gateway",
        )

        # --- Layer 1b: mark the SOURCE session as merged too. ---
        # /merge records the merge on the TARGET (the _merged_into ledger) but
        # historically wrote nothing back to the SOURCE — so if you kept talking
        # in the branch (or resumed it later), the agent had no idea a merge ever
        # happened. Append a small labeled marker into the source transcript so a
        # resumed branch knows its exploration was already folded into the parent.
        # Best-effort; never blocks the merge.
        try:
            from hermes_time import now as _merge_now
            _ts = _merge_now().strftime("%Y-%m-%d %H:%M")
        except Exception:
            from datetime import datetime as _dt
            _ts = _dt.now().strftime("%Y-%m-%d %H:%M")
        try:
            await self._session_db.append_message(
                session_id=source_session_id,
                role="user",
                content=t(
                    "gateway.merge.source_marker" if is_thread_form else "gateway.merge.source_marker_named",
                    target=target_title, when=_ts,
                ),
            )
        except Exception as exc:
            logger.debug("merge: source-marker append skipped: %s", exc)
        # Evict the source's cached agent so the marker is live if it keeps going.
        try:
            src_key = self._session_key_for_source(source)
            self._evict_cached_agent(src_key)
        except Exception as exc:
            logger.debug("merge: source agent eviction skipped: %s", exc)

        # Evict the target's cached agent so the fold is live on its next turn.
        target_entry = self.session_store.lookup_by_session_id(target_id)
        if target_entry is not None:
            try:
                self._evict_cached_agent(target_entry.session_key)
            except Exception as exc:
                logger.debug("merge: target agent eviction skipped: %s", exc)

        # --- Layer 3: visible note posted where the fold LANDED — the TARGET
        # session's own origin channel/thread — for BOTH forms. (Previously the
        # thread form posted to source.parent_chat_id, the Discord *hosting
        # channel*, which for a thread-branched-from-a-thread is NOT where the
        # parent SESSION lives — so the note showed up in the wrong place.) We
        # resolve the target's real origin and post there so the "merged in" note
        # always lands in the conversation that just received the summary.
        #
        # The note is INTENTIONALLY terse: it says WHAT happened (context added),
        # HOW MUCH (N user turns folded), and WHERE the detail is (the .md path).
        # The full summary is already in the agent's context — dumping it into the
        # channel as a wall of text is noise, not signal (Ace, 2026-07-08). ---
        turns_folded = sum(1 for m in summarize_history if m.get("role") == "user")
        path_str = str(record_path) if record_path else ""
        md_note = t("gateway.merge.note_record_suffix", path=path_str) if path_str else ""
        try:
            target_origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            target_origin = None
        posted_note = False
        if isinstance(target_origin, SessionSource) and target_origin.platform:
            dest_adapter = self.adapters.get(target_origin.platform) if getattr(self, "adapters", None) else None
            dest_chat = target_origin.thread_id or target_origin.chat_id
            if dest_adapter is not None and dest_chat:
                try:
                    note = t(
                        "gateway.merge.target_note" if is_thread_form else "gateway.merge.target_note_named",
                        source=source_title, turns=turns_folded, record=md_note,
                    )
                    await dest_adapter.send(str(dest_chat), note)
                    posted_note = True
                except Exception as exc:
                    logger.debug("merge: target-origin note send failed: %s", exc)
        # Fallback (thread form only): if the target's origin couldn't be resolved
        # (e.g. the parent session isn't live in the store), post to the thread's
        # hosting channel so the merge is still visible somewhere sensible.
        if not posted_note and is_thread_form and thread_adapter is not None and source.parent_chat_id:
            try:
                note = t("gateway.merge.target_note", source=source_title,
                         turns=turns_folded, record=md_note)
                await thread_adapter.send(str(source.parent_chat_id), note)
            except Exception as exc:
                logger.debug("merge: parent-channel fallback note send failed: %s", exc)

        # --- Archive the thread (thread form only). ---
        archived = False
        if is_thread_form and thread_adapter is not None:
            try:
                archived = await thread_adapter.archive_thread(str(source.thread_id), lock=True)
            except Exception as exc:
                logger.debug("merge: archive_thread failed: %s", exc)

        # --- Confirm to the caller. ---
        md_part = t("gateway.merge.md_suffix", path=path_str) if path_str else ""
        if is_thread_form:
            confirm_key = "gateway.merge.merged_thread" if archived else "gateway.merge.merged_thread_no_archive"
            return t(confirm_key, title=target_title, md=md_part)
        return t("gateway.merge.merged_named", title=target_title, md=md_part)

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

    def _compact_account_limit_lines(self) -> list:
        """Build the compact '📈 Account limits' block (one line per subscription).

        DRY: loads the SAME ~/.hermes/scripts/claude_usage_lib.py that powers the
        full /claude-usage report and calls its render_compact_lines() — so the
        set of subscriptions (all Claude subs + Codex) tracks automatically with
        the registry. Returns [] (header omitted) when nothing is available so
        the caller can fall back to the single-provider snapshot. Never raises.
        """
        try:
            import importlib.util
            import os
            import sys

            lib_path = os.path.expanduser("~/.hermes/scripts/claude_usage_lib.py")
            if not os.path.isfile(lib_path):
                return []
            # Cache the loaded module under a stable sys.modules key so the
            # library's top-level imports (subprocess, sqlite3, etc.) are only
            # executed once instead of re-run on every /usage call (Greptile P2).
            _MOD_KEY = "_hermes_claude_usage_lib"
            mod = sys.modules.get(_MOD_KEY)
            if mod is None:
                spec = importlib.util.spec_from_file_location(_MOD_KEY, lib_path)
                if spec is None or spec.loader is None:
                    return []
                mod = importlib.util.module_from_spec(spec)
                sys.modules[_MOD_KEY] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    sys.modules.pop(_MOD_KEY, None)  # don't cache a half-loaded module
                    raise
            render = getattr(mod, "render_compact_lines", None)
            if render is None:
                return []
            sub_lines = render() or []
            if not sub_lines:
                return []
            return ["📈 **Account limits**", *sub_lines]
        except Exception:
            return []

    def _context_breakdown_lines(self, agent, source) -> list:
        """Render the per-category context breakdown for /usage.

        Estimated (chars/4) — same engine the desktop popover uses (upstream
        PR #54907/#55204; reuses agent.context_breakdown — no new tool/engine).
        Returns an empty list and never raises on failure so /usage stays robust.
        """
        try:
            from agent.context_breakdown import compute_session_context_breakdown

            history: list = []
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

    def _render_last_turn_card(self, source, thin_snap, fallback_label=None, compressions=None) -> list:
        """Render the rich /context last-turn card for the invoking channel, or a
        reworded thin fallback. Returns a list of lines (may be empty).

        PRD usage-format-codex Part A: /usage's last-turn section reuses the SAME
        renderer /context uses (plugins.blackbox.last_turn) so the numbers are
        byte-identical. Channel is passed EXPLICITLY (event.source) and the
        returned record's channel is verified to match — a found-but-wrong-channel
        record falls back + WARNs, never renders confidently-wrong cross-channel
        numbers (D-7). `thin_snap` is the eviction-safe get_last_turn_usage dict
        (or the resident agent's session counters, or None) used for the fallback
        when blackbox has no matching turn. `fallback_label` overrides the thin
        fallback's parenthetical so the header is honest at each call site (the
        persisted branch says "agent not resident"; the resident branch says
        "session totals").
        """
        platform = source.platform.value if source and source.platform else ""
        chat_id = str(source.chat_id) if source and source.chat_id is not None else ""
        rec = None
        render_last_turn_record = None
        try:
            from plugins.blackbox.last_turn import (
                compute_last_turn_record,
                render_last_turn_record,
            )
            rec = compute_last_turn_record(platform, chat_id)
        except Exception as e:
            logger.warning("usage last-turn card: blackbox compute failed (%s); using fallback", e)
            rec = None

        if rec and rec.get("found") and render_last_turn_record is not None:
            # Channel-match guard (D-7): a non-empty channel request must return
            # THIS channel's row. If the store handed back another channel's row,
            # do NOT render it — fall back loudly.
            if platform and chat_id:
                if str(rec.get("platform", "")) == platform and str(rec.get("chat_id", "")) == chat_id:
                    try:
                        return render_last_turn_record(rec, compressions=compressions)
                    except Exception as e:
                        logger.warning("usage last-turn card: render failed (%s); using fallback", e)
                else:
                    logger.warning(
                        "usage last-turn card: blackbox returned channel %s/%s != requested %s/%s; "
                        "falling back to thin snapshot",
                        rec.get("platform"), rec.get("chat_id"), platform, chat_id,
                    )
            else:
                # No channel bound (shouldn't happen for a messaging /usage) — the
                # global-newest rec is acceptable; render it.
                try:
                    return render_last_turn_record(rec, compressions=compressions)
                except Exception as e:
                    logger.warning("usage last-turn card: render failed (%s); using fallback", e)

        # Fallback: blackbox unavailable / no turn for this channel / mismatch.
        # Reworded thin snapshot in the same vocabulary (uncached + cache + a
        # labelled total). Reasoning is folded into the output total so "Total"
        # means the same magnitude as the rich card's billed-out (INV-4).
        if not thin_snap:
            return []
        logger.warning("usage last-turn card: degraded to thin get_last_turn_usage snapshot")

        def _as_int(v):
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0
        lt_in = _as_int(thin_snap.get("input_tokens"))
        lt_out = _as_int(thin_snap.get("output_tokens"))
        lt_cr = _as_int(thin_snap.get("cache_read_tokens"))
        lt_cw = _as_int(thin_snap.get("cache_write_tokens"))
        lt_rsn = _as_int(thin_snap.get("reasoning_tokens"))
        in_billed = lt_in + lt_cr + lt_cw
        out_billed = lt_out + lt_rsn  # fold reasoning into the output total
        out_label = fallback_label or "persisted; agent not resident"
        out_lines = [f"📊 **Last turn** ({out_label})"]
        if in_billed:
            out_lines.append(
                f"• Tokens in: {in_billed:,} billed "
                f"({lt_cr:,} cache-read + {lt_cw:,} cache-write + {lt_in:,} uncached)"
            )
        if out_billed:
            out_lines.append(f"• Tokens out: {out_billed:,} billed")
        out_lines.append(f"• Total (billed in+out): {in_billed + out_billed:,}")
        return out_lines

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns)
        so that rate limits, cost estimates, and detailed token breakdowns are
        available whenever the user asks, not only while the agent is running.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        source = event.source
        session_key = self._session_key_for_source(source)

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
                _entry_for_billing = self.session_store.get_or_create_session(source)
                persisted = await self._session_db.get_session(_entry_for_billing.session_id) or {}
            except Exception:
                persisted = {}
            provider = provider or persisted.get("billing_provider")
            base_url = base_url or persisted.get("billing_base_url")

        # Account limits — the DRY compact multi-subscription block (one line per
        # sub: all Claude subs + Codex), sourced from the SAME claude_usage_lib
        # render the full /claude-usage uses, so add/remove a subscription and
        # both surfaces track automatically. Falls back to the single-provider
        # snapshot when the shared lib isn't available. Off the event loop;
        # fail-open (account_lines stays []).
        account_lines: list[str] = []
        credits_lines: list[str] = []
        try:
            account_lines = await asyncio.to_thread(self._compact_account_limit_lines)
        except Exception:
            account_lines = []
        if not account_lines and provider:
            # Fallback: the resident provider's own snapshot (legacy single-sub).
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

            # Rate limits (provider throttling headroom) are surfaced DOWN next to
            # the Account-limits section (both are "how close am I to a ceiling"),
            # not at the top — see below. Render them as a small block rather than
            # a cramped pipe-separated row so the ceiling sections stay readable.
            rate_limit_lines: list[str] = []
            rl_state = agent.get_rate_limit_state()
            if rl_state and rl_state.has_data:
                from agent.rate_limit_tracker import format_rate_limit_compact
                _compact_rl = format_rate_limit_compact(rl_state)
                _rl_parts = [p.strip() for p in str(_compact_rl).split("|") if p.strip()]
                if _rl_parts:
                    rate_limit_lines.append(t("gateway.usage.rate_limits", state="").rstrip())
                    rate_limit_lines.extend(f"• {p}" for p in _rl_parts)

            # The full last-turn card (PRD usage-format-codex Part A) is the SAME
            # renderer /context uses, and already carries Model / Agent / Session /
            # API Calls / tokens / Compressions — so the old session header +
            # Model/Total/API-calls lines were redundant with it and were removed
            # (Ace, 2026-06-30). The card (cost, tokens in/out with finished/
            # unfinished + uncached, context window, cached, compressions, session)
            # follows directly.

            # The rich /context last-turn card for THIS channel (replaces the old
            # hand-built input/output + char/4 composition block). When the
            # blackbox store has no turn recorded for this channel yet (e.g. the
            # agent is resident but its first turn hasn't landed in the store, or
            # tests), fall back to the resident agent's OWN session counters so the
            # card is never emptier than the pre-card display. The live compressor's
            # compression_count is threaded into the card (• Compressions: N row,
            # right after Cached) instead of being appended orphaned below.
            def _as_int(v):
                try:
                    return int(v or 0)
                except (TypeError, ValueError):
                    return 0
            agent_thin = {
                "input_tokens": _as_int(getattr(agent, "session_input_tokens", 0)),
                "output_tokens": _as_int(getattr(agent, "session_output_tokens", 0)),
                "cache_read_tokens": _as_int(getattr(agent, "session_cache_read_tokens", 0)),
                "cache_write_tokens": _as_int(getattr(agent, "session_cache_write_tokens", 0)),
                "reasoning_tokens": _as_int(getattr(agent, "session_reasoning_tokens", 0)),
            }
            ctx = agent.context_compressor
            _comp_count = _as_int(getattr(ctx, "compression_count", 0))

            # Per-category context breakdown (estimated — chars/4 heuristic) goes
            # FIRST (Ace 2026-06-30): "where is my CURRENT context budget going"
            # (system prompt / tools / rules / skills / MCP / subagents / memory /
            # conversation) frames the rest. Same engine the desktop popover uses
            # (upstream PR #54907/#55204). Fail-open: error → no breakdown.
            breakdown_lines = self._context_breakdown_lines(agent, source)
            if breakdown_lines:
                lines.extend(breakdown_lines)

            # The full last-turn card (PRD usage-format-codex Part A) — the SAME
            # renderer /context uses — answers "what the last turn cost". When the
            # blackbox store has no turn for this channel yet, fall back to the
            # resident agent's OWN session counters so the card is never emptier
            # than the pre-card display. compression_count is threaded in (•
            # Compressions: N row, after Cached).
            try:
                card = self._render_last_turn_card(
                    source, agent_thin,
                    fallback_label="session totals; first turn not yet recorded",
                    compressions=_comp_count,
                )
            except Exception:
                card = []
            if card:
                # The card carries a leading blank line — keep it as a separator
                # from the breakdown above, but strip it when the card is the very
                # first block (no breakdown rendered) so /usage doesn't open blank.
                if not breakdown_lines and card and card[0] == "":
                    card = card[1:]
                lines.extend(card)

            # Rate limits + Account limits together — both answer "how close am I
            # to a ceiling": rate limits = provider throttling headroom (this
            # minute/hour), account limits = subscription quota (5h/7d windows).
            if rate_limit_lines or account_lines:
                lines.append("")
            if rate_limit_lines:
                lines.extend(rate_limit_lines)
            if rate_limit_lines and account_lines:
                lines.append("")
            if account_lines:
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)

            return "\n".join(lines)

        # No agent at all -- check session history for a rough count
        session_entry = self.session_store.get_or_create_session(source)

        # Eviction-safe last-turn card: even with no resident agent, render the
        # full /context last-turn card for THIS channel from the blackbox store
        # (PRD usage-format-codex Part A). Falls back to the thin persisted
        # get_last_turn_usage snapshot (reworded) when blackbox has no matching
        # turn — the helper handles both + the channel-match guard.
        last_turn_lines: list[str] = []
        thin_snap = None
        if getattr(self, "_session_db", None) is not None:
            try:
                # _session_db is an AsyncSessionDB facade (upstream d153918f1):
                # every method is offloaded via asyncio.to_thread and returns a
                # coroutine, so this MUST be awaited or `thin_snap` is a coroutine
                # (truthy) and downstream .get(...) blows up.
                thin_snap = await self._session_db.get_last_turn_usage(session_entry.session_id)
            except Exception:
                thin_snap = None
        try:
            last_turn_lines = self._render_last_turn_card(source, thin_snap)
        except Exception:
            last_turn_lines = []

        history = self.session_store.load_transcript(session_entry.session_id)
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
            if last_turn_lines:
                lines.append("")
                lines.extend(last_turn_lines)
            if account_lines:
                lines.append("")
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)
            return "\n".join(lines)
        if last_turn_lines:
            if account_lines:
                last_turn_lines.append("")
                last_turn_lines.extend(account_lines)
            if credits_lines:
                last_turn_lines.append("")
                last_turn_lines.extend(credits_lines)
            return "\n".join(last_turn_lines)
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

        args = event.get_command_args().strip().lower()
        resolve_all = "all" in args

        count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all)
        if not count:
            return t("gateway.deny.no_pending")

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info("User denied %d dangerous command(s) via /deny", count)
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
