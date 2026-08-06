"""/agents, /background, /stop, /restart, /status, /context slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Union
import asyncio
import dataclasses
import hashlib
import os
import time

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply
from gateway.platforms.base import MessageEvent
from utils import atomic_json_write

from gateway.slash_commands._shared import _clean_str, _int_value, logger

class AgentsOpsCommandsMixin:
    """/agents, /background, /stop, /restart, /status, /context handlers."""

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

    async def _handle_context_command(self, event: MessageEvent) -> str:
        """Handle /context — the dedicated context-window view.

        /status shows a one-line ``used / total`` summary; this command is the
        deep view: a usage gauge, auto-compression threshold and headroom,
        compression count and last savings, and cumulative throughput — the last
        clearly labelled as throughput, NOT context size.

        Resolves from the running agent (mid-turn), then the cached agent
        (between turns), then the SessionStore/SessionDB metadata for a gauge
        even when no agent is resident. Falls back to a transcript estimate only
        as a last resort.

        ``/context all`` appends the expanded per-skill / per-toolset cost
        listings (requires a resident agent).
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        source = event.source
        session_key = self._session_key_for_source(source)
        session_entry = await self.async_session_store.get_or_create_session(source)
        expanded = event.get_command_args().strip().lower() in {"all", "full", "details"}

        # Try running agent first (mid-turn), then cached agent (between turns).
        agent = self._running_agents.get(session_key)
        if not agent or agent is _AGENT_PENDING_SENTINEL:
            cache_lock = getattr(self, "_agent_cache_lock", None)
            cache = getattr(self, "_agent_cache", None)
            if cache_lock is not None and cache is not None:
                try:
                    with cache_lock:
                        cached = cache.get(session_key)
                    if cached:
                        agent = cached[0]
                except Exception:
                    agent = None
        has_agent = bool(agent) and agent is not _AGENT_PENDING_SENTINEL

        ctx = getattr(agent, "context_compressor", None) if has_agent else None

        # Resolve current-context size + window with cascading fallbacks.
        #   used  : compressor.last_prompt_tokens → SessionStore.last_prompt_tokens
        #   model : agent.model → SessionDB row model
        #   window: compressor.context_length → get_model_context_length(model)
        used = 0
        context_length = 0
        if ctx is not None:
            used = getattr(ctx, "last_prompt_tokens", 0) or 0
            context_length = getattr(ctx, "context_length", 0) or 0

        model_name = _clean_str(getattr(agent, "model", "")) if has_agent else ""

        if not used:
            used = _int_value(getattr(session_entry, "last_prompt_tokens", 0))

        if not model_name and self._session_db:
            try:
                row = await self._session_db.get_session(session_entry.session_id) or {}
                if isinstance(row, dict):
                    model_name = _clean_str(row.get("model", ""))
            except Exception:
                model_name = ""

        if not context_length and model_name:
            try:
                from agent.model_metadata import get_model_context_length

                context_length = _int_value(
                    await asyncio.to_thread(get_model_context_length, model_name)
                )
            except Exception:
                context_length = 0

        # Gauge path: real current-context figure
        if used > 0 and context_length > 0:
            pct = min(100.0, used / context_length * 100)
            headroom = max(0, context_length - used)
            BAR_WIDTH = 24
            filled = int(round(pct / 100 * BAR_WIDTH))
            bar = "█" * max(0, filled) + "░" * max(0, BAR_WIDTH - filled)

            lines = [
                t("gateway.context.header"),
                "",
                t("gateway.context.model", model=model_name or "?"),
                t("gateway.context.window", total=f"{context_length:,}"),
                t(
                    "gateway.context.in_use",
                    used=f"{used:,}",
                    total=f"{context_length:,}",
                    pct=f"{pct:.0f}",
                ),
                t("gateway.context.bar", bar=bar),
                t("gateway.context.headroom", headroom=f"{headroom:,}"),
            ]

            # Full view — compression / throughput need the live agent.
            if ctx is not None:
                threshold = getattr(ctx, "threshold_tokens", 0) or 0
                threshold_pct = (getattr(ctx, "threshold_percent", 0) or 0) * 100
                lines.append("")
                if threshold > 0:
                    if used >= threshold:
                        lines.append(
                            t(
                                "gateway.context.over_threshold",
                                threshold=f"{threshold:,}",
                                threshold_pct=f"{threshold_pct:.0f}",
                            )
                        )
                    else:
                        lines.append(
                            t(
                                "gateway.context.threshold",
                                threshold=f"{threshold:,}",
                                threshold_pct=f"{threshold_pct:.0f}",
                                to_go=f"{threshold - used:,}",
                            )
                        )
                compressions = getattr(ctx, "compression_count", 0) or 0
                lines.append(t("gateway.context.compressions", count=compressions))
                if compressions:
                    savings = getattr(ctx, "_last_compression_savings_pct", None)
                    if savings is not None:
                        lines.append(
                            t("gateway.context.last_savings", savings=f"{savings:.0f}")
                        )

                api_calls = getattr(agent, "session_api_calls", 0) or 0
                input_tokens = getattr(agent, "session_input_tokens", 0) or 0
                output_tokens = getattr(agent, "session_output_tokens", 0) or 0
                reasoning_tokens = getattr(agent, "session_reasoning_tokens", 0) or 0
                total_tokens = getattr(agent, "session_total_tokens", 0) or 0
                lines.append("")
                lines.append(
                    t("gateway.context.totals_header", calls=api_calls)
                )
                lines.append(
                    t(
                        "gateway.context.totals_line",
                        input=f"{input_tokens:,}",
                        output=f"{output_tokens:,}",
                        reasoning=f"{reasoning_tokens:,}",
                    )
                )
                lines.append(t("gateway.context.total_billed", total=f"{total_tokens:,}"))
                lines.append(t("gateway.context.throughput_note"))
            else:
                lines.append("")
                lines.append(t("gateway.context.detail_after_first"))

            # Per-category estimated breakdown (+ optional expanded listings).
            # Same chars/4 engine the desktop popover and /usage use; plain
            # text (no glyph grid — monospace isn't guaranteed on messaging
            # platforms). Fail-open: rendering errors never break /context.
            if has_agent:
                breakdown = await asyncio.to_thread(
                    self._context_breakdown_block, agent, source, expanded
                )
                if breakdown:
                    lines.append("")
                    lines.extend(breakdown)

            return "\n".join(lines)

        # Last resort: rough estimate from transcript
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough

            msgs = [
                m
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            approx = estimate_messages_tokens_rough(msgs)
            return "\n".join(
                [
                    t("gateway.context.header"),
                    "",
                    t(
                        "gateway.context.estimated",
                        count=f"{approx:,}",
                        messages=len(msgs),
                    ),
                    t("gateway.context.detail_after_first"),
                ]
            )
        return t("gateway.context.no_data")

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

        # Background (async) delegations — delegate_task(background=true).
        # Live per-child activity comes from the registry's progress sampler
        # (#51690): api calls, current tool, seconds since last activity.
        delegations: list[dict] = []
        try:
            from tools.async_delegation import list_async_delegations
            delegations = [
                d for d in list_async_delegations()
                if d.get("status") in ("running", "stalling", "finalizing")
            ]
        except Exception:
            delegations = []
        if delegations:
            lines.extend(
                [
                    "",
                    t(
                        "gateway.agents.background_delegations",
                        count=len(delegations),
                    ),
                ]
            )
            for d in delegations[:12]:
                goal = " ".join(str(d.get("goal") or "").split())
                if len(goal) > 70:
                    goal = goal[:67] + "..."
                status = d.get("status", "?")
                row = f"- `{d.get('delegation_id', '?')}` · {status}"
                if status == "stalling":
                    quiet = d.get("stalled_after_quiet_seconds")
                    if quiet is not None:
                        row += f" · no progress {quiet:.0f}s"
                elif d.get("seconds_since_progress", 0) >= 60:
                    row += f" · quiet {d['seconds_since_progress']:.0f}s"
                if goal:
                    row += f" · {goal}"
                lines.append(row)
                for i, child in enumerate(d.get("children_activity") or []):
                    if not isinstance(child, dict):
                        continue
                    tool = child.get("current_tool")
                    doing = f"`{tool}`" if tool else "between turns"
                    part = (
                        f"  - child {i + 1}: "
                        f"{child.get('api_calls', '?')} api calls · {doing}"
                    )
                    idle = child.get("seconds_since_activity")
                    if idle is not None:
                        part += f" · active {idle:.0f}s ago"
                    lines.append(part)
            if len(delegations) > 12:
                lines.append(
                    t("gateway.agents.more", count=len(delegations) - 12)
                )

        if (
            not agent_rows
            and not running_processes
            and not background_tasks
            and not delegations
        ):
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

        # No running agent anywhere for this scope. A platform status
        # indicator can still be stuck — e.g. Slack's persistent
        # assistant.threads.setStatus survives a gateway restart or a turn
        # that died without a final send (#32295). Best-effort clear so
        # /stop always dismisses a phantom "is thinking...".
        adapter = getattr(self, "adapters", {}).get(source.platform)
        if adapter and hasattr(adapter, "_stop_typing_with_metadata"):
            try:
                await adapter._stop_typing_with_metadata(
                    source.chat_id,
                    self._thread_metadata_for_source(
                        source, self._reply_anchor_for_event(event)
                    ),
                )
            except Exception:
                logger.debug(
                    "Failed to clear typing on /stop with no active agent",
                    exc_info=True,
                )

        return t("gateway.stop.no_active")

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
            if event.source.delivered_via_upstream_relay is True:
                notify_data["delivered_via_upstream_relay"] = True
                if event.source.user_id:
                    notify_data["user_id"] = event.source.user_id
                if event.source.scope_id:
                    notify_data["scope_id"] = event.source.scope_id
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
        # Native supervisor markers cover direct systemd/launchd starts. The
        # explicit marker covers wrappers such as ``sudo env -i`` that strip
        # those markers before execing the foreground gateway.
        from gateway.restart import (
            is_container_restart_context,
            is_gateway_supervisor_process,
        )

        _under_service = is_gateway_supervisor_process()
        _in_container = is_container_restart_context()
        if _under_service or _in_container:
            self.request_restart(detached=False, via_service=True)
        else:
            self.request_restart(detached=True, via_service=False)
        if active_agents:
            return t("gateway.draining", count=active_agents)
        return EphemeralReply(t("gateway.restart.restarting"))

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

    def _context_breakdown_block(self, agent, source, expanded: bool) -> list[str]:
        """Render the /context per-category block (plain text, no grid).

        Estimated (chars/4) — same engine as the desktop popover and /usage.
        ``expanded`` appends the per-skill / per-toolset listings from the
        prompt-size attribution mechanism. Runs in a thread (sync store reads);
        returns [] and never raises so /context stays robust.
        """
        try:
            from agent.context_breakdown import (
                compute_context_details,
                compute_session_context_breakdown,
                render_context_breakdown_lines,
            )

            history: list[dict] = []
            try:
                entry = self.session_store.get_or_create_session(source)
                history = self.session_store.load_transcript(entry.session_id) or []
            except Exception:
                history = []

            payload = compute_session_context_breakdown(agent, history)
            if not (payload.get("categories") or []):
                return []

            details = None
            if expanded:
                try:
                    details = compute_context_details(agent)
                except Exception:
                    details = {"skills": [], "toolsets": []}

            return render_context_breakdown_lines(payload, details=details, grid=False)
        except Exception:
            return []
