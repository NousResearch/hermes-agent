"""Read-only gateway introspection commands: /status, /context, /usage, /agents, /insights, /topup.
Bound onto ``GatewayRunner`` through ``GatewaySlashCommandsMixin``."""

from __future__ import annotations

import logging
import asyncio
import hashlib
import os
import re
import time
from typing import Any

from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session_transcript import TranscriptReadError

# Log-record parity with gateway/run.py and the origin module.
logger = logging.getLogger("gateway.run")

_LIST_CAP = 12  # /agents shows at most this many rows per section


def _clean_str(value: Any) -> str:
    """Strip and return a non-empty string value, or empty string."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _int_value(value: Any) -> int:
    """Safely coerce to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _n(obj, attr: str):
    return getattr(obj, attr, 0) or 0


def _fmt(n) -> str:
    return f"{n:,}"


def _pct(used, total) -> float:  # clamped occupancy percentage; 0 for an unknown window
    return min(100, used / total * 100) if total else 0


def _clip(text: str, limit: int) -> str:
    return text[: limit - 3] + "..." if len(text) > limit else text


def _transcript_estimate(history) -> tuple[int, int]:
    """``(approx_tokens, message_count)`` over the user/assistant messages of a transcript."""
    from agent.model_metadata import estimate_messages_tokens_rough
    msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
    return estimate_messages_tokens_rough(msgs), len(msgs)


async def _quiet(call, default=None):
    """Await ``call()`` fail-open: any exception (sync or in the awaitable) yields *default*."""
    try:
        return await call()
    except Exception:
        return default


HISTORY_UNREADABLE = ("⚠️ Conversation history is unreadable (state.db). "
                      "This is not a new conversation — earlier messages exist but cannot be loaded.")


def _quiet_sync(call, default=None):
    """Sync twin of ``_quiet``."""
    try:
        return call()
    except Exception:
        return default


def _status_model_route(status_agent, persisted_route: dict, session_row: dict, session_entry):
    """``(model, provider, context_used, context_total)`` for /status.

    Order: live/cached agent route -> persisted dominant route -> SessionDB row -> gateway config
    (only loaded when something is still missing).
    """
    from gateway.run import _AGENT_PENDING_SENTINEL, _load_gateway_config, _resolve_gateway_model
    context_used = context_total = 0
    routes = []
    if status_agent is not None and status_agent is not _AGENT_PENDING_SENTINEL:
        routes.append((_clean_str(getattr(status_agent, "model", "")),
                       _clean_str(getattr(status_agent, "provider", ""))))
        ctx = getattr(status_agent, "context_compressor", None)
        if ctx is not None:
            context_used = _int_value(getattr(ctx, "last_prompt_tokens", 0))
            context_total = _int_value(getattr(ctx, "context_length", 0))
    routes.append((_clean_str(persisted_route.get("model")),
                   _clean_str(persisted_route.get("billing_provider"))))
    row_route = (_clean_str(session_row.get("model")), _clean_str(session_row.get("billing_provider")))
    # First fully-resolved (model AND provider) route wins; the SessionDB row is used even if partial.
    model_name, provider_name = next((r for r in routes if r[0] and r[1]), row_route)
    context_used = context_used or _int_value(getattr(session_entry, "last_prompt_tokens", 0))
    user_config: dict[str, Any] = {}
    if not model_name or not provider_name or not context_total:
        user_config = _quiet_sync(_load_gateway_config, {})
    model_cfg = user_config.get("model", {}) if isinstance(user_config, dict) else {}
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    model_name = model_name or _resolve_gateway_model(user_config)
    provider_name = provider_name or _clean_str(model_cfg.get("provider"))
    configured_context = model_cfg.get("context_length")
    if not context_total and isinstance(configured_context, int) and configured_context > 0:
        context_total = configured_context
    return model_name, provider_name, context_used, context_total


def _context_compressor_lines(agent, ctx, used: int) -> list[str]:
    """/context full view: auto-compression threshold/headroom, compression count + last savings,
    and cumulative throughput (labelled as throughput, NOT context size)."""
    lines: list[str] = []
    threshold = _n(ctx, "threshold_tokens")
    threshold_pct = f"{_n(ctx, 'threshold_percent') * 100:.0f}"
    if threshold > 0:
        if used >= threshold:
            lines.append(t("gateway.context.over_threshold", threshold=_fmt(threshold),
                           threshold_pct=threshold_pct))
        else:
            lines.append(t("gateway.context.threshold", threshold=_fmt(threshold),
                           threshold_pct=threshold_pct, to_go=_fmt(threshold - used)))
    compressions = _n(ctx, "compression_count")
    lines.append(t("gateway.context.compressions", count=compressions))
    savings = getattr(ctx, "_last_compression_savings_pct", None) if compressions else None
    if savings is not None:
        lines.append(t("gateway.context.last_savings", savings=f"{savings:.0f}"))
    lines += [
        "",
        t("gateway.context.totals_header", calls=_n(agent, "session_api_calls")),
        t("gateway.context.totals_line",
          input=_fmt(_n(agent, "session_input_tokens")),
          output=_fmt(_n(agent, "session_output_tokens")),
          reasoning=_fmt(_n(agent, "session_reasoning_tokens"))),
        t("gateway.context.total_billed", total=_fmt(_n(agent, "session_total_tokens"))),
        t("gateway.context.throughput_note"),
    ]
    return lines


def _agents_delegation_lines(d: dict) -> list[str]:
    """/agents rows for one background delegation. Live per-child activity comes from the
    registry's progress sampler: api calls, current tool, seconds since last activity."""
    goal = _clip(" ".join(str(d.get("goal") or "").split()), 70)
    status = d.get("status", "?")
    row = f"- `{d.get('delegation_id', '?')}` · {status}"
    quiet = d.get("stalled_after_quiet_seconds")
    if status == "stalling" and quiet is not None:
        row += f" · no progress {quiet:.0f}s"
    elif status != "stalling" and d.get("seconds_since_progress", 0) >= 60:
        row += f" · quiet {d['seconds_since_progress']:.0f}s"
    if goal:
        row += f" · {goal}"
    lines = [row]
    for i, child in enumerate(d.get("children_activity") or []):
        if not isinstance(child, dict):
            continue
        tool = child.get("current_tool")
        doing = f"`{tool}`" if tool else "between turns"
        part = f"  - child {i + 1}: {child.get('api_calls', '?')} api calls · {doing}"
        idle = child.get("seconds_since_activity")
        lines.append(part + (f" · active {idle:.0f}s ago" if idle is not None else ""))
    return lines


def _usage_agent_stats_lines(agent) -> list[str]:
    """/usage session block for a live agent: rate limits, token breakdown (matches the CLI),
    context window and compression count."""
    lines: list[str] = []
    rl_state = agent.get_rate_limit_state()
    if rl_state and rl_state.has_data:
        from agent.rate_limit_tracker import format_rate_limit_compact
        lines += [t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)), ""]
    lines += [
        t("gateway.usage.header_session"),
        t("gateway.usage.label_model", model=agent.model),
        t("gateway.usage.label_input_tokens", count=_fmt(_n(agent, "session_input_tokens"))),
        t("gateway.usage.label_output_tokens", count=_fmt(_n(agent, "session_output_tokens"))),
        t("gateway.usage.label_total", count=_fmt(agent.session_total_tokens)),
        t("gateway.usage.label_api_calls", count=agent.session_api_calls),
    ]
    ctx = agent.context_compressor
    if ctx.last_prompt_tokens > 0:
        pct = _pct(ctx.last_prompt_tokens, ctx.context_length)
        lines.append(t("gateway.usage.label_context", used=_fmt(ctx.last_prompt_tokens),
                       total=_fmt(ctx.context_length), pct=f"{pct:.0f}"))
    if ctx.compression_count:
        lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))
    return lines


def _capped_rows(items: list, render) -> list[str]:
    """Render up to ``_LIST_CAP`` items via *render* (list of lines each) plus an overflow line."""
    lines: list[str] = []
    for item in items[:_LIST_CAP]:
        lines.extend(render(item))
    if len(items) > _LIST_CAP:
        lines.append(t("gateway.agents.more", count=len(items) - _LIST_CAP))
    return lines


class GatewayStatusCommandsMixin:
    """Read-only gateway introspection commands: /status, /context, /usage, /agents, /insights, /topup."""

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        # Keep the sentinel distinct: a starting/pending run is not a usable agent for
        # model/context display, but it still occupies the session slot.
        agent = self._running_agents.get(session_key)
        is_running = agent is not None and agent is not _AGENT_PENDING_SENTINEL
        # Pending /queue follow-ups (slot + overflow).
        adapter = self.adapters.get(source.platform) if source else None
        queue_depth = self._queue_depth(session_key, adapter=adapter)
        title, session_row, db_total_tokens, persisted_route = await self._status_session_db_facts(
            session_entry.session_id
        )
        # Prefer the live or cached agent (actual runtime route + context compressor); fall back
        # to SessionDB metadata + last_prompt_tokens so /status stays useful between turns.
        status_agent = agent if is_running else self._cached_agent_for(session_key)
        model_name, provider_name, context_used, context_total = _status_model_route(
            status_agent, persisted_route, session_row, session_entry
        )

        stamp = "%Y-%m-%d %H:%M"
        lines = [t("gateway.status.header"), "",
                 t("gateway.status.session_id", session_id=session_entry.session_id)]
        if title:
            lines.append(t("gateway.status.title", title=title))
        lines += [t("gateway.status.created", timestamp=session_entry.created_at.strftime(stamp)),
                  t("gateway.status.last_activity", timestamp=session_entry.updated_at.strftime(stamp))]
        if model_name and provider_name:
            lines.append(t("gateway.status.model_provider", model=model_name, provider=provider_name))
        elif model_name:
            lines.append(t("gateway.status.model", model=model_name))
        if context_total:
            pct = min(100, round((context_used / context_total) * 100))
            lines.append(t("gateway.status.context", used=_fmt(context_used), total=_fmt(context_total),
                           pct=f"{pct}"))
        elif context_used:
            lines.append(t("gateway.status.context_used", used=_fmt(context_used)))
        state = t("gateway.status.state_yes") if is_running else t("gateway.status.state_no")
        lines += [t("gateway.status.tokens", tokens=_fmt(db_total_tokens)),
                  t("gateway.status.agent_running", state=state)]
        if queue_depth:
            lines.append(t("gateway.status.queued", count=queue_depth))
        if source.platform == Platform.MATRIX:
            scope = getattr(self.adapters.get(Platform.MATRIX), "_matrix_session_scope",
                            os.getenv("MATRIX_SESSION_SCOPE", "auto"))
            lines += [
                "",
                t("gateway.status.matrix_scope_header"),
                t("gateway.status.matrix_scope_room", room=source.chat_name or source.chat_id),
                t("gateway.status.matrix_scope_room_id", room_id=source.chat_id),
                t("gateway.status.matrix_scope_thread", thread_id=source.thread_id or "none"),
                t("gateway.status.matrix_scope_mode", scope=scope),
                t("gateway.status.matrix_scope_key",
                  session_key=self._redact_matrix_session_key(session_key)),
            ]
        lines += ["", t("gateway.status.platforms", platforms=', '.join(p.value for p in self.adapters))]
        return "\n".join(lines)

    async def _status_session_db_facts(self, session_id: str):
        """``(title, session_row, db_total_tokens, persisted_route)`` for /status; each fail-open.

        Token totals come from the SQLite session DB, not SessionStore: run_agent.py persists per-turn
        token deltas into sessions_db, never into SessionEntry (its total_tokens is always 0).
        """
        db = self._session_db
        if not db:
            return None, {}, 0, {}
        title = await _quiet(lambda: db.get_session_title(session_id))
        row = await _quiet(lambda: db.get_session(session_id))
        session_row = row if isinstance(row, dict) else {}
        db_total_tokens = sum(
            _int_value(session_row.get(k))
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
        )
        route = await _quiet(lambda: db.get_dominant_session_model_route(session_id))
        return title, session_row, db_total_tokens, route if isinstance(route, dict) else {}

    @staticmethod
    def _redact_matrix_session_key(session_key: str) -> str:
        """Return a stable Matrix session-key fingerprint for shared room status."""
        digest = hashlib.sha256(str(session_key or "").encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    async def _handle_context_command(self, event: MessageEvent) -> str:
        """Handle /context — the deep context-window view (/status has the one-line summary).

        Gauge, auto-compression threshold/headroom, compression count + last savings, and cumulative
        throughput (labelled as throughput, NOT context size). Resolution: running agent -> cached
        agent -> SessionStore/SessionDB metadata -> transcript estimate. ``all`` adds listings.
        """
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        expanded = event.get_command_args().strip().lower() in {"all", "full", "details"}
        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(self._session_key_for_source(source)) or None
        ctx = getattr(agent, "context_compressor", None) if agent else None
        used, context_length, model_name = await self._resolve_context_figures(
            agent, ctx, session_entry, source
        )
        # Gauge path: real current-context figure
        if used > 0 and context_length > 0:
            pct = _pct(used, context_length)
            filled = int(round(pct / 100 * 24))
            lines = [
                t("gateway.context.header"), "",
                t("gateway.context.model", model=model_name or "?"),
                t("gateway.context.window", total=_fmt(context_length)),
                t("gateway.context.in_use", used=_fmt(used), total=_fmt(context_length), pct=f"{pct:.0f}"),
                t("gateway.context.bar", bar="█" * max(0, filled) + "░" * max(0, 24 - filled)),
                t("gateway.context.headroom", headroom=_fmt(max(0, context_length - used))),
                "",
            ]
            # Full view — compression / throughput need the live agent.
            lines += _context_compressor_lines(agent, ctx, used) if ctx is not None else [
                t("gateway.context.detail_after_first")]
            # Per-category estimated breakdown (+ optional expanded listings). Same chars/4 engine
            # the desktop popover and /usage use; plain text (monospace isn't guaranteed on
            # messaging platforms). Fail-open: rendering errors never break /context.
            breakdown = await asyncio.to_thread(self._context_breakdown_block, agent, source, expanded) if agent else []
            return "\n".join(lines + ([""] + breakdown if breakdown else []))
        # Last resort: rough estimate from transcript
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        if not history:
            return t("gateway.context.no_data")
        approx, count = _transcript_estimate(history)
        return "\n".join([
            t("gateway.context.header"), "",
            t("gateway.context.estimated", count=_fmt(approx), messages=count),
            t("gateway.context.detail_after_first"),
        ])

    async def _resolve_context_figures(self, agent, ctx, session_entry, source):
        """``(used, context_length, model_name)`` for /context: used = compressor -> SessionStore;
        model = agent -> SessionDB row; window = compressor -> gateway model route -> model metadata."""
        used = _n(ctx, "last_prompt_tokens") or _int_value(getattr(session_entry, "last_prompt_tokens", 0))
        context_length = _n(ctx, "context_length")
        model_name = _clean_str(getattr(agent, "model", "")) if agent is not None else ""
        if not model_name and self._session_db:
            row = await _quiet(lambda: self._session_db.get_session(session_entry.session_id))
            model_name = _clean_str(row.get("model", "")) if isinstance(row, dict) else ""
        if not context_length:
            from gateway.run import _profile_runtime_scope, _resolve_gateway_model_context

            def _resolve_nonresident_context():
                if getattr(getattr(self, "config", None), "multiplex_profiles", False):
                    with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                        return _resolve_gateway_model_context(model_name or None)
                return _resolve_gateway_model_context(model_name or None)
            resolved = await _quiet(lambda: asyncio.to_thread(_resolve_nonresident_context))
            if resolved is not None:
                model_name = model_name or resolved.model
                context_length = _int_value(resolved.context_length)
        if not context_length and model_name:
            from agent.model_metadata import get_model_context_length
            context_length = _int_value(
                await _quiet(lambda: asyncio.to_thread(get_model_context_length, model_name))
            )
        return used, context_length, model_name

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command - list active agents and running tasks."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        from tools.process_registry import format_uptime_short, process_registry
        now = time.time()
        current_session_key = self._session_key_for_source(event.source)
        running_started: dict = getattr(self, "_running_agents_ts", {}) or {}
        agent_rows: list[dict] = []
        for session_key, agent in (getattr(self, "_running_agents", {}) or {}).items():
            pending = agent is _AGENT_PENDING_SENTINEL
            agent_rows.append({
                "session_key": session_key,
                "elapsed": max(0, int(now - float(running_started.get(session_key, now)))),
                "state": t("gateway.agents.state_starting") if pending else t("gateway.agents.state_running"),
                "session_id": "" if pending else str(getattr(agent, "session_id", "") or ""),
                "model": "" if pending else str(getattr(agent, "model", "") or ""),
            })
        agent_rows.sort(key=lambda row: row["elapsed"], reverse=True)
        procs = _quiet_sync(process_registry.list_sessions, [])
        running_processes = [p for p in procs if p.get("status") == "running"]
        background_tasks = [task for task in (getattr(self, "_background_tasks", set()) or set())
                            if hasattr(task, "done") and not task.done()]

        # Background (async) delegations — delegate_task(background=true).
        # Live per-child activity comes from the registry's progress sampler (#51690): api calls, current
        # tool, seconds since last activity.
        from tools.async_delegation import list_async_delegations
        delegations = [d for d in _quiet_sync(list_async_delegations, [])
                       if d.get("status") in ("running", "stalling", "finalizing")]

        def _agent_row(idx_row):
            idx, row = idx_row
            current = t("gateway.agents.this_chat") if row["session_key"] == current_session_key else ""
            sid = f" · `{row['session_id']}`" if row["session_id"] else ""
            model = f" · `{row['model']}`" if row["model"] else ""
            return [f"{idx}. `{row['session_key']}` · {row['state']} · "
                    f"{format_uptime_short(row['elapsed'])}{sid}{model}{current}"]

        def _proc_row(proc):
            cmd = _clip(" ".join(str(proc.get("command", "")).split()), 90)
            return [f"- `{proc.get('session_id', '?')}` · "
                    f"{format_uptime_short(int(proc.get('uptime_seconds', 0)))} · `{cmd}`"]

        lines = [t("gateway.agents.header"), "", t("gateway.agents.active_agents", count=len(agent_rows))]
        lines += _capped_rows(list(enumerate(agent_rows, 1)), _agent_row)
        lines += ["", t("gateway.agents.running_processes", count=len(running_processes))]
        lines += _capped_rows(running_processes, _proc_row)
        lines += ["", t("gateway.agents.async_jobs", count=len(background_tasks))]
        if delegations:
            lines += ["", t("gateway.agents.background_delegations", count=len(delegations))]
            lines += _capped_rows(delegations, _agents_delegation_lines)
        if not (agent_rows or running_processes or background_tasks or delegations):
            lines += ["", t("gateway.agents.none")]
        return "\n".join(lines)

    async def _handle_topup_command(self, event: MessageEvent) -> str:
        """Handle /topup -- show the Nous balance and hand off to the portal. Does NOT charge, confirm,
        or track payment (that happens in the browser; the next /topup shows the new balance)."""
        from agent.account_usage import build_credits_view
        view = await _quiet(lambda: asyncio.to_thread(build_credits_view, markdown=True))
        if view is None or not view.logged_in:
            return t("gateway.credits.not_logged_in")
        # Drop the helper's 📈 header; we print our own.
        lines = ["💳 **Nous balance**"] + [ln for ln in view.balance_lines if not ln.lstrip().startswith("📈")]
        if view.identity_line:
            lines += ["", view.identity_line]
        if view.topup_url:
            lines += ["", f"Manage billing on the portal: {view.topup_url}",
                      "Top up and manage billing in the browser — your balance updates here after."]
        return "\n".join(lines)

    def _context_breakdown_block(self, agent, source, expanded: bool) -> list[str]:
        """/context per-category block (plain text, chars/4 estimate, same engine as /usage).
        Runs in a thread; returns [] and never raises."""
        try:
            from agent.context_breakdown import compute_context_details, render_context_breakdown_lines
            try:
                payload = self._session_context_breakdown(agent, source)
            except TranscriptReadError:
                return [HISTORY_UNREADABLE]  # a read failure is not an empty transcript
            if not (payload.get("categories") or []):
                return []
            details = _quiet_sync(lambda: compute_context_details(agent), {"skills": [], "toolsets": []}) if expanded else None
            return render_context_breakdown_lines(payload, details=details, grid=False)
        except Exception:
            return []

    def _session_context_breakdown(self, agent, source) -> dict:
        """Per-category context estimate (chars/4) for *agent* over the session transcript (sync).
        Raises ``TranscriptReadError`` (unreadable rows must not pass as an empty transcript)."""
        from agent.context_breakdown import compute_session_context_breakdown
        store = self.session_store
        try:
            history = store.load_transcript(store.get_or_create_session(source).session_id) or []
        except TranscriptReadError:
            raise
        except Exception:
            history = []
        return compute_session_context_breakdown(agent, history)

    def _context_breakdown_lines(self, agent, source) -> list[str]:
        """/usage per-category context breakdown (chars/4 estimate). Returns [] and never raises."""
        try:
            try:
                payload = self._session_context_breakdown(agent, source)
            except TranscriptReadError:
                return [HISTORY_UNREADABLE]
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
                if label.endswith(f"breakdown_cat_{cat_id}"):  # missing key: t() echoes it back
                    label = str(cat.get("label") or cat_id)
                pct = round(tokens / total * 100) if total else 0
                out.append(t("gateway.usage.breakdown_line", label=label, count=_fmt(tokens), pct=pct))
            return out if len(out) > 1 else []
        except Exception:
            return []

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage -- token usage for the current session (live or cached agent) plus
        account/credit blocks; ``/usage reset [--force]`` redeems a banked Codex reset credit."""
        source = event.source
        session_key = self._session_key_for_source(source)
        raw_args = event.get_command_args().strip()
        args = [a.lower() for a in raw_args.split()] if raw_args else []
        wants_reset = bool(args) and args[0] == "reset"
        if args and not wants_reset:
            return t("gateway.usage.unknown_subcommand", args=raw_args)

        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(session_key)

        # Provider/base_url/api_key for the account-usage fetch: live agent first, else persisted
        # billing data on the SessionDB row so `/usage` still returns account info between turns.
        provider, base_url, api_key = (
            getattr(agent, k, None) if agent else None for k in ("provider", "base_url", "api_key")
        )
        if not provider and getattr(self, "_session_db", None) is not None:
            provider, base_url = await self._persisted_billing_route(source)
        if wants_reset:
            if str(provider or "").strip().lower() != "openai-codex":
                return t("gateway.usage.reset_wrong_provider")
            from agent.account_usage import redeem_codex_reset_credit
            result = await asyncio.to_thread(
                redeem_codex_reset_credit, base_url=base_url, api_key=api_key, force="--force" in args[1:],
            )
            return result.message

        # Account usage off the event loop so slow provider APIs don't block the gateway;
        # failures are non-fatal (account_lines stays []).
        account_snapshot = provider and await _quiet(
            lambda: asyncio.to_thread(fetch_account_usage, provider, base_url=base_url, api_key=api_key)
        )
        account_lines = (
            render_account_usage_lines(account_snapshot, markdown=True) if account_snapshot else []
        )

        # Nous credits + monthly-grant gauge (shared with CLI/TUI). Gates on "a Nous account is
        # logged in" — NOT the inference provider — so a Nous user inferring elsewhere still sees
        # a balance. Fail-open: never break /usage.
        from agent.account_usage import nous_credits_lines
        credits_lines = await _quiet(lambda: asyncio.to_thread(nous_credits_lines, markdown=True), [])

        def _with_account_blocks(lines: list[str]) -> str:
            # Each block is preceded by a blank divider only when something precedes it.
            for block in (account_lines, credits_lines):
                if block:
                    if lines:
                        lines.append("")
                    lines.extend(block)
            return "\n".join(lines)
        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = _usage_agent_stats_lines(agent)
            # Per-category breakdown (chars/4 estimate, same engine as the desktop popover): prompt
            # / tools / skills / memory off the live agent, conversation from the transcript.
            breakdown_lines = await asyncio.to_thread(self._context_breakdown_lines, agent, source)
            if breakdown_lines:
                lines += [""] + breakdown_lines
            return _with_account_blocks(lines)

        # No agent at all -- rough count from session history
        session_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        if history:
            approx, count = _transcript_estimate(history)
            return _with_account_blocks([
                t("gateway.usage.header_session_info"),
                t("gateway.usage.label_messages", count=count),
                t("gateway.usage.label_estimated_context", count=_fmt(approx)),
                t("gateway.usage.detailed_after_first"),
            ])
        if account_lines or credits_lines:
            return _with_account_blocks([])
        return t("gateway.usage.no_data")

    async def _persisted_billing_route(self, source):
        """``(provider, base_url)`` from the SessionDB row / dominant route when no agent is resident."""
        async def _rows():
            entry = await self.async_session_store.get_or_create_session(source)
            persisted = await self._session_db.get_session(entry.session_id) or {}
            route = await self._session_db.get_dominant_session_model_route(entry.session_id)
            return persisted, route if isinstance(route, dict) else {}
        persisted, dominant = await _quiet(_rows, ({}, {}))
        row = dominant if dominant.get("billing_provider") else persisted
        return row.get("billing_provider"), row.get("billing_base_url")

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights [N | --days N] [--source S] -- usage insights and analytics."""
        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', event.get_command_args().strip())
        days, source = 30, None
        parts = args.split()
        i = 0
        while i < len(parts):
            flag, value = parts[i], parts[i + 1] if i + 1 < len(parts) else None
            if flag == "--days" and value is not None:
                try:
                    days = int(value)
                except ValueError:
                    return t("gateway.insights.invalid_days", value=value)
                i += 2
            elif flag == "--source" and value is not None:
                source, i = value, i + 2
            else:
                days = int(flag) if flag.isdigit() else days
                i += 1
        try:
            from hermes_state_registry import acquire
            from agent.insights import InsightsEngine

            def _run_insights():
                db = acquire()
                try:
                    engine = InsightsEngine(db)
                    return engine.format_gateway(engine.generate(days=days, source=source))
                finally:
                    from hermes_state_registry import release_or_close
                    release_or_close(db)

            # Not a bare hop: ``SessionDB()`` resolves ``get_hermes_home()`` at call time, a
            # contextvar set by ``_profile_runtime_scope``; a default-executor hop starts with an
            # EMPTY context and would read the DEFAULT profile's state.db.
            return await self._run_in_executor_with_context(_run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)
