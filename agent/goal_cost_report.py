"""
Per-Goal Cost and Execution Report for Hermes Agent.

Generates a compact cost/usage summary at the end of a cron job or agent goal,
so users see exactly what a task consumed (tokens, cost, tool calls, duration).

Designed to be called from ``cron/scheduler.py`` after ``run_job()`` completes
and before ``_deliver_result()`` sends the output to the user.

Usage:
    from agent.goal_cost_report import format_cost_summary
    summary = format_cost_summary(agent, duration_seconds=42.5)
    # → "📊 Cost: $0.18 · 42,381 tokens · 12 tool calls · 3m 41s"
"""

from __future__ import annotations

import time
from typing import Any, Optional


# ── Cost summary for delivery ─────────────────────────────────────────────


def format_cost_summary(
    agent: Any,
    *,
    duration_seconds: Optional[float] = None,
    tool_call_count: int = 0,
) -> str:
    """Return a one-line cost summary string for the agent's session.

    Reads token/cost info that ``turn_finalizer.py`` already accumulated
    on the agent object.  Returns an empty string when there is no usage
    data (e.g. a no_agent script job or a fresh agent that never called a
    model).

    The summary is designed to be appended to a cron job's delivered message.
    """
    # Collect usage data — these are set by turn_finalizer during run_conversation
    input_tokens = getattr(agent, "session_input_tokens", 0) or 0
    output_tokens = getattr(agent, "session_output_tokens", 0) or 0
    total_tokens = getattr(agent, "session_total_tokens", 0) or 0
    cost_usd = getattr(agent, "session_estimated_cost_usd", None)
    cost_status = getattr(agent, "session_cost_status", "")
    cost_source = getattr(agent, "session_cost_source", "")
    model = getattr(agent, "model", "") or ""
    provider = getattr(agent, "provider", "") or ""

    # Fallback: if total_tokens wasn't kept but we have in/out, compute it
    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens

    # Nothing to report — agent never called a model
    if total_tokens == 0 and cost_usd is None:
        return ""

    # Build the summary
    parts: list[str] = []

    # Cost
    if cost_usd is not None:
        cost_float = float(cost_usd)
        cost_label = _format_cost(cost_float)
        status_tag = ""
        if cost_status and cost_status != "actual":
            status_tag = f" ({cost_status})"
        parts.append(f"💰 {cost_label}{status_tag}")
    elif total_tokens > 0:
        # Can't estimate cost — unknown pricing
        parts.append(f"💰 unknown (no pricing data)")

    # Tokens
    if total_tokens > 0:
        token_str = _format_number(total_tokens)
        detail = ""
        if input_tokens > 0 or output_tokens > 0:
            detail = f"  ({_format_number(input_tokens)} in / {_format_number(output_tokens)} out)"
        parts.append(f"📝 {token_str} tokens{detail}")

    # Model
    if model:
        model_short = model.split("/")[-1] if "/" in model else model
        provider_short = provider.split("/")[-1] if "/" in provider else provider
        via = f" via {provider_short}" if provider_short and provider_short != model_short else ""
        parts.append(f"🤖 {model_short}{via}")

    # Tool calls
    if tool_call_count > 0:
        parts.append(f"🔧 {tool_call_count} tool calls")

    # Duration
    if duration_seconds is not None and duration_seconds > 0:
        parts.append(f"⏱ {_format_duration(duration_seconds)}")

    return "  ·  ".join(parts)


# ── Duration tracking helper ────────────────────────────────────────────


class GoalDuration:
    """Simple context-manager-style duration tracker for agent goals.

    Usage::

        timer = GoalDuration()
        timer.start()
        # ... run the goal ...
        elapsed = timer.stop()  # float seconds
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start


# ── Internals ───────────────────────────────────────────────────────────


def _format_cost(amount_usd: float) -> str:
    """Format a USD cost value compactly."""
    if amount_usd < 0.001:
        return f"${amount_usd:.6f}"
    if amount_usd < 0.01:
        return f"${amount_usd:.4f}"
    if amount_usd < 1.0:
        return f"${amount_usd:.3f}"
    return f"${amount_usd:.2f}"


def _format_number(n: int) -> str:
    """Format an integer with comma separators."""
    s = str(n)
    groups = []
    while s:
        groups.append(s[-3:])
        s = s[:-3]
    return ",".join(reversed(groups))


def _format_duration(seconds: float) -> str:
    """Format a duration compactly."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs}s"
