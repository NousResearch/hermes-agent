"""Compact /status card formatting for gateway sessions."""

from __future__ import annotations

from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_count(value: Any) -> str:
    n = _as_int(value, 0)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{sign}{val:.1f}m"
    if n >= 1_000:
        val = n / 1_000
        return f"{sign}{val:.1f}k" if n % 1_000 else f"{sign}{int(val)}k"
    return f"{sign}{n}"


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "unknown"
    total = _as_int(seconds, -1)
    if total < 0:
        return "unknown"
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{secs}s")
    return " ".join(parts[:2])


def _format_cost(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _format_context(current: Any, limit: Any) -> str:
    cur = _as_int(current, 0)
    lim = _as_int(limit, 0)
    if lim > 0:
        pct = int(round((cur / lim) * 100))
        return f"{_format_count(cur)}/{_format_count(lim)} ({pct}%)"
    # No known limit — still show current if available
    if cur > 0:
        return f"{_format_count(cur)} used"
    return "unknown"


def _format_cache_hit(snapshot: dict[str, Any]) -> str:
    if snapshot.get("cache_hit_pct") is not None:
        return f"{_as_int(snapshot.get('cache_hit_pct'), 0)}%"
    read = _as_int(snapshot.get("cache_read_tokens"), 0)
    write = _as_int(snapshot.get("cache_write_tokens"), 0)
    fresh_input = _as_int(snapshot.get("input_tokens"), 0)
    denom = read + write + fresh_input
    pct = int(round((read / denom) * 100)) if denom > 0 else 0
    return f"{pct}%"


def format_hermes_status_card(snapshot: dict[str, Any]) -> str:
    """Format a compact Telegram-friendly Hermes status card."""
    version = str(snapshot.get("version") or "unknown")
    commit = snapshot.get("commit")
    header = f"🪽 Hermes {version}"
    if commit:
        header += f" ({str(commit)[:7]})"

    fallbacks = snapshot.get("fallbacks") or []
    fallbacks_text = ", ".join(str(item) for item in fallbacks if item) or "none"

    compactions = snapshot.get("compactions")
    compactions_text = str(compactions) if compactions is not None else "unknown"

    return "\n".join(
        [
            header,
            (
                "⏱️ Uptime: "
                f"gateway {_format_duration(snapshot.get('gateway_uptime_seconds'))} · "
                f"system {_format_duration(snapshot.get('system_uptime_seconds'))}"
            ),
            f"🧠 Model: {snapshot.get('model') or 'unknown'}",
            f"🔄 Fallbacks: {fallbacks_text}",
            (
                "🧮 Tokens: "
                f"{_format_count(snapshot.get('input_tokens'))} in / "
                f"{_format_count(snapshot.get('output_tokens'))} out · "
                f"💵 Cost: {_format_cost(snapshot.get('estimated_cost_usd'))}"
            ),
            (
                "🗄️ Cache: "
                f"{_format_cache_hit(snapshot)} hit · "
                f"{_format_count(snapshot.get('cache_read_tokens'))} read, "
                f"{_format_count(snapshot.get('cache_write_tokens'))} write"
            ),
            (
                "📚 Context: "
                f"{_format_context(snapshot.get('context_tokens'), snapshot.get('context_limit'))} · "
                f"🧹 Compactions: {compactions_text}"
            ),
            f"🧵 Session: {snapshot.get('session_id') or 'unknown'}",
            f"📌 Tasks: {_format_count(snapshot.get('active_tasks'))} active",
            (
                "🪢 Queue: "
                f"{snapshot.get('queue_mode') or 'queue'} "
                f"(depth {_format_count(snapshot.get('queue_depth'))})"
            ),
        ]
    )
