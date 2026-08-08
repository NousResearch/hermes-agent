"""Shared formatting helpers for /status snapshot displays.

Used by both the CLI (cli.py) and gateway (gateway/slash_commands.py) to render
consistent status output without duplicating logic.
"""

from datetime import datetime
from typing import Any, Optional


def format_status_relative_time(ts: Optional[datetime]) -> str:
    """Return a compact relative timestamp for status displays."""
    if not ts:
        return "unknown"
    delta = datetime.now() - ts
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def format_status_cost(amount: Optional[float], status: str) -> Optional[str]:
    """Return a human-readable cost label for status output."""
    normalized = (status or "").strip().lower()
    if normalized == "included":
        return "included"
    if amount is None and normalized in ("", "unknown", "none"):
        return None
    if amount is None:
        return normalized
    label = f"${amount:,.4f}"
    if normalized in ("", "actual"):
        return label
    if normalized == "estimated":
        return f"{label} est."
    return f"{label} {normalized}"


def format_reasoning_effort_label(config: Optional[dict]) -> str:
    """Return the effective reasoning effort label."""
    if config is None:
        return "medium"
    if config.get("enabled") is False:
        return "none"
    return str(config.get("effort") or "medium")


def format_api_mode_label(api_mode: Optional[str]) -> Optional[str]:
    """Convert internal API mode names into compact user-facing labels."""
    if not api_mode:
        return None
    labels = {
        "chat_completions": "Chat Completions",
        "codex_responses": "Responses",
        "anthropic_messages": "Anthropic Messages",
    }
    return labels.get(api_mode, str(api_mode).replace("_", " ").title())


def safe_status_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion for status values."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_status_float(value: Any) -> Optional[float]:
    """Best-effort float coercion for status values."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
