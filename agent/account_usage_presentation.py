"""Allowlisted account-usage projection for clients without provider credentials."""
from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any

from agent.account_usage import AccountUsageRow, AccountUsageSnapshot, AccountUsageWindow, WINDOW_LABEL_KEYS

SUPPORTED_PROVIDERS = frozenset({"openrouter", "openai-codex", "anthropic"})
_MAX_NUMBER = 1_000_000_000_000_000
_WINDOW_LABELS = {key: label for label, key in WINDOW_LABEL_KEYS.items()}
_ROW_FIELDS = {
    "credits_balance": (("value",), ("value",)),
    "banked_resets": (("count",), ("count",)),
    "credits_unlimited": ((), ()),
    "extra_usage": (("used", "limit"), ("used", "limit")),
    "api_key_usage": (("total",), ("total", "daily", "weekly", "monthly")),
}
_RESET_INTERVALS = frozenset({"daily", "weekly", "monthly"})


def _number(value: Any) -> float | None:
    """Bound magnitudes before conversion, excluding bool, NaN, infinity and enormous integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if -_MAX_NUMBER <= value <= _MAX_NUMBER else None


def _amount(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()


def _window(window: AccountUsageWindow) -> dict[str, Any] | None:
    key = window.label_key or WINDOW_LABEL_KEYS.get(window.label)
    if key not in _WINDOW_LABELS or window.scope not in ("account", "model"):
        return None
    number = _number(window.used_percent)
    used = max(0.0, min(100.0, number)) if number is not None else None
    limit, remaining = _amount(window.limit), _amount(window.limit_remaining)
    if limit is not None and remaining is not None:
        remaining = min(limit, remaining)
    return {
        "label": _WINDOW_LABELS[key], "label_key": key, "scope": window.scope,
        "used_percent": used, "remaining_percent": 100.0 - used if used is not None else None,
        "reset_at": _timestamp(window.reset_at), "limit": limit, "limit_remaining": remaining,
        "reset_interval": window.reset_interval if window.reset_interval in _RESET_INTERVALS else None,
    }


def _row(row: AccountUsageRow) -> dict[str, Any] | None:
    if row.key not in _ROW_FIELDS:
        return None
    required, fields = _ROW_FIELDS[row.key]
    args: dict[str, float | int | str] = {}
    for name in fields:
        value = _amount(row.args.get(name))
        if value is not None:
            args[name] = value
    if any(name not in args for name in required):
        return None
    if row.key == "banked_resets":
        if not float(args["count"]).is_integer():
            return None
        args["count"] = int(args["count"])
    currency = row.args.get("currency")
    if row.key in ("credits_balance", "extra_usage") and isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency):
        args["currency"] = currency
    return {"key": row.key, "args": args}


def serialize_account_usage_snapshot(snapshot: AccountUsageSnapshot) -> dict[str, Any] | None:
    """Project known numbers and fixed labels, never raw payloads, identities or free-form details."""
    if snapshot.provider not in SUPPORTED_PROVIDERS or snapshot.unavailable_reason:
        return None
    fetched_at = _timestamp(snapshot.fetched_at)
    if fetched_at is None:
        return None
    windows = [item for window in snapshot.windows if (item := _window(window)) is not None]
    rows = [item for row in snapshot.rows if (item := _row(row)) is not None]
    credits = _amount(snapshot.credits_balance)
    if not windows and not rows and credits is None:
        return None
    plan = snapshot.plan
    if not isinstance(plan, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,63}", plan):
        plan = None
    return {
        "provider": snapshot.provider, "fetched_at": fetched_at, "plan": plan,
        "windows": windows, "rows": rows, "credits_balance": credits,
        "usage_window": snapshot.usage_window is True,
    }
