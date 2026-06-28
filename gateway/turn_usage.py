"""Gateway-facing turn usage telemetry.

Small, dependency-free helpers for making long messaging turns visible without
changing the agent prompt or tool schema.  All values are best-effort: missing
provider usage should degrade to elapsed/API-call telemetry, never break a turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TurnUsage:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    cost_status: str | None = None
    elapsed_seconds: float = 0.0
    model: str | None = None
    provider: str | None = None
    context_tokens: int = 0
    context_length: int = 0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def agent_usage_snapshot(agent: Any) -> dict[str, Any]:
    """Return cumulative usage counters from an AIAgent-like object."""
    ctx = getattr(agent, "context_compressor", None)
    return {
        "input_tokens": _as_int(getattr(agent, "session_input_tokens", 0)),
        "output_tokens": _as_int(getattr(agent, "session_output_tokens", 0)),
        "cache_read_tokens": _as_int(getattr(agent, "session_cache_read_tokens", 0)),
        "cache_write_tokens": _as_int(getattr(agent, "session_cache_write_tokens", 0)),
        "prompt_tokens": _as_int(getattr(agent, "session_prompt_tokens", 0)),
        "completion_tokens": _as_int(getattr(agent, "session_completion_tokens", 0)),
        "total_tokens": _as_int(getattr(agent, "session_total_tokens", 0)),
        "estimated_cost_usd": _as_float(getattr(agent, "session_estimated_cost_usd", None)),
        "cost_status": getattr(agent, "session_cost_status", None),
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "context_tokens": _as_int(getattr(ctx, "last_prompt_tokens", 0)) if ctx else 0,
        "context_length": _as_int(getattr(ctx, "context_length", 0)) if ctx else 0,
    }


def usage_delta(after: Mapping[str, Any] | None, before: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute a non-negative usage delta between cumulative snapshots."""
    after = after or {}
    before = before or {}
    out: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        out[key] = max(0, _as_int(after.get(key)) - _as_int(before.get(key)))

    after_cost = _as_float(after.get("estimated_cost_usd"))
    before_cost = _as_float(before.get("estimated_cost_usd"))
    if after_cost is not None and before_cost is not None:
        out["estimated_cost_usd"] = max(0.0, after_cost - before_cost)
    elif after_cost is not None:
        out["estimated_cost_usd"] = after_cost
    else:
        out["estimated_cost_usd"] = None

    for key in ("cost_status", "model", "provider", "context_tokens", "context_length"):
        out[key] = after.get(key)
    return out


def turn_usage_from_result(result: Mapping[str, Any] | None, *, elapsed_seconds: float = 0.0) -> TurnUsage:
    """Build a TurnUsage from a gateway result dict."""
    result_map: Mapping[str, Any] = result or {}
    raw_value = result_map.get("turn_usage")
    raw: Mapping[str, Any] = raw_value if isinstance(raw_value, Mapping) else result_map
    total = _as_int(raw.get("total_tokens"))
    if total <= 0:
        total = (
            _as_int(raw.get("input_tokens"))
            + _as_int(raw.get("output_tokens"))
            + _as_int(raw.get("cache_read_tokens"))
            + _as_int(raw.get("cache_write_tokens"))
        )
    return TurnUsage(
        api_calls=_as_int(result_map.get("api_calls") or raw.get("api_calls")),
        input_tokens=_as_int(raw.get("input_tokens")),
        output_tokens=_as_int(raw.get("output_tokens")),
        cache_read_tokens=_as_int(raw.get("cache_read_tokens")),
        cache_write_tokens=_as_int(raw.get("cache_write_tokens")),
        total_tokens=total,
        estimated_cost_usd=_as_float(raw.get("estimated_cost_usd")),
        cost_status=str(raw.get("cost_status") or "") or None,
        elapsed_seconds=max(0.0, float(elapsed_seconds or raw.get("elapsed_seconds") or 0.0)),
        model=str(raw.get("model") or result_map.get("model") or "") or None,
        provider=str(raw.get("provider") or result_map.get("provider") or "") or None,
        context_tokens=_as_int(raw.get("context_tokens") or result_map.get("last_prompt_tokens")),
        context_length=_as_int(raw.get("context_length") or result_map.get("context_length")),
    )


def format_compact_usage(usage: TurnUsage, *, prefix: str = "usage") -> str:
    """Render one compact, James-facing usage line."""
    parts = [f"{prefix}: {usage.api_calls} calls"]
    if usage.input_tokens or usage.output_tokens or usage.cache_read_tokens or usage.cache_write_tokens:
        parts.append(f"{usage.input_tokens:,} in")
        if usage.cache_read_tokens:
            parts.append(f"{usage.cache_read_tokens:,} cache-r")
        if usage.cache_write_tokens:
            parts.append(f"{usage.cache_write_tokens:,} cache-w")
        parts.append(f"{usage.output_tokens:,} out")
    if usage.estimated_cost_usd is not None and usage.cost_status != "included":
        marker = "~" if usage.cost_status == "estimated" else ""
        parts.append(f"{marker}${usage.estimated_cost_usd:.4f}")
    elif usage.cost_status == "included":
        parts.append("included")
    if usage.elapsed_seconds:
        if usage.elapsed_seconds >= 60:
            parts.append(f"{usage.elapsed_seconds / 60:.1f}m")
        else:
            parts.append(f"{usage.elapsed_seconds:.0f}s")
    return " · ".join(parts)


def should_show_usage_receipt(
    usage: TurnUsage,
    *,
    min_api_calls: int = 2,
    min_tokens: int = 25_000,
    min_seconds: float = 30.0,
) -> bool:
    return (
        usage.api_calls >= max(1, int(min_api_calls))
        or usage.total_tokens >= max(0, int(min_tokens))
        or usage.elapsed_seconds >= max(0.0, float(min_seconds))
    )


def budget_status(
    usage: TurnUsage,
    *,
    warn_api_calls: int = 8,
    warn_tokens: int = 100_000,
    hard_api_calls: int = 30,
    hard_tokens: int = 350_000,
) -> str | None:
    """Return None, 'warn', or 'hard' for a Telegram turn budget."""
    if hard_api_calls > 0 and usage.api_calls >= hard_api_calls:
        return "hard"
    if hard_tokens > 0 and usage.total_tokens >= hard_tokens:
        return "hard"
    if warn_api_calls > 0 and usage.api_calls >= warn_api_calls:
        return "warn"
    if warn_tokens > 0 and usage.total_tokens >= warn_tokens:
        return "warn"
    return None
