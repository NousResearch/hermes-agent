"""Cache-hit and savings breakdown for ``hermes sessions cost``.

Hermes stores each session's aggregate ``estimated_cost_usd`` but not the per-token price split,
so the "cost if the prompt cache had never hit" figure is an estimate: the recorded cost is
attributed to the input side in proportion to token counts, and the cache-read share is
re-priced as if billed at the cold input rate::

    savings = cost * (input_side / total_tokens) * (cache_read / input_side) * (1 / ratio - 1)

``ratio`` is ``cost.cache_hit_ratio`` (cache-read price / cold input price, default 0.10).
Unpriced usage is persisted as ``estimated_cost_usd = 0`` with ``cost_status = 'unknown'``,
so ``cost_status`` decides whether a zero is a real price.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_CACHE_HIT_RATIO = 0.10
# Below this hit rate the prompt prefix was probably invalidated (toolset swap, compression...).
CACHE_HIT_WARN_THRESHOLD_PCT = 70.0


def cache_hit_ratio_from_config(config: Optional[dict[str, Any]]) -> float:
    """``cost.cache_hit_ratio`` when it is a number in (0, 1], else the default."""
    cost_cfg = config.get("cost") if isinstance(config, dict) else None
    raw = cost_cfg.get("cache_hit_ratio") if isinstance(cost_cfg, dict) else None
    if raw is None:
        return DEFAULT_CACHE_HIT_RATIO
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CACHE_HIT_RATIO
    return parsed if 0.0 < parsed <= 1.0 else DEFAULT_CACHE_HIT_RATIO


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _recorded_cost(row: dict[str, Any], has_usage: bool) -> Optional[float]:
    """The stored cost, or None when the zero is the writer's COALESCE of a missing price
    (usage recorded, but ``cost_status`` is ``unknown`` or absent)."""
    status = (row.get("cost_status") or "").strip().lower()
    raw = row.get("estimated_cost_usd")
    if raw is None:
        return None
    try:
        cost = float(raw)
    except (TypeError, ValueError):
        return None
    if cost == 0.0 and has_usage and status not in ("included", "actual", "estimated"):
        return None
    return cost


def session_cost_breakdown(
    row: dict[str, Any], *, cache_hit_ratio: float = DEFAULT_CACHE_HIT_RATIO,
) -> dict[str, Any]:
    """Token totals, hit %, recorded cost, 0%-cache counterfactual and savings for one
    ``sessions`` row. Cost-derived fields are None when the session was never priced."""
    inp, cr, cw, out = (_int(row.get(k)) for k in
                        ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"))
    input_side = inp + cr + cw
    total = input_side + out
    hit_pct = cr / input_side * 100.0 if input_side else None
    cost = _recorded_cost(row, total > 0)
    ratio = cache_hit_ratio if 0.0 < cache_hit_ratio <= 1.0 else DEFAULT_CACHE_HIT_RATIO
    savings = counterfactual = None
    if cost is not None:
        savings = cost * (input_side / total) * (cr / input_side) * (1.0 / ratio - 1.0) if cr else 0.0
        counterfactual = cost + savings
    return {
        "input_tokens": inp, "cache_read_tokens": cr, "cache_write_tokens": cw, "output_tokens": out,
        "cache_hit_pct": hit_pct,
        "cost_status": row.get("cost_status") or None,
        "estimated_cost": cost,
        "counterfactual_cost": counterfactual,
        "savings": savings,
        "below_threshold": hit_pct is not None and hit_pct < CACHE_HIT_WARN_THRESHOLD_PCT,
    }


def format_usd(value: Optional[float]) -> str:
    """``$1.23``, ``$0.0040`` below a cent, ``n/a`` when unpriced."""
    if value is None:
        return "n/a"
    return f"${value:,.4f}" if abs(value) < 0.01 else f"${value:,.2f}"


def format_hit_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}%"
