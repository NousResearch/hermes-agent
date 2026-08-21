"""Cost / cache-hit reporting for `hermes sessions cost`.

Pure computation + formatting helpers that turn a session row (as
returned by ``SessionDB.list_sessions_rich`` / ``get_session_rich_row``)
into a cache-hit and savings breakdown.

The counterfactual ("what would this session have cost if the prompt
cache had never hit?") is deliberately an **estimate**: Hermes records
the aggregate ``estimated_cost_usd`` at call time but does not persist
the per-token price split, so the exact cold-price delta is unknowable
after the fact. We attribute the recorded cost to the input side in
proportion to token counts, then re-price the cache-read share as if it
had been billed at the cold input rate:

    savings = estimated_cost * (input_side / total_tokens)
                           * (cache_read / input_side_tokens)
                           * (1 / cache_hit_ratio - 1)

The cache-to-cold price ratio is configurable via
``cost.cache_hit_ratio`` in config.yaml (``0.10`` = cache reads are
billed at 10% of the cold input rate, e.g. DeepSeek's ~10x spread),
defaulting to a conservative 0.10 when unset or invalid.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Default assumption: cache reads are billed at 10% of the cold input
#: price (a conservative cross-provider default — DeepSeek ~10x, OpenAI
#: 5x for long contexts, Anthropic ~5x for 1h TTL).
DEFAULT_CACHE_HIT_RATIO = 0.10

#: Below this cache-hit rate a session gets the ``⚠`` marker — the signal
#: that something (workspace switch, toolset swap, compression) may have
#: invalidated the prompt prefix.
CACHE_HIT_WARN_THRESHOLD_PCT = 70.0

_TOKEN_KEYS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)


def cache_hit_ratio_from_config(config: Optional[Dict[str, Any]]) -> float:
    """Read ``cost.cache_hit_ratio`` from a loaded config dict.

    Returns ``DEFAULT_CACHE_HIT_RATIO`` when the key is missing, not a
    dict, or outside ``(0, 1]`` — a hand-edited config degrades to the
    conservative default instead of producing nonsense counterfactuals.
    """
    ratio = DEFAULT_CACHE_HIT_RATIO
    if not isinstance(config, dict):
        return ratio
    cost_cfg = config.get("cost")
    if not isinstance(cost_cfg, dict):
        return ratio
    raw = cost_cfg.get("cache_hit_ratio")
    if raw is None:
        return ratio
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return ratio
    if 0.0 < parsed <= 1.0:
        return parsed
    return ratio


def session_cost_breakdown(
    session_row: Dict[str, Any],
    *,
    cache_hit_ratio: float = DEFAULT_CACHE_HIT_RATIO,
) -> Dict[str, Any]:
    """Compute the cache-hit + savings breakdown for one session row.

    The row is a ``sessions`` table dict (any of the ``SessionDB`` list /
    get shapes); every token column defaults to 0 and ``estimated_cost``
    defaults to None when absent, so rows from older schemas or test
    doubles degrade safely.

    Returns a dict with:

    * ``input_tokens`` / ``cache_read_tokens`` / ``cache_write_tokens`` /
      ``output_tokens`` — the per-type totals (ints)
    * ``input_side_tokens`` — input + cache_read + cache_write
    * ``total_tokens`` — input_side + output
    * ``cache_hit_pct`` — ``cache_read / input_side * 100`` (None when
      there are no input-side tokens)
    * ``estimated_cost`` — the recorded ``estimated_cost_usd`` (or None)
    * ``counterfactual_cost`` — estimated cost if cache hits had been 0%
      (None when ``estimated_cost`` is None)
    * ``savings`` — counterfactual − estimated (0.0 when there is cost
      but nothing to save; None when cost is unknown)
    * ``below_threshold`` — True when ``cache_hit_pct`` is not None and
      below ``CACHE_HIT_WARN_THRESHOLD_PCT``
    """
    inp = _int_or_zero(session_row.get("input_tokens"))
    cr = _int_or_zero(session_row.get("cache_read_tokens"))
    cw = _int_or_zero(session_row.get("cache_write_tokens"))
    out = _int_or_zero(session_row.get("output_tokens"))

    raw_cost = session_row.get("estimated_cost_usd")
    cost: Optional[float] = None
    if raw_cost is not None:
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError):
            cost = None

    input_side = inp + cr + cw
    total = input_side + out
    cache_hit_pct: Optional[float] = (
        (cr / input_side * 100.0) if input_side else None
    )

    counterfactual_cost: Optional[float] = None
    savings: Optional[float] = None
    if cost is not None and input_side and cr:
        ratio = cache_hit_ratio
        if not (0.0 < ratio <= 1.0):
            ratio = DEFAULT_CACHE_HIT_RATIO
        input_share = input_side / total if total else 0.0
        cache_share = cr / input_side
        savings = cost * input_share * cache_share * (1.0 / ratio - 1.0)
        counterfactual_cost = cost + savings
    elif cost is not None:
        # Cost recorded but nothing cacheable — the counterfactual is the
        # same bill (no savings).
        counterfactual_cost = cost
        savings = 0.0

    return {
        "input_tokens": inp,
        "cache_read_tokens": cr,
        "cache_write_tokens": cw,
        "output_tokens": out,
        "input_side_tokens": input_side,
        "total_tokens": total,
        "cache_hit_pct": cache_hit_pct,
        "estimated_cost": cost,
        "counterfactual_cost": counterfactual_cost,
        "savings": savings,
        "below_threshold": (
            cache_hit_pct is not None
            and cache_hit_pct < CACHE_HIT_WARN_THRESHOLD_PCT
        ),
    }


def format_usd(value: Optional[float]) -> str:
    """``$1.23`` / ``$0.0040`` (4 decimals for sub-cent amounts) / ``—``."""
    if value is None:
        return "—"
    if abs(value) < 0.01:
        return f"${value:,.4f}"
    return f"${value:,.2f}"


def format_hit_pct(value: Optional[float]) -> str:
    """``36.9%`` / ``—`` for unknown."""
    if value is None:
        return "—"
    return f"{value:.1f}%"


def format_tokens(value: Optional[int]) -> str:
    """``12,345`` (thousands separator) / ``0`` / ``—`` for unknown."""
    if value is None:
        return "—"
    return f"{int(value):,}"


def _int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
