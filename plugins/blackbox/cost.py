"""Cost reconciliation for blackbox turn telemetry."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost


_STATUS_RANK = {
    "included": 0,
    "actual": 1,
    "estimated": 1,
    "partial": 2,
    "unknown": 3,
}


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def compute_turn_cost(
    model: str,
    provider: Optional[str],
    base_url: Optional[str],
    calls: list[dict],
) -> tuple[float | None, str]:
    """Price each provider call and return reconciled turn cost/status."""
    if not calls:
        return 0.0, "included"

    known_total = Decimal("0")
    known_count = 0
    unknown_count = 0
    known_statuses: list[str] = []

    for call in calls:
        try:
            usage = CanonicalUsage(
                input_tokens=_int_value(call.get("input_tokens")),
                output_tokens=_int_value(call.get("output_tokens")),
                cache_read_tokens=_int_value(call.get("cache_read_tokens")),
                cache_write_tokens=_int_value(call.get("cache_write_tokens")),
                reasoning_tokens=_int_value(call.get("reasoning_tokens")),
            )
            result = estimate_usage_cost(
                model,
                usage,
                provider=provider,
                base_url=base_url,
            )
        except Exception:
            unknown_count += 1
            continue

        amount = result.amount_usd
        status = result.status or "unknown"
        if amount is None or status == "unknown":
            unknown_count += 1
            continue

        known_total += Decimal(str(amount))
        known_count += 1
        known_statuses.append("estimated" if status == "actual" else status)

    if unknown_count and known_count:
        return float(known_total), "partial"
    if unknown_count:
        return None, "unknown"

    worst = max(known_statuses or ["included"], key=lambda s: _STATUS_RANK.get(s, 3))
    return float(known_total), worst
