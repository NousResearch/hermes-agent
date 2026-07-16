from __future__ import annotations

import math
from numbers import Real
from typing import Any

_KNOWN_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})


def effective_reasoning_effort(reasoning_config: dict[str, Any] | None) -> str:
    """Return the live reasoning state without inventing a provider default."""
    if not isinstance(reasoning_config, dict):
        return "provider-default"
    if reasoning_config.get("enabled") is False:
        return "none"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort if effort in _KNOWN_REASONING_EFFORTS else "provider-default"


def format_reasoning_label(
    reasoning_config: dict[str, Any] | None,
    *,
    compact: bool = False,
) -> str:
    """Format the effective reasoning effort for a runtime display."""
    effort = effective_reasoning_effort(reasoning_config)
    if compact:
        return f"r:{'default' if effort == 'provider-default' else effort}"
    return f"reasoning {effort}"


def _format_usd(amount: float) -> str:
    return f"{amount:.4f}" if amount < 0.01 else f"{amount:.2f}"


def format_session_cost(
    amount_usd: object,
    status: object,
    *,
    compact: bool = False,
) -> str:
    """Format cumulative session cost while preserving billing semantics."""
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "included":
        return "included" if compact else "cost included"
    unavailable = "cost n/a" if compact else "cost unavailable"
    if normalized_status not in {"actual", "estimated"}:
        return unavailable
    if isinstance(amount_usd, bool) or not isinstance(amount_usd, Real):
        return unavailable
    amount = float(amount_usd)
    if not math.isfinite(amount) or amount < 0:
        return unavailable
    prefix = "~$" if normalized_status == "estimated" else "$"
    value = f"{prefix}{_format_usd(amount)}"
    return value if compact else f"cost {value}"
