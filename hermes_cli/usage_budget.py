"""Month-to-date usage per billing provider, measured against ``usage.budgets``.

A token plan is budgeted in tokens and pay-as-you-go in dollars. Most sessions on a token plan carry no
price at all (``cost_status = 'unknown'``), so tokens are always counted, money only where it is known,
and the number of unpriced sessions is reported instead of being shown as $0.
"""
from __future__ import annotations


def read_budgets(config: dict) -> dict[str, tuple[str, float]]:
    """``usage.budgets`` as ``{provider: ("tokens" | "usd", limit)}``; null or non-positive limits are no budget."""
    raw = ((config.get("usage") or {}).get("budgets") or {}) if isinstance(config, dict) else {}
    budgets: dict[str, tuple[str, float]] = {}
    for provider, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        for key, kind in (("monthly_tokens", "tokens"), ("monthly_usd", "usd")):
            value = spec.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                budgets[str(provider)] = (kind, float(value))
                break
    return budgets
