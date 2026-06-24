"""Telegram-only OpenRouter per-turn cost footer helpers.

This is deliberately separate from ``runtime_footer``: runtime footer is an
opt-in diagnostic footer, while this feature is a small billing breadcrumb for
OpenRouter-backed Telegram turns.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlparse


def _platform_key(platform_key: Any) -> str:
    return str(getattr(platform_key, "value", platform_key) or "").strip().lower()


def _provider_key(agent_result: Mapping[str, Any]) -> str:
    return str(
        agent_result.get("billing_provider")
        or agent_result.get("provider")
        or ""
    ).strip().lower()


def _is_openrouter_route(agent_result: Mapping[str, Any]) -> bool:
    provider = _provider_key(agent_result)
    if provider == "openrouter" or provider == "custom:openrouter":
        return True
    base_url = str(
        agent_result.get("billing_base_url")
        or agent_result.get("base_url")
        or ""
    ).strip().lower()
    if not base_url:
        return False
    try:
        host = urlparse(base_url).netloc.lower()
    except Exception:
        host = ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _format_usd(amount: float) -> str:
    """Format tiny OpenRouter costs without rounding them all to $0.0000."""
    if amount < 0:
        amount = 0.0
    decimals = 6 if amount and amount < 0.0001 else 4
    return f"${amount:.{decimals}f}"


def _api_calls(mapping: Mapping[str, Any], key: str) -> int:
    try:
        return int(mapping.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _cost_amount(mapping: Mapping[str, Any], key: str) -> float | None:
    raw_amount = mapping.get(key)
    if raw_amount is None:
        return None
    try:
        return float(raw_amount)
    except (TypeError, ValueError):
        return None


def _cost_prefix(status: str) -> str:
    return "~" if str(status or "").strip().lower() == "estimated" else ""


def build_openrouter_cost_line(
    *,
    platform_key: Any,
    agent_result: Mapping[str, Any],
) -> str:
    """Return a short per-turn cost line, or ``""`` when not applicable.

    Scope is intentionally narrow:
    - Telegram only;
    - main runtime provider must be OpenRouter;
    - only actual model API turns (``turn_api_calls`` > 0).
    """
    if _platform_key(platform_key) != "telegram":
        return ""
    if not _is_openrouter_route(agent_result):
        return ""

    if _api_calls(agent_result, "turn_api_calls") <= 0:
        return ""

    status = str(agent_result.get("turn_cost_status") or "unknown").strip().lower()
    if status == "unknown":
        return "💸 OpenRouter: cost n/a per request"

    amount = _cost_amount(agent_result, "turn_estimated_cost_usd")
    if amount is None:
        return "💸 OpenRouter: cost n/a per request"

    return f"💸 OpenRouter: {_cost_prefix(status)}{_format_usd(amount)} per request"


def build_openrouter_background_review_cost_line(
    *,
    platform_key: Any,
    main_agent_result: Mapping[str, Any],
    background_review_result: Mapping[str, Any],
) -> str:
    """Return a Telegram follow-up line with foreground + background review cost.

    The background review runs after the main answer is returned, so this line is
    intended for a follow-up message rather than the main response footer.
    """
    if _platform_key(platform_key) != "telegram":
        return ""
    if not _is_openrouter_route(main_agent_result):
        return ""
    if _api_calls(main_agent_result, "turn_api_calls") <= 0:
        return ""
    if _api_calls(background_review_result, "api_calls") <= 0:
        return ""

    main_status = str(main_agent_result.get("turn_cost_status") or "unknown").strip().lower()
    bg_status = str(background_review_result.get("cost_status") or "unknown").strip().lower()
    main_amount = _cost_amount(main_agent_result, "turn_estimated_cost_usd")
    bg_amount = _cost_amount(background_review_result, "estimated_cost_usd")
    if main_status == "unknown" or bg_status == "unknown" or main_amount is None or bg_amount is None:
        return "💸 OpenRouter total: cost n/a per request (bg cost n/a)"

    total = max(0.0, main_amount) + max(0.0, bg_amount)
    total_prefix = "~" if "estimated" in {main_status, bg_status} else ""
    return (
        f"💸 OpenRouter total: {total_prefix}{_format_usd(total)} per request "
        f"(main {_cost_prefix(main_status)}{_format_usd(main_amount)} + "
        f"bg {_cost_prefix(bg_status)}{_format_usd(bg_amount)})"
    )
