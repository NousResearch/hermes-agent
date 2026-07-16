"""Expensive-model confirmation helpers for model selection surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

from agent.models_dev import ModelInfo, PROVIDER_TO_MODELS_DEV


INPUT_COST_WARNING_THRESHOLD = Decimal("20")
OUTPUT_COST_WARNING_THRESHOLD = Decimal("100")
GPT55_PRO_OPENROUTER_ID = "openai/gpt-5.5-pro"
GPT55_SUGGESTION = "did you mean to select openai/gpt-5.5?"


@dataclass(frozen=True)
class ExpensiveModelWarning:
    """Confirmation payload for models above Hermes' cost guardrail."""

    model: str
    provider: str
    input_cost_per_million: Optional[Decimal]
    output_cost_per_million: Optional[Decimal]
    source: str
    message: str


def _to_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _format_money(value: Optional[Decimal]) -> str:
    if value is None:
        return "unknown"
    return f"${value:.2f}/M"


def _pricing_from_model_info(
    model_info: Optional[ModelInfo],
) -> tuple[Optional[Decimal], Optional[Decimal], str]:
    if model_info is None or not model_info.has_cost_data():
        return None, None, ""
    return (
        _to_decimal(model_info.cost_input),
        _to_decimal(model_info.cost_output),
        "models.dev",
    )


def _known_models_dev_provider(provider: Optional[str]) -> Optional[str]:
    normalized = (provider or "").strip().lower()
    if not normalized:
        return None
    return PROVIDER_TO_MODELS_DEV.get(normalized)


def _can_trust_model_info_pricing(
    provider: Optional[str],
    model_info: Optional[ModelInfo],
) -> bool:
    expected_provider = _known_models_dev_provider(provider)
    if not expected_provider or model_info is None:
        return False

    actual_provider = str(getattr(model_info, "provider_id", "") or "").strip().lower()
    return not actual_provider or actual_provider == expected_provider


def _can_trust_pricing_lookup(
    model_name: str,
    *,
    provider: Optional[str],
    base_url: Optional[str],
) -> bool:
    try:
        from agent.usage_pricing import resolve_billing_route

        route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    except Exception:
        return False
    return route.billing_mode != "unknown"


def expensive_model_warning(
    model_name: str,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_info: Optional[ModelInfo] = None,
) -> Optional[ExpensiveModelWarning]:
    """Return a warning payload when known pricing exceeds safety thresholds.

    The guard only triggers when pricing is known. Callers should use this after
    model resolution so aliases and provider-specific model IDs have settled.
    """
    model = (model_name or "").strip()
    if not model:
        return None

    input_cost: Optional[Decimal] = None
    output_cost: Optional[Decimal] = None
    source = ""

    if _can_trust_model_info_pricing(provider, model_info):
        input_cost, output_cost, source = _pricing_from_model_info(model_info)

    if (
        input_cost is None
        and output_cost is None
        and _known_models_dev_provider(provider)
    ):
        try:
            from agent.models_dev import get_model_info

            input_cost, output_cost, source = _pricing_from_model_info(
                get_model_info(provider, model)
            )
        except Exception:
            pass

    if (
        input_cost is None
        and output_cost is None
        and _can_trust_pricing_lookup(model, provider=provider, base_url=base_url)
    ):
        try:
            from agent.usage_pricing import get_pricing_entry

            entry = get_pricing_entry(
                model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception:
            entry = None
        if entry is not None:
            input_cost = entry.input_cost_per_million
            output_cost = entry.output_cost_per_million
            source = entry.source

    over_input = (
        input_cost is not None and input_cost > INPUT_COST_WARNING_THRESHOLD
    )
    over_output = (
        output_cost is not None and output_cost > OUTPUT_COST_WARNING_THRESHOLD
    )
    if not over_input and not over_output:
        return None

    lines = [
        "!!! EXPENSIVE MODEL WARNING !!!",
        "",
        f"{model} has known pricing above Hermes' safety threshold.",
        f"Input tokens: {_format_money(input_cost)}",
        f"Output tokens: {_format_money(output_cost)}",
        (
            "Threshold: more than $20/M input tokens or more than "
            "$100/M output tokens."
        ),
    ]
    if source:
        lines.append(f"Pricing source: {source}.")
    if model.lower() == GPT55_PRO_OPENROUTER_ID:
        lines.append(GPT55_SUGGESTION)
    lines.append("Confirm only if you intend to use this model.")

    return ExpensiveModelWarning(
        model=model,
        provider=(provider or "").strip(),
        input_cost_per_million=input_cost,
        output_cost_per_million=output_cost,
        source=source or "unknown",
        message="\n".join(lines),
    )
