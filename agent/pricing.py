"""Model pricing helpers for session cost estimation."""

from __future__ import annotations


MODEL_PRICING = {
    "claude-sonnet-4-20250514": (3.00, 15.00, 0.30),
    "claude-opus-4-20250514": (15.00, 75.00, 1.50),
    "claude-haiku-3-5-20241022": (0.80, 4.00, 0.08),
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "o3-mini": (1.10, 4.40, 0.55),
    "deepseek-chat": (0.27, 1.10, 0.07),
    "deepseek-reasoner": (0.55, 2.19, 0.14),
    "meta-llama/llama-4-maverick": (0.20, 0.60, 0.05),
}


def _resolve_pricing(model: str) -> tuple[float, float, float] | None:
    """Resolve pricing with loose model-name matching."""
    if not model:
        return None

    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    model_name = model.split("/", 1)[-1]
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    for name, pricing in MODEL_PRICING.items():
        if (
            model in name
            or name in model
            or model_name in name
            or name.split("/", 1)[-1] in model_name
        ):
            return pricing

    return None


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate USD cost for one model call."""
    pricing = _resolve_pricing(model)
    if pricing is None:
        return 0.0

    input_rate, output_rate, cached_input_rate = pricing
    cached_tokens = max(0, cached_tokens)
    billable_prompt_tokens = max(0, prompt_tokens - cached_tokens)

    return (
        (billable_prompt_tokens / 1_000_000) * input_rate
        + (completion_tokens / 1_000_000) * output_rate
        + (cached_tokens / 1_000_000) * cached_input_rate
    )
