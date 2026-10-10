"""Pricing extraction for generic ``/models`` payloads.

Novita top-level rates, DeepInfra ``metadata.pricing``, and the alias-map shape;
rescales everything to the per-token strings ``usage_pricing`` expects.
"""

from typing import Any

# Generic ``/models`` pricing: an explicit ``unit`` beside the rates wins; without one, a token rate
# at or above $0.001/token ($1,000/MTok — no real model charges that) can only be a per-million quote.
_PRICING_UNIT_DIVISORS = {
    "per_token": 1, "per_1k_tokens": 1_000, "per_thousand_tokens": 1_000,
    "per_1m_tokens": 1_000_000, "per_million_tokens": 1_000_000,
}
_PER_MILLION_QUOTE_MIN = 0.001
_TOKEN_RATE_FIELDS = ("prompt", "completion", "cache_read", "cache_write")


def _normalize_token_rates(pricing: dict[str, Any], unit: Any) -> dict[str, Any]:
    """Rescale the generic path's token rates to per-token strings (the contract usage_pricing
    multiplies by 1e6), the way the Novita/DeepInfra branches already do for their known units."""
    rates: dict[str, float] = {}
    for key in _TOKEN_RATE_FIELDS:
        try:
            rates[key] = float(pricing[key])
        except (KeyError, TypeError, ValueError):
            continue
    divisor = _PRICING_UNIT_DIVISORS.get(str(unit or "").strip().lower())
    if divisor is None:
        divisor = 1_000_000 if any(v >= _PER_MILLION_QUOTE_MIN for v in rates.values()) else 1
    if divisor != 1:
        pricing.update({key: str(value / divisor) for key, value in rates.items()})
    return pricing


def _extract_pricing(payload: dict[str, Any]) -> dict[str, Any]:
    def _per_token(source: dict[str, Any], fields: dict[str, str], scale) -> dict[str, Any]:
        # Provider $/MTok (or Novita's 1/10_000-$ per M) -> per-token strings, the same path usage_pricing uses for OpenRouter.
        return {target: str(scale(float(source[key]))) for target, key in fields.items() if source.get(key) is not None}
    novita_fields = {"prompt": "input_token_price_per_m", "completion": "output_token_price_per_m"}
    if any(payload.get(k) is not None for k in novita_fields.values()):
        return _per_token(payload, novita_fields, lambda v: v / 10_000 / 1_000_000)
    # DeepInfra ships pricing under ``metadata.pricing`` in $/MTok.
    metadata = payload.get("metadata")
    deepinfra_pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
    deepinfra_fields = {"prompt": "input_tokens", "completion": "output_tokens", "cache_read": "cache_read_tokens"}
    if isinstance(deepinfra_pricing, dict) and any(k in deepinfra_pricing for k in deepinfra_fields.values()):
        return _per_token(deepinfra_pricing, deepinfra_fields, lambda v: v / 1_000_000)
    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    # Siblings late-import the facade inside functions; never a module-level cycle.
    from agent.model_metadata import _iter_nested_dicts
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        pricing: dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in {None, ""}:
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            return _normalize_token_rates(pricing, normalized.get("unit"))
    return {}
