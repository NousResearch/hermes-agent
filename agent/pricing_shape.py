"""Pricing-entry shape tolerance for endpoint ``/models`` metadata (PR 3).

OpenAI-compatible ``/models`` endpoints do not agree on a pricing convention:

- OpenRouter-style: ``pricing.prompt = "0.00000075"`` — **dollars per token**,
  keys ``prompt``/``completion``, flat.
- Per-million style (observed on a hosted LLM gateway): ``pricing.global.prompt
  = 0.75`` — **dollars per million**, keys ``prompt``/``completions``, nested
  under ``global`` with a ``regional_increase_percent`` sibling.

The parser assumed the OpenRouter shape unconditionally. On a per-million
endpoint that produced a 1,000,000× inflated input rate ($750,000/M for a
$0.75/M model), dropped the output rate entirely (``completions`` alias not
recognized), and displayed a session cost of hundreds of thousands of dollars.

Unit inference is CONTAINER-level, not per-value: if any core token-rate field
(``prompt``/``completion``/``input``/``output``) is >= $0.01/token — a price no
real model has — the whole container is treated as dollars-per-million. A
per-value threshold would misclassify sub-cent $/M fields (cache reads, batch
tiers at $0.005/M) as per-token inside an otherwise per-million container.
Values below the ceiling in an ambiguous container are already per-token.

Nested containers are unwrapped ONLY for known nest keys (``global``/
``default``/``usd``) — other nested metadata (architecture, limits) must not
bleed into pricing. Flat root keys win over unwrapped nested values when both
define the same key.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# A genuine per-token price >= $0.01/token means $10,000+/M — no real model.
# Used to infer the container's unit from its core token-rate fields.
_PER_TOKEN_CEILING = Decimal("0.01")

_NEST_KEYS = ("global", "default", "usd")

_PROMPT_KEYS = ("prompt", "input", "input_cost")
_COMPLETION_KEYS = ("completion", "completions", "output", "output_cost")
_CACHE_READ_KEYS = ("cache_read", "cached_prompt", "input_cache_read")
_CACHE_WRITE_KEYS = ("cache_write", "cache_creation", "input_cache_write")


def _to_decimal(raw: Any) -> Optional[Decimal]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return Decimal(str(raw))
    except Exception:  # noqa: BLE001 — malformed metadata must not break pricing
        return None


def unwrap_pricing_container(pricing: Any) -> Dict[str, Any]:
    """Flatten known nest containers (``global``/``default``/``usd``) into a flat
    key->value dict. ONLY nest keys are flattened — unrelated nested objects
    (``architecture``, ``limits``, ...) must not bleed their keys into pricing.
    Flat root keys take precedence over unwrapped nested values."""
    if not isinstance(pricing, dict):
        return {}
    flat: Dict[str, Any] = {}
    for nest_key in _NEST_KEYS:
        nested = pricing.get(nest_key)
        if isinstance(nested, dict):
            flat.update(nested)
    for key, value in pricing.items():
        if key in _NEST_KEYS or isinstance(value, dict):
            continue
        flat[key] = value
    return flat


def _is_per_million_container(flat: Dict[str, Any]) -> bool:
    """True when the container's token-rate fields are per-MILLION figures.

    Inference anchor: prompt/completion/input/output AND cache rates — every
    convention prices the core fields, and cache rates are always CHEAPER than
    completion (a cache price >= the per-token ceiling can only be $/M). If any
    anchor is >= the ceiling, the whole container is $/M (mixed-unit containers
    would otherwise normalize a $0.005/M cache-read as $0.005/token = $5,000/M)."""
    for keys in (_PROMPT_KEYS, _COMPLETION_KEYS, _CACHE_READ_KEYS, _CACHE_WRITE_KEYS):
        for key in keys:
            value = _to_decimal(flat.get(key))
            if value is not None and value >= _PER_TOKEN_CEILING:
                return True
    return False


def extract_pricing_fields(pricing: Any) -> Dict[str, Optional[Decimal]]:
    """Shape-tolerant (key -> $/million) extraction from a pricing container."""
    flat = unwrap_pricing_container(pricing)
    per_million_units = _is_per_million_container(flat)

    def first_value(*keys: str) -> Optional[Decimal]:
        for key in keys:
            value = _to_decimal(flat.get(key))
            if value is not None:
                return value
        return None

    def rate(*keys: str) -> Optional[Decimal]:
        value = first_value(*keys)
        if value is None:
            return None
        if per_million_units:
            return value
        # $/token container: scale to $/M
        return value * Decimal(1_000_000)

    return {
        "prompt": rate(*_PROMPT_KEYS),
        "completion": rate(*_COMPLETION_KEYS),
        "cache_read": rate(*_CACHE_READ_KEYS),
        "cache_write": rate(*_CACHE_WRITE_KEYS),
        "request": first_value("request"),
    }
