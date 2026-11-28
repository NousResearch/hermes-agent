"""Bounded integer reads from ``/models`` payloads (sibling of ``model_metadata``)."""

from __future__ import annotations

from typing import Any, Optional


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        result = int(value.strip().replace(",", "") if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return result if minimum <= result <= maximum else None


def _extract_first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() in keyset:
                coerced = _coerce_reasonable_int(value)
                if coerced is not None:
                    return coerced
    return None
