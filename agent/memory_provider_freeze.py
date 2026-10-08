"""Recursive JSON validation and freezing for memory-provider observations."""

from __future__ import annotations

import math
from typing import Any, List, Optional

from agent import memory_provider as _provider


_UNSUPPORTED = object()


def _freeze_scalar(value: Any, budget: List[int]) -> Any:
    if value is None or type(value) in (bool, int):
        _provider._account_encoded_bytes(budget, _provider._encoded_json_scalar_size(value))
        return value
    if isinstance(value, str):
        value = str.__str__(value)
        if str.__len__(value) > _provider.MAX_MEMORY_OBSERVATION_STRING_CHARS:
            raise ValueError("observation payload string is too long")
        _provider._account_encoded_bytes(budget, _provider._encoded_json_scalar_size(value))
        return value
    if isinstance(value, int):
        value = int.__int__(value)
        _provider._account_encoded_bytes(budget, _provider._encoded_json_scalar_size(value))
        return value
    if isinstance(value, float):
        value = float.__float__(value)
        if not math.isfinite(value):
            raise ValueError("observation payload contains a non-finite number")
        _provider._account_encoded_bytes(budget, _provider._encoded_json_scalar_size(value))
        return value
    return _UNSUPPORTED


def _freeze_mapping(
    value: dict, *, depth: int, budget: List[int], operation_budget: Optional[List[int]]
) -> Any:
    item_count = dict.__len__(value)
    if item_count > _provider.MAX_MEMORY_OBSERVATION_ITEMS:
        raise ValueError("observation payload object has too many keys")
    _provider._account_encoded_bytes(budget, 2)
    frozen = {}
    for index, (key, child) in enumerate(dict.items(value)):
        if not isinstance(key, str):
            raise ValueError("observation payload object keys must be bounded strings")
        key = str.__str__(key)
        if str.__len__(key) > _provider.MAX_MEMORY_OBSERVATION_STRING_CHARS:
            raise ValueError("observation payload object keys must be bounded strings")
        if index:
            _provider._account_encoded_bytes(budget, 1)
        _provider._account_encoded_bytes(
            budget, _provider._encoded_json_scalar_size(key) + 1
        )
        frozen[key] = _freeze_json_value(
            child,
            depth=depth + 1,
            budget=budget,
            operation_budget=operation_budget,
        )
    return _provider._FrozenDict(frozen)


def _freeze_sequence(
    value: list | tuple, *, depth: int, budget: List[int], operation_budget: Optional[List[int]]
) -> tuple:
    item_count = list.__len__(value) if isinstance(value, list) else tuple.__len__(value)
    if item_count > _provider.MAX_MEMORY_OBSERVATION_ITEMS:
        raise ValueError("observation payload array has too many items")
    _provider._account_encoded_bytes(budget, 2)
    frozen = []
    for index in range(item_count):
        if index:
            _provider._account_encoded_bytes(budget, 1)
        child = (
            list.__getitem__(value, index)
            if isinstance(value, list)
            else tuple.__getitem__(value, index)
        )
        frozen.append(
            _freeze_json_value(
                child,
                depth=depth + 1,
                budget=budget,
                operation_budget=operation_budget,
            )
        )
    return tuple(frozen)


def _freeze_json_value(
    value: Any,
    *,
    depth: int = 0,
    budget: Optional[List[int]] = None,
    operation_budget: Optional[List[int]] = None,
) -> Any:
    """Validate and recursively freeze one JSON-safe observation value.

    ``budget`` tracks the node/byte allowance for this payload. ``operation_budget``
    is an additional shared counter spanning all payloads in one prefetch.
    """
    if budget is None:
        budget = [_provider.MAX_MEMORY_OBSERVATION_NODES]
    if depth > _provider.MAX_MEMORY_OBSERVATION_DEPTH:
        raise ValueError("observation payload is too deeply nested")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("observation payload has too many nodes")
    if operation_budget is not None:
        operation_budget[0] -= 1
        if operation_budget[0] < 0:
            raise ValueError("observation operation exhausted node budget")

    scalar = _freeze_scalar(value, budget)
    if scalar is not _UNSUPPORTED:
        return scalar
    if isinstance(value, dict):
        return _freeze_mapping(
            value, depth=depth, budget=budget, operation_budget=operation_budget
        )
    if isinstance(value, (list, tuple)):
        return _freeze_sequence(
            value, depth=depth, budget=budget, operation_budget=operation_budget
        )
    raise TypeError("observation payload must contain only JSON-safe values")
