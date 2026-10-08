"""Plugin-host wire encoders for memory observation records."""

from __future__ import annotations

from typing import Any, Callable, Optional


def _encode_memory_observation(
    value: Any, refs: Optional[Callable], depth: int, budget: list[int]
) -> Any:
    from hermes_cli.plugin_host_wire import _opaque_wire_value, encode
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_BYTES,
        MAX_MEMORY_OBSERVATION_FIELD_CHARS,
        MemoryObservation,
        _encoded_json_scalar_size,
        _freeze_memory_observation_payload,
        _thaw_json_value,
    )

    if type(value) is MemoryObservation:
        source_kind = value.source_kind
        schema = value.schema
        version = value.version
        provider = value.provider
        payload = value.payload
    elif type(value) is dict:
        source_kind = dict.get(value, "source_kind")
        schema = dict.get(value, "schema")
        version = dict.get(value, "version")
        provider = dict.get(value, "provider", "")
        payload = dict.get(value, "payload")
    else:
        return encode(_opaque_wire_value(value), refs, depth + 1)

    if type(version) is int:
        try:
            if _encoded_json_scalar_size(version) > MAX_MEMORY_OBSERVATION_BYTES:
                raise ValueError("version too large")
        except (TypeError, ValueError, OverflowError):
            version = _opaque_wire_value(version)
    elif type(version) is not bool:
        version = _opaque_wire_value(version)

    def bounded_text(field: Any) -> Any:
        if not isinstance(field, str):
            return _opaque_wire_value(field)
        text = str.__str__(field)
        return (
            text
            if str.__len__(text) <= MAX_MEMORY_OBSERVATION_FIELD_CHARS
            else _opaque_wire_value(field)
        )

    try:
        frozen_payload, _ = _freeze_memory_observation_payload(
            payload, operation_budget=budget
        )
        payload = _thaw_json_value(frozen_payload)
    except (TypeError, ValueError, OverflowError):
        payload = _opaque_wire_value(payload)

    return {
        "__record__": "MemoryObservation",
        "fields": {
            "source_kind": encode(bounded_text(source_kind), refs, depth + 1),
            "schema": encode(bounded_text(schema), refs, depth + 1),
            "version": encode(version, refs, depth + 1),
            "payload": encode(payload, refs, depth + 1),
            "provider": encode(bounded_text(provider), refs, depth + 1),
        },
    }


def _encode_memory_prefetch_result(
    value: Any, refs: Optional[Callable], depth: int
) -> Any:
    from hermes_cli.plugin_host_wire import encode
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
        MAX_MEMORY_OBSERVATION_OPERATION_NODES,
        MemoryPrefetchResult,
    )

    if type(value) is not MemoryPrefetchResult:
        return None
    observations = value.observations
    if type(observations) is list:
        count = list.__len__(observations)
        get_item = lambda index: list.__getitem__(observations, index)
    elif isinstance(observations, tuple):
        count = tuple.__len__(observations)
        get_item = lambda index: tuple.__getitem__(observations, index)
    else:
        count = 0
        get_item = lambda _index: None
    count = min(count, MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1)
    budget = [MAX_MEMORY_OBSERVATION_OPERATION_NODES]
    encoded_observations = [
        _encode_memory_observation(get_item(index), refs, depth + 1, budget)
        for index in range(count)
    ]
    context = value.context
    if isinstance(context, str):
        context = str.__str__(context)
    else:
        from hermes_cli.plugin_host_wire import _opaque_wire_value

        context = _opaque_wire_value(context)
    return {
        "__record__": "MemoryPrefetchResult",
        "fields": {
            "context": encode(context, refs, depth + 1),
            "observations": encoded_observations,
        },
    }


def _encode_memory_observation_tuple(
    value: Any, refs: Optional[Callable], depth: int
) -> Any:
    from hermes_cli.plugin_host_wire import encode
    from agent.memory_provider import (
        MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
        MAX_MEMORY_OBSERVATION_OPERATION_NODES,
        MemoryObservation,
    )

    if not isinstance(value, tuple):
        return None
    count = tuple.__len__(value)
    if not count or type(tuple.__getitem__(value, 0)) is not MemoryObservation:
        return None
    count = min(count, MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1)
    budget = [MAX_MEMORY_OBSERVATION_OPERATION_NODES]
    return {
        "__record__": "MemoryObservationTuple",
        "fields": {
            "items": [
                _encode_memory_observation(
                    tuple.__getitem__(value, index), refs, depth + 1, budget
                )
                for index in range(count)
            ]
        },
    }


def encode_memory_value(value: Any, refs: Optional[Callable], depth: int) -> Any:
    """Encode a memory observation record, or return None for ordinary wire values."""
    if type(value).__module__ == "agent.memory_provider":
        if type(value).__name__ == "MemoryPrefetchResult":
            return _encode_memory_prefetch_result(value, refs, depth)
        if type(value).__name__ == "MemoryObservation":
            from agent.memory_provider import MAX_MEMORY_OBSERVATION_OPERATION_NODES

            return _encode_memory_observation(
                value, refs, depth, [MAX_MEMORY_OBSERVATION_OPERATION_NODES]
            )
    if isinstance(value, tuple) and tuple.__len__(value):
        first = tuple.__getitem__(value, 0)
        if (
            type(first).__module__ == "agent.memory_provider"
            and type(first).__name__ == "MemoryObservation"
        ):
            encoded = _encode_memory_observation_tuple(value, refs, depth)
            if encoded is not None:
                return encoded
    return None
