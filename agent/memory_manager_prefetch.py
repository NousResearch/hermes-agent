"""Bounded normalization for memory-provider prefetch results."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional

from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_BYTES,
    MAX_MEMORY_OBSERVATION_FIELD_CHARS,
    MAX_MEMORY_OBSERVATION_ITEMS,
    MAX_MEMORY_OBSERVATIONS,
    MemoryObservation,
    MemoryPrefetchResult,
    _encoded_json_scalar_size,
    _freeze_memory_observation_payload,
    _thaw_json_value,
)

logger = logging.getLogger("agent.memory_manager")


@dataclass(frozen=True)
class _NormalizedPrefetchResult:
    """Validated provider result plus private operation-boundary metadata."""

    result: MemoryPrefetchResult
    observation_sizes: tuple[int, ...]
    truncated_reason: Optional[str] = None


def _builtin_dict_fields(value: dict, names: tuple[str, ...]) -> dict[str, Any]:
    """Read selected fields without hashing or comparing provider-controlled keys."""
    wanted = set(names)
    fields = {}
    for index, (key, child) in enumerate(dict.items(value)):
        if isinstance(key, str):
            key = str.__str__(key)
            if key in wanted:
                fields[key] = child
        if index + 1 >= MAX_MEMORY_OBSERVATION_ITEMS:
            break
    return fields


def coerce_prefetch_result(raw_result: Any) -> Any:
    """Restore dict-shaped structured results without calling mapping overrides."""
    if isinstance(raw_result, dict):
        fields = _builtin_dict_fields(raw_result, ("context", "observations"))
        if "context" not in fields:
            return raw_result
        raw_context: Any = fields["context"]
        raw_observations: Any = fields.get("observations", ())
        if type(raw_observations) is not list and not isinstance(raw_observations, tuple):
            raw_observations = ()
        return MemoryPrefetchResult(
            context=raw_context,
            observations=raw_observations,
        )
    return raw_result


def _invalid_observation(provider: Any, exc: Exception) -> None:
    logger.warning(
        "Memory provider '%s' returned a malformed prefetch observation; dropping it: %s",
        provider.name,
        exc,
    )


def _normalize_observation(
    provider: Any, candidate: Any, traversal_budget: Optional[list[int]]
) -> Optional[tuple[MemoryObservation, int]]:
    try:
        if isinstance(candidate, dict):
            fields = _builtin_dict_fields(
                candidate, ("source_kind", "schema", "version", "provider", "payload")
            )
            source_kind: Any = fields.get("source_kind")
            schema: Any = fields.get("schema")
            version: Any = fields.get("version")
            provider_name: Any = fields.get("provider", "")
            payload: Any = fields.get("payload")
            candidate = MemoryObservation(
                source_kind=source_kind,
                schema=schema,
                version=version,
                provider=provider_name,
                payload=payload,
            )
        elif provider.name == "builtin" and isinstance(candidate, Mapping):
            # Preserve the trusted builtin provider's legacy Mapping format.
            # External provider mapping methods never run on the turn thread.
            source_kind: Any = candidate.get("source_kind")
            schema: Any = candidate.get("schema")
            version: Any = candidate.get("version")
            provider_name: Any = candidate.get("provider", "")
            payload: Any = candidate.get("payload")
            candidate = MemoryObservation(
                source_kind=source_kind,
                schema=schema,
                version=version,
                provider=provider_name,
                payload=payload,
            )
        if type(candidate) is not MemoryObservation:
            raise TypeError("observation has the wrong type")
        source_kind = candidate.source_kind
        schema = candidate.schema
        raw_version = candidate.version
        raw_provider = candidate.provider
        if not isinstance(source_kind, str) or not isinstance(schema, str):
            raise ValueError("observation source_kind or schema is invalid")
        source_kind = str.__str__(source_kind)
        schema = str.__str__(schema)
        if (
            not source_kind
            or str.__len__(source_kind) > MAX_MEMORY_OBSERVATION_FIELD_CHARS
        ):
            raise ValueError("observation source_kind is invalid")
        if not schema or str.__len__(schema) > MAX_MEMORY_OBSERVATION_FIELD_CHARS:
            raise ValueError("observation schema is invalid")
        if (
            isinstance(raw_version, bool)
            or not isinstance(raw_version, int)
        ):
            raise ValueError("observation version is invalid")
        # Convert int subclasses with the base implementation before comparing or encoding.
        version = int.__int__(raw_version)
        if version < 1:
            raise ValueError("observation version is invalid")
        if _encoded_json_scalar_size(version) > MAX_MEMORY_OBSERVATION_BYTES:
            raise ValueError("observation version is too large")
        provider_name = provider.name
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("provider name is invalid")
        provider_name = str.__str__(provider_name)
        if str.__len__(provider_name) > MAX_MEMORY_OBSERVATION_FIELD_CHARS:
            raise ValueError("provider name is too long")
        if not isinstance(raw_provider, str):
            raise ValueError("observation provider is invalid")
        observation_provider = str.__str__(raw_provider)
        if observation_provider not in ("", provider_name):
            raise ValueError("observation provider does not match its source provider")

        frozen_payload, _payload_bytes = _freeze_memory_observation_payload(
            candidate.payload, operation_budget=traversal_budget
        )
        encoded = json.dumps(
            {
                "source_kind": source_kind,
                "provider": provider_name,
                "schema": schema,
                "version": version,
                "payload": _thaw_json_value(frozen_payload),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_MEMORY_OBSERVATION_BYTES:
            raise ValueError("observation envelope is too large")
        return (
            MemoryObservation(
                source_kind=source_kind,
                provider=provider_name,
                schema=schema,
                version=version,
                payload=frozen_payload,
            ),
            len(encoded),
        )
    except Exception as exc:  # health: allow BLE001 -- malformed candidates are isolated
        _invalid_observation(provider, exc)
        return None


def _normalize_observations(
    provider: Any,
    raw_observations: Any,
    *,
    remaining_count: int,
    remaining_bytes: int,
    traversal_budget: Optional[list[int]],
    inspected_budget: Optional[list[int]],
) -> tuple[tuple[MemoryObservation, ...], tuple[int, ...], Optional[str]]:
    observations: list[MemoryObservation] = []
    sizes: list[int] = []
    observation_bytes = 0
    if raw_observations is None:
        raw_observations = ()
    try:
        # Use builtin base iterators; never invoke an arbitrary provider iterator on the turn thread.
        if isinstance(raw_observations, tuple):
            iterator = tuple.__iter__(raw_observations)
        elif isinstance(raw_observations, list):
            iterator = list.__iter__(raw_observations)
        else:
            raise TypeError("observations must be a list or tuple")
    except Exception:  # health: allow BLE001 -- provider-controlled iterables may raise arbitrary exceptions
        logger.warning(
            "Memory provider '%s' returned an unreadable observation container; dropping it",
            provider.name,
            exc_info=True,
        )
        iterator = iter(())

    truncated_reason = None
    while True:
        if inspected_budget is not None and inspected_budget[0] <= 0:
            truncated_reason = "inspected"
            break
        at_limit = (
            len(observations) >= remaining_count
            or observation_bytes >= remaining_bytes
        )
        if inspected_budget is not None:
            inspected_budget[0] -= 1
        try:
            candidate = next(iterator)
        except StopIteration:
            break
        except Exception:  # health: allow BLE001 -- provider-controlled iterators may raise arbitrary exceptions
            logger.warning(
                "Memory provider '%s' returned an unreadable observation container; "
                "dropping its remaining observations",
                provider.name,
                exc_info=True,
            )
            break
        if at_limit:
            truncated_reason = (
                "count" if len(observations) >= remaining_count else "bytes"
            )
            break

        normalized = _normalize_observation(provider, candidate, traversal_budget)
        if normalized is None:
            continue
        observation, encoded_size = normalized
        if observation_bytes + encoded_size > remaining_bytes:
            truncated_reason = "bytes"
            break
        observations.append(observation)
        sizes.append(encoded_size)
        observation_bytes += encoded_size

    return tuple(observations), tuple(sizes), truncated_reason


def normalize_prefetch_result(
    provider: Any,
    raw_result: Any,
    *,
    remaining_count: int = MAX_MEMORY_OBSERVATIONS,
    remaining_bytes: int,
    inspect_observations: bool = True,
    traversal_budget: Optional[list[int]] = None,
    inspected_budget: Optional[list[int]] = None,
) -> _NormalizedPrefetchResult:
    """Validate one provider result within the operation's remaining budgets."""
    if raw_result is None:
        raw_result = ""
    raw_result = coerce_prefetch_result(raw_result)
    if isinstance(raw_result, str):
        return _NormalizedPrefetchResult(
            MemoryPrefetchResult(context=str.__str__(raw_result)), ()
        )
    if type(raw_result) is not MemoryPrefetchResult:
        raise TypeError(
            f"Memory provider '{provider.name}' prefetch() must return str "
            "or MemoryPrefetchResult"
        )
    if not isinstance(raw_result.context, str):
        raise TypeError(
            f"Memory provider '{provider.name}' returned non-string prefetch context"
        )
    context = str.__str__(raw_result.context)
    if not inspect_observations:
        return _NormalizedPrefetchResult(
            MemoryPrefetchResult(context=context), ()
        )

    observations, sizes, truncated_reason = _normalize_observations(
        provider,
        raw_result.observations,
        remaining_count=remaining_count,
        remaining_bytes=remaining_bytes,
        traversal_budget=traversal_budget,
        inspected_budget=inspected_budget,
    )
    return _NormalizedPrefetchResult(
        MemoryPrefetchResult(
            context=context,
            observations=observations,
        ),
        sizes,
        truncated_reason,
    )
