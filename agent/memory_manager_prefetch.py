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


def coerce_prefetch_result(raw_result: Any) -> Any:
    """Restore structured result fields after the plugin-host wire codec."""
    if isinstance(raw_result, Mapping) and "context" in raw_result:
        return MemoryPrefetchResult(
            context=raw_result["context"],
            observations=raw_result.get("observations", ()),
        )
    return raw_result


def _invalid_observation(provider: Any, exc: Exception) -> None:
    logger.warning(
        "Memory provider '%s' returned a malformed prefetch observation; dropping it: %s",
        provider.name,
        exc,
    )


def _normalize_observation(
    provider: Any, candidate: Any, traversal_budget: Optional[List[int]]
) -> Optional[tuple[MemoryObservation, int]]:
    try:
        if isinstance(candidate, Mapping):
            # Read the fixed contract fields; never expand arbitrary plugin keys.
            raw_candidate: Any = candidate
            candidate = MemoryObservation(
                source_kind=raw_candidate.get("source_kind"),
                schema=raw_candidate.get("schema"),
                version=raw_candidate.get("version"),
                provider=raw_candidate.get("provider", ""),
                payload=raw_candidate.get("payload"),
            )
        if not isinstance(candidate, MemoryObservation):
            raise TypeError("observation has the wrong type")
        for field_name in ("source_kind", "schema"):
            field = getattr(candidate, field_name)
            if (
                not isinstance(field, str)
                or not field
                or str.__len__(field) > MAX_MEMORY_OBSERVATION_FIELD_CHARS
            ):
                raise ValueError(f"observation {field_name} is invalid")
        if (
            isinstance(candidate.version, bool)
            or not isinstance(candidate.version, int)
            or candidate.version < 1
        ):
            raise ValueError("observation version is invalid")
        if _encoded_json_scalar_size(candidate.version) > MAX_MEMORY_OBSERVATION_BYTES:
            raise ValueError("observation version is too large")
        if candidate.provider not in ("", provider.name):
            raise ValueError("observation provider does not match its source provider")
        if not isinstance(provider.name, str) or not provider.name:
            raise ValueError("provider name is invalid")
        if str.__len__(provider.name) > MAX_MEMORY_OBSERVATION_FIELD_CHARS:
            raise ValueError("provider name is too long")

        frozen_payload, _payload_bytes = _freeze_memory_observation_payload(
            candidate.payload, operation_budget=traversal_budget
        )
        encoded = json.dumps(
            {
                "source_kind": candidate.source_kind,
                "provider": provider.name,
                "schema": candidate.schema,
                "version": candidate.version,
                "payload": _thaw_json_value(frozen_payload),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_MEMORY_OBSERVATION_BYTES:
            raise ValueError("observation envelope is too large")
        return (
            MemoryObservation(
                source_kind=candidate.source_kind,
                provider=provider.name,
                schema=candidate.schema,
                version=candidate.version,
                payload=frozen_payload,
            ),
            len(encoded),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        _invalid_observation(provider, exc)
        return None


def _normalize_observations(
    provider: Any,
    raw_observations: Any,
    *,
    remaining_count: int,
    remaining_bytes: int,
    traversal_budget: Optional[List[int]],
    inspected_budget: Optional[List[int]],
) -> tuple[tuple[MemoryObservation, ...], tuple[int, ...], Optional[str]]:
    observations: list[MemoryObservation] = []
    sizes: list[int] = []
    observation_bytes = 0
    if raw_observations is None:
        raw_observations = ()
    try:
        iterator = iter(raw_observations)
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
    traversal_budget: Optional[List[int]] = None,
    inspected_budget: Optional[List[int]] = None,
) -> _NormalizedPrefetchResult:
    """Validate one provider result within the operation's remaining budgets."""
    if raw_result is None:
        raw_result = ""
    raw_result = coerce_prefetch_result(raw_result)
    if isinstance(raw_result, str):
        return _NormalizedPrefetchResult(MemoryPrefetchResult(context=raw_result), ())
    if not isinstance(raw_result, MemoryPrefetchResult):
        raise TypeError(
            f"Memory provider '{provider.name}' prefetch() must return str "
            "or MemoryPrefetchResult"
        )
    if not isinstance(raw_result.context, str):
        raise TypeError(
            f"Memory provider '{provider.name}' returned non-string prefetch context"
        )
    if not inspect_observations:
        return _NormalizedPrefetchResult(
            MemoryPrefetchResult(context=raw_result.context), ()
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
            context=raw_result.context,
            observations=observations,
        ),
        sizes,
        truncated_reason,
    )
