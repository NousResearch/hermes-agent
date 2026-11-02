"""The inspected cap reports truncation only when candidates remain."""

from types import SimpleNamespace

import pytest

from agent.memory_manager_prefetch import normalize_prefetch_result
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_BATCH_BYTES,
    MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
    MemoryPrefetchResult,
)


@pytest.mark.parametrize("extra", [0, 1])
def test_inspected_cap_exact_boundary_and_shared_empty_provider(extra):
    cap = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES
    budget = [cap]
    provider = SimpleNamespace(name="test")

    def normalize(candidates):
        return normalize_prefetch_result(
            provider,
            MemoryPrefetchResult(observations=candidates),
            remaining_bytes=MAX_MEMORY_OBSERVATION_BATCH_BYTES,
            inspected_budget=budget,
        )

    result = normalize([None] * (cap + extra))
    assert budget == [0]
    assert result.truncated_reason == ("inspected" if extra else None)
    assert normalize([]).truncated_reason is None
    assert normalize([None]).truncated_reason == "inspected"
    assert budget == [0]
