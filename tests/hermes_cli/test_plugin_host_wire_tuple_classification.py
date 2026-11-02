"""Mixed tuples must not be mistaken for memory-observation batches."""

import json

import pytest

from agent.memory_provider import MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES, MemoryObservation
from hermes_cli.plugin_host_wire import decode, encode


def test_mixed_tuple_preserves_non_observation_values():
    observation = MemoryObservation("recall", "test", 1, {"ok": True})
    result = decode(json.loads(json.dumps(encode((observation, "kept", {"other": 2})))))

    assert result[0] == observation
    assert result[1:] == ["kept", {"other": 2}]


@pytest.mark.parametrize("mixed", [False, True])
def test_oversized_observation_first_tuple_is_rejected_without_iteration(mixed):
    observation = MemoryObservation("recall", "test", 1, {"ok": True})
    tail = ("other" if mixed else observation,) * (MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1)

    class GuardedTuple(tuple):
        def __iter__(self):
            raise AssertionError("oversized tuple must not be traversed")

    with pytest.raises(ValueError, match="MemoryObservationTuple exceeds"):
        encode(GuardedTuple((observation, *tail)))
