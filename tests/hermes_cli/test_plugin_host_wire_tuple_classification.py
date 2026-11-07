"""Mixed tuples must not be mistaken for memory-observation batches."""

import json

from agent.memory_provider import MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES, MemoryObservation
from hermes_cli.plugin_host_wire import decode, encode


def test_mixed_tuple_preserves_non_observation_values():
    observation = MemoryObservation("recall", "test", 1, {"ok": True})
    result = decode(json.loads(json.dumps(encode((observation, "kept", {"other": 2})))))

    assert result[0] == observation
    assert result[1:] == ["kept", {"other": 2}]


def test_oversized_mixed_tuple_is_generic_without_custom_iteration():
    observation = MemoryObservation("recall", "test", 1, {"ok": True})

    class GuardedTuple(tuple):
        def __iter__(self):
            raise AssertionError("custom iteration must not run")

    values = GuardedTuple((observation,) * (MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1) + ("kept",))
    result = decode(json.loads(json.dumps(encode(values))))
    assert result == [observation] * (MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1) + ["kept"]


def test_oversized_observation_tuple_is_generic_without_custom_iteration():
    observation = MemoryObservation("recall", "test", 1, {"ok": True})

    class GuardedTuple(tuple):
        def __iter__(self):
            raise AssertionError("custom iteration must not run")

    values = GuardedTuple((observation,) * (MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 2))
    assert decode(json.loads(json.dumps(encode(values)))) == list(tuple.__iter__(values))
