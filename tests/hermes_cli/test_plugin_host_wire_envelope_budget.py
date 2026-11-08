"""Invalid observation envelopes must not crowd valid ones off the host wire."""

from agent.memory_manager_prefetch import normalize_prefetch_result
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_BATCH_BYTES, MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
    MemoryObservation, MemoryPrefetchResult,
)
from hermes_cli.plugin_host_wire import decode
from hermes_cli.plugin_host_wire_memory import encode_memory_prefetch_result
from tests.agent.test_memory_provider import FakeMemoryProvider


def test_oversized_envelopes_do_not_hide_later_valid_observation():
    provider = FakeMemoryProvider(name="fixture")
    invalid = MemoryObservation("recall", "fixture", 1, ["x" * 4090] * 4)
    raw = MemoryPrefetchResult(observations=(invalid,) * 55 + (
        MemoryObservation("recall", "fixture", 1, {"kept": True}),
    ))
    direct = normalize_prefetch_result(provider, raw, remaining_bytes=MAX_MEMORY_OBSERVATION_BATCH_BYTES)
    hosted = normalize_prefetch_result(
        provider, decode(encode_memory_prefetch_result(raw, None, 0, provider.name)),
        remaining_bytes=MAX_MEMORY_OBSERVATION_BATCH_BYTES,
    )
    assert len(direct.result.observations) == 1
    assert hosted == direct


def test_list_subclass_uses_builtin_access_and_same_wire_budgets():
    class TrappedList(list):
        def __len__(self):
            raise AssertionError("overridden len called")

        def __getitem__(self, index):
            raise AssertionError("overridden getitem called")

        def __iter__(self):
            raise AssertionError("overridden iter called")

    valid = MemoryObservation("recall", "fixture", 1, {"kept": True})
    malformed = MemoryObservation("recall", "fixture", 1, [object()])
    items = [malformed, valid] + [valid] * MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES
    expected = encode_memory_prefetch_result({"context": "ctx", "observations": items}, None, 0, "fixture")
    actual = encode_memory_prefetch_result({"context": "ctx", "observations": TrappedList(items)}, None, 0, "fixture")
    assert actual == expected
    assert len(actual["fields"]["observations"]) == MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1
    assert actual["fields"]["observations_truncated"]
    assert decode(actual).observations[1].payload == {"kept": True}
