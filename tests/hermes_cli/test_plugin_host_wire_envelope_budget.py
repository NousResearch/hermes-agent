"""Invalid observation envelopes must not crowd valid ones off the host wire."""

from agent.memory_manager_prefetch import normalize_prefetch_result
from agent.memory_provider import MAX_MEMORY_OBSERVATION_BATCH_BYTES, MemoryObservation, MemoryPrefetchResult
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
