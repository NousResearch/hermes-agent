"""Hosted and in-process observation budget parity."""

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryObservation, MemoryPrefetchResult
from tests.agent.test_memory_observation import (
    StructuredMemoryProvider, _disable_hook, _stub_direct_prefetch, _observation,
)


def test_malformed_prefix_keeps_valid_and_later_provider_observations(monkeypatch):
    from hermes_cli.plugin_host_wire import decode, encode

    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    raw = MemoryPrefetchResult(
        context="first",
        observations=tuple(
            MemoryObservation("", "fixture", 1, {"text": "x" * 3910})
            for _ in range(55)
        ) + (_observation({"first": True}),),
    )
    def collect(first):
        manager = MemoryManager()
        manager.add_provider(StructuredMemoryProvider(name="builtin", result=first))
        manager.add_provider(StructuredMemoryProvider(name="external", result=MemoryPrefetchResult(
            context="later", observations=(_observation({"later": True}),)
        )))
        return manager.prefetch_all_result("question")

    direct = collect(raw)
    hosted = collect(decode(encode(raw)))
    assert hosted == direct
    assert [item.provider for item in hosted.observations] == ["builtin", "external"]


def test_host_prefetch_rejects_wrong_provider_before_payload_freezing(monkeypatch):
    from agent import memory_provider
    from agent.memory_manager_prefetch import normalize_prefetch_result
    from hermes_cli.plugin_host_child import HostRuntime
    from hermes_cli.plugin_host_wire import decode

    class Version(int):
        pass

    rejected_payload = {"text": "x" * 3910}
    raw = MemoryPrefetchResult(observations=tuple(
        MemoryObservation("recall", "fixture", 1, rejected_payload, provider="other")
        for _ in range(55)
    ) + (MemoryObservation("recall", "fixture", Version(1), {"kept": True}),))
    provider = StructuredMemoryProvider(name="fixture", result=raw)
    freeze = memory_provider._freeze_memory_observation_payload

    def checked_freeze(payload, *args, **kwargs):
        assert payload is not rejected_payload
        return freeze(payload, *args, **kwargs)

    monkeypatch.setattr(memory_provider, "_freeze_memory_observation_payload", checked_freeze)
    runtime = HostRuntime.__new__(HostRuntime)
    runtime.refs = {1: provider}
    runtime.owners = {1: "fixture"}
    direct = normalize_prefetch_result(provider, raw, remaining_bytes=memory_provider.MAX_MEMORY_OBSERVATION_BATCH_BYTES)
    hosted = normalize_prefetch_result(provider, decode(runtime.op_obj_invoke({
        "ref": 1, "method": "prefetch", "args": ["question"],
    })), remaining_bytes=memory_provider.MAX_MEMORY_OBSERVATION_BATCH_BYTES)
    assert hosted.result == direct.result
    assert len(hosted.result.observations) == 1
    assert hosted.result.observations[0].version == 1


def test_invalid_host_provider_name_does_not_exhaust_wire_budget():
    from agent.memory_provider import MAX_MEMORY_OBSERVATION_FIELD_CHARS
    from hermes_cli.plugin_host_wire import decode
    from hermes_cli.plugin_host_wire_memory import encode_memory_prefetch_result

    raw = MemoryPrefetchResult(observations=tuple(
        MemoryObservation("recall", "fixture", 1, {"text": "x" * 3910})
        for _ in range(55)
    ))
    for name in (None, "", "x" * (MAX_MEMORY_OBSERVATION_FIELD_CHARS + 1)):
        result = decode(encode_memory_prefetch_result(raw, None, 0, name))
        assert len(result.observations) == 55
        assert not result._wire_truncated
