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
