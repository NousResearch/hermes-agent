"""Per-consumer memory observation isolation."""

import threading

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryPrefetchResult
from tests.agent.test_memory_observation import StructuredMemoryProvider, _observation


def test_observer_base_dict_mutation_is_isolated(monkeypatch):
    from agent import plugin_stream_hooks as dispatcher
    from tests.agent.test_plugin_stream_hooks import _patch_test_callback_manager

    dispatcher.shutdown_plugin_observer_dispatcher()
    mutated = threading.Event()
    observed = threading.Event()
    seen = []

    def first(**payload):
        dict.__setitem__(payload["observations"][0].payload["nested"][0], "value", "changed")
        mutated.set()

    def second(**payload):
        assert mutated.wait(timeout=5)
        seen.append(payload["observations"][0].payload["nested"][0]["value"])
        observed.set()

    _patch_test_callback_manager(monkeypatch, {"memory_prefetch": [first, second]})
    manager = MemoryManager()
    manager.add_provider(StructuredMemoryProvider(result=MemoryPrefetchResult(
        context="context", observations=(_observation({"nested": [{"value": "original"}]}),),
    )))
    try:
        result = manager.prefetch_all_result("question")
        assert observed.wait(timeout=5)
        assert seen == ["original"]
        assert result.observations[0].payload["nested"][0]["value"] == "original"
    finally:
        dispatcher.shutdown_plugin_observer_dispatcher(timeout=5)
