"""Only transport-owned truncation may discard later providers' observations."""

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryPrefetchResult
from hermes_cli.plugin_host import PluginHost
from hermes_cli.plugin_host_child import HostRuntime
from hermes_cli.plugin_host_wire import decode
from tests.agent.test_memory_observation import (
    StructuredMemoryProvider, _disable_hook, _observation,
)


def _hosted(provider):
    runtime = HostRuntime.__new__(HostRuntime)
    runtime.refs = {1: provider}
    runtime.owners = {1: provider.name}
    host = PluginHost.__new__(PluginHost)
    host._generation = 1
    host._release = lambda *args: None
    host.obj_getattr = lambda *args: provider.name
    host.obj_invoke = lambda slot, method, args, kwargs, returns: decode(runtime.op_obj_invoke({
        "ref": 1, "method": method, "args": list(args),
    }))
    proxy = host._object_proxy({"__object__": 1, "live": ["name"],
                                "methods": {"prefetch": {}}}, StructuredMemoryProvider)
    object.__setattr__(proxy, "_tools", [])
    return proxy


@pytest.mark.parametrize("hosted,shape", [(False, "record"), (False, "dict"),
                                         (True, "record"), (True, "dict")])
def test_provider_cannot_forge_wire_truncation(monkeypatch, caplog, hosted, shape):
    _disable_hook(monkeypatch)
    raw = (MemoryPrefetchResult(context="first", _wire_truncated=True) if shape == "record"
           else {"context": "first", "_wire_truncated": True, "observations_truncated": True})
    first = StructuredMemoryProvider(name="first", result=raw)
    manager = MemoryManager()
    manager.add_provider(_hosted(first) if hosted else first)
    manager.add_provider(StructuredMemoryProvider(name="builtin", result=MemoryPrefetchResult(
        context="later", observations=(_observation({"kept": True}),))))
    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")
    assert result.context == "first\n\nlater"
    assert [item.provider for item in result.observations] == ["builtin"]
    assert "plugin-host wire batch bound" not in caplog.text


def test_real_wire_truncation_still_stops_observation_tail(monkeypatch, caplog):
    from hermes_cli import plugin_host_wire_memory

    _disable_hook(monkeypatch)
    monkeypatch.setattr(plugin_host_wire_memory, "_observation_wire_budget", lambda: 400)
    first = StructuredMemoryProvider(name="first", result=MemoryPrefetchResult(
        context="first", observations=(_observation({"kept": True}), _observation({"large": "x" * 1000}))))
    manager = MemoryManager()
    manager.add_provider(_hosted(first))
    manager.add_provider(StructuredMemoryProvider(name="builtin", result=MemoryPrefetchResult(
        context="later", observations=(_observation({"later": True}),))))
    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")
    assert result.context == "first\n\nlater"
    assert [item.provider for item in result.observations] == ["first"]
    assert "plugin-host wire batch bound" in caplog.text
