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


def test_hosted_malformed_payload_spends_shared_traversal_budget(monkeypatch):
    import agent.memory_manager as manager_module
    import agent.memory_provider as provider_module
    from hermes_cli.plugin_host_child import HostRuntime
    from hermes_cli.plugin_host_wire import decode

    monkeypatch.setattr(manager_module, "MAX_MEMORY_OBSERVATION_OPERATION_NODES", 9)
    monkeypatch.setattr(provider_module, "MAX_MEMORY_OBSERVATION_OPERATION_NODES", 9)
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    raw = MemoryPrefetchResult(observations=(
        _observation([0, 0, 0, 0, object()]),
        _observation({"kept": True}),
    ))

    def collect(first):
        manager = MemoryManager()
        manager.add_provider(first)
        manager.add_provider(StructuredMemoryProvider(name="external", result=MemoryPrefetchResult(
            observations=(_observation({"later": True}),)
        )))
        return manager.prefetch_all_result("question")

    direct = collect(StructuredMemoryProvider(name="builtin", result=raw))
    runtime = HostRuntime.__new__(HostRuntime)
    runtime.refs = {1: StructuredMemoryProvider(name="builtin", result=raw)}
    runtime.owners = {1: "builtin"}
    from hermes_cli.plugin_host import PluginHost
    host = PluginHost.__new__(PluginHost)
    host._generation = 1
    host._release = lambda *args: None
    host.obj_getattr = lambda *args: "builtin"
    host.obj_invoke = lambda slot, method, args, kwargs, returns: decode(runtime.op_obj_invoke({
        "ref": 1, "method": method, "args": list(args),
    }))
    proxy = host._object_proxy({"__object__": 1, "live": ["name"],
                                "methods": {"prefetch": {}}}, StructuredMemoryProvider)
    object.__setattr__(proxy, "_tools", [])
    hosted = collect(proxy)
    assert [item.provider for item in direct.observations] == ["builtin"]
    assert hosted == direct

    from agent.memory_manager_prefetch import normalize_prefetch_result
    from hermes_cli.plugin_host_wire import _MalformedMemoryPayload
    for cost in (-100, True, 11, "1"):
        budget = [9]
        normalize_prefetch_result(proxy, MemoryPrefetchResult(
            observations=(_observation(_MalformedMemoryPayload(cost)),)),
            remaining_bytes=provider_module.MAX_MEMORY_OBSERVATION_BATCH_BYTES,
            traversal_budget=budget)
        assert budget == [8]


def test_in_process_provider_cannot_spoof_traversal_cost(monkeypatch):
    import agent.memory_manager as manager_module
    import agent.memory_provider as provider_module
    from hermes_cli.plugin_host_wire import _MalformedMemoryPayload

    monkeypatch.setattr(manager_module, "MAX_MEMORY_OBSERVATION_OPERATION_NODES", 4)
    monkeypatch.setattr(provider_module, "MAX_MEMORY_OBSERVATION_OPERATION_NODES", 4)
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    class SpoofProvider(StructuredMemoryProvider):
        __hermes_plugin_host_ref__ = True

    manager = MemoryManager()
    manager.add_provider(SpoofProvider(name="builtin", result=MemoryPrefetchResult(
        observations=(_observation(_MalformedMemoryPayload(-100)),
                      _observation([0, 0, 0, 0, 0])),
    )))
    assert manager.prefetch_all_result("question").observations == ()

    from agent.memory_manager_prefetch import normalize_prefetch_result
    budget = [4]
    normalize_prefetch_result(SpoofProvider(name="builtin", result=""),
        MemoryPrefetchResult(observations=(_observation(_MalformedMemoryPayload(4)),)),
        remaining_bytes=provider_module.MAX_MEMORY_OBSERVATION_BATCH_BYTES,
        traversal_budget=budget)
    assert budget == [3]


def test_forged_slot_cannot_spend_transport_cost():
    from agent.memory_manager_prefetch import normalize_prefetch_result
    from agent.memory_provider import MAX_MEMORY_OBSERVATION_BATCH_BYTES
    from hermes_cli.plugin_host import PluginHost, _ObjectSlot
    from hermes_cli.plugin_host_wire import _MalformedMemoryPayload

    host = PluginHost.__new__(PluginHost)
    host._generation = 1
    host._release = lambda *args: None
    slot = _ObjectSlot(host, {"__object__": 1}, None)
    class ForgedProvider(StructuredMemoryProvider):
        __hermes_plugin_host_ref__ = slot

    provider = ForgedProvider(name="fixture", result="")
    slot.track(provider)
    budget = [10]
    normalize_prefetch_result(provider, MemoryPrefetchResult(
        observations=(_observation(_MalformedMemoryPayload(9)),)),
        remaining_bytes=MAX_MEMORY_OBSERVATION_BATCH_BYTES, traversal_budget=budget)
    assert budget == [9]


def test_wire_cost_is_transport_owned_and_old_records_remain_readable(monkeypatch):
    import agent.memory_provider as provider_module
    from agent.memory_manager_prefetch import normalize_prefetch_result
    from hermes_cli.plugin_host_wire import decode, encode

    provider = StructuredMemoryProvider(name="fixture", result="")
    forged = {
        "source_kind": "recall", "schema": "fixture", "version": 1,
        "payload": {"ok": True}, "malformed_traversal_cost": 999999,
    }
    raw = MemoryPrefetchResult(observations=(forged,))
    wire = encode(raw)
    assert wire["fields"]["observations"][0]["malformed_traversal_cost"] is None
    # Old peers omit the accounting field entirely.
    del wire["fields"]["observations"][0]["malformed_traversal_cost"]
    budget = [10]
    result = normalize_prefetch_result(provider, decode(wire),
        remaining_bytes=provider_module.MAX_MEMORY_OBSERVATION_BATCH_BYTES,
        traversal_budget=budget)
    assert len(result.result.observations) == 1
    assert budget == [8]
