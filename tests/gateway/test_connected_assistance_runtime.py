"""Gateway-owned automatic assistance: offline services, real planner and store."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.connected_assistance_store import AssistanceStore
from tests.tools.test_connected_source_capabilities import SourceTransport
from tests.agent.test_connected_assistance import completion


def runner_for(home):
    runner = SimpleNamespace(_running=True, _draining=False)
    authority = SimpleNamespace(profile_id=str(home), runner=runner, sessions={}, retiring=False)
    runner.session_authority = authority
    runner.config = SimpleNamespace(multiplex_profiles=False)
    return runner, authority


def wire_services(monkeypatch):
    from agent import auxiliary_client
    from tools.connectors.gateway import bridge, client, config
    from tools.connectors.portal import client as portal

    service = SourceTransport()
    monkeypatch.setattr(bridge, "guest_identity_pending", lambda: False)
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    for module in (client, portal):
        monkeypatch.setattr(module, "_default_transport", lambda: service)
        monkeypatch.setattr(module, "_default_endpoint_resolver", lambda: "https://fixture.test")
        monkeypatch.setattr(module, "_default_header_provider", lambda _: {"Authorization": "Bearer fixture"})
    calls = []

    def model(**kw):
        calls.append(kw)
        payload = json.loads(kw["messages"][1]["content"])
        if "capabilities" in payload:
            cap = payload["capabilities"][0]
            return completion({"reads": [{"capability_id": cap["capability_id"], "arguments": {"query": "to:me"}}]})
        record = payload["records"][0]
        return completion({"findings": [{"kind": "direct_request", "entity_key": "Atlas", "title": "Atlas review",
            "summary": "A review was requested", "why_now": "Please review Atlas today", "suggested_action": "review",
            "evidence": [{k: record[k] for k in ("source_id", "record_id", "json_pointer")} |
                         {"excerpt": "Please review Atlas today", "url": "https://mail.test/m1"}]}]})

    monkeypatch.setattr(auxiliary_client, "call_llm", model)
    return service, calls


@pytest.mark.asyncio
async def test_real_source_to_private_event_stream_smoke(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance
    from tests.gateway.test_connected_assistance_delivery import local_owner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _service, calls = wire_services(monkeypatch)
    runner, authority, connection, ref, live = await local_owner(tmp_path, monkeypatch)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    frames = []
    live.event_stream.observers.add(frames.append)
    try:
        await ConnectedAssistance(runner).tick()
        assert len(AssistanceStore(tmp_path).list_findings()) == 1
        assert len(frames) == 1 and frames[0]["params"]["type"] == "notification.show"
        assert frames[0]["params"]["session_id"] == ref.session_id
        assert "hermes connectors findings" in frames[0]["params"]["payload"]["text"]
        assert len(calls) == 2
        print("FIXTURE PRIVATE EVENT:", frames[0]["params"])
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_startup_registers_real_supervised_assistance(monkeypatch):
    from gateway.run_startup import GatewayStartupMixin

    class Startup(GatewayStartupMixin):
        _PRE_RECONNECT_WATCHERS = ()
        _POST_RECONNECT_WATCHERS = ()
        _failed_platforms = {}
        _running = True
        _draining = False

        def __init__(self):
            self.spawns = {}

        def _spawn_supervised(self, factory, name):
            self.spawns[name] = factory

        def _spawn_reconnect_watcher(self):
            return None

        def _scale_to_zero_should_arm(self):
            return False

        def _log_scale_to_zero_not_armed_reason(self):
            return None

        async def _drain_control_watcher(self):
            return None

    runner = Startup()
    runner._start_spawn_background_watchers()
    assert runner.spawns["connected_assistance"].__self__ is runner._connected_assistance


@pytest.mark.asyncio
async def test_stop_retains_blocking_worker_and_prevents_late_publication(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, _calls = wire_services(monkeypatch)
    runner, _authority = runner_for(tmp_path)
    runtime = ConnectedAssistance(runner)
    entered, release = threading.Event(), threading.Event()
    from agent import auxiliary_client
    actual = auxiliary_client.call_llm

    def blocking(**kwargs):
        entered.set()
        assert release.wait(10)
        return actual(**kwargs)

    monkeypatch.setattr(auxiliary_client, "call_llm", blocking)
    waiter = asyncio.create_task(runtime.tick())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        worker = runtime.worker
        assert not await runtime.stop(timeout=0)
        await runtime.tick()
        assert runtime.worker is worker and not worker.done()
    finally:
        release.set()
        if runtime.worker is not None:
            await asyncio.gather(runtime.worker, return_exceptions=True)
    assert AssistanceStore(tmp_path).list_findings() == []
    assert not any(u.endswith("/execute") for _, u, _ in service.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["shutdown", "profile_remove"])
async def test_lifecycle_fences_assistance_before_releasing_runtime(tmp_path, monkeypatch, operation):
    from gateway.connected_assistance import ConnectedAssistance
    from gateway import run_runtime
    from gateway.run_shutdown import GatewayShutdownMixin
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    runner, authority = runner_for(tmp_path)
    authority.db = SimpleNamespace(db_path=str(tmp_path / "state.db"))
    runner._connected_assistance = runtime = ConnectedAssistance(runner)
    runner._restart_requested = False
    runner._clear_plugin_message_injector = lambda: None
    runner.session_authorities = SimpleNamespace(for_home=lambda _: authority, serving=lambda: [authority])

    async def release_boundary(*args):
        assert not runtime.current(authority)
        raise RuntimeError("release boundary reached")

    with pytest.raises(RuntimeError, match="release boundary reached"):
        if operation == "shutdown":
            monkeypatch.setattr(run_runtime, "drain_gateway_runtime", release_boundary)
            await GatewayShutdownMixin._stop_begin_teardown(runner, SimpleNamespace())
            assert runtime.stopped.is_set()
        else:
            monkeypatch.setattr(run_runtime, "unserve_profile_runtime", release_boundary)
            await GatewayProfileReconcileMixin._unserve_profile(runner, "fixture", tmp_path)
    assert runtime.stopped.is_set() if operation == "shutdown" else runtime.retired.get(authority.profile_id) is authority


@pytest.mark.asyncio
async def test_cancelled_scan_cannot_persist_when_its_thread_finishes(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance
    from agent import auxiliary_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _service, calls = wire_services(monkeypatch)
    runner, _authority = runner_for(tmp_path)
    runtime = ConnectedAssistance(runner)
    entered, release = threading.Event(), threading.Event()
    actual = auxiliary_client.call_llm

    def blocked(**kwargs):
        entered.set()
        assert release.wait(10)
        return actual(**kwargs)

    monkeypatch.setattr(auxiliary_client, "call_llm", blocked)
    waiter = asyncio.create_task(runtime.tick())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        release.set()
        await asyncio.gather(runtime.worker, return_exceptions=True)
    assert AssistanceStore(tmp_path).list_findings() == []
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["pause", "retire", "disconnect", "disable"])
async def test_policy_change_during_judgment_discards_result(tmp_path, monkeypatch, change):
    from gateway.connected_assistance import ConnectedAssistance
    from agent import auxiliary_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, calls = wire_services(monkeypatch)
    runner, authority = runner_for(tmp_path)
    actual = auxiliary_client.call_llm

    def revoke(**kwargs):
        result = actual(**kwargs)
        if len(calls) == 2:
            if change == "pause":
                AssistanceStore(tmp_path).set_paused(True)
            elif change == "retire":
                authority.retiring = True
            elif change == "disconnect":
                service.connected = False
            else:
                (tmp_path / "config.yaml").write_text("gateway:\n  connected_assistance: false\n")
        return result

    monkeypatch.setattr(auxiliary_client, "call_llm", revoke)
    await ConnectedAssistance(runner).tick()
    assert AssistanceStore(tmp_path).list_findings() == []
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["pause", "disabled", "disconnected"])
async def test_no_reads_or_models_without_eligible_unpaused_connection(tmp_path, monkeypatch, reason):
    from gateway.connected_assistance import ConnectedAssistance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, calls = wire_services(monkeypatch)
    if reason == "pause":
        AssistanceStore(tmp_path).set_paused(True)
    elif reason == "disabled":
        (tmp_path / "config.yaml").write_text("gateway:\n  connected_assistance: false\n")
    else:
        service.connected = False
    runner, _ = runner_for(tmp_path)
    await ConnectedAssistance(runner).tick()
    assert not calls
    assert not any(u.endswith("/execute") for _, u, _ in service.requests)
    if reason != "disconnected":
        assert not service.requests


@pytest.mark.asyncio
async def test_multiplex_reads_model_and_state_belong_to_owning_profile(tmp_path, monkeypatch):
    from pathlib import Path
    from gateway.connected_assistance import ConnectedAssistance
    from gateway.session_authorities import SessionAuthorities
    from hermes_constants import get_hermes_home
    from agent import auxiliary_client

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    secondary = default / "profiles" / "work"
    secondary.mkdir(parents=True)
    (default / "config.yaml").write_text("gateway:\n  connected_assistance: false\n")
    (secondary / "config.yaml").write_text("gateway:\n  connected_assistance: true\n")
    monkeypatch.setenv("HERMES_HOME", str(default))
    service, calls = wire_services(monkeypatch)
    runner, launch = runner_for(default)
    runner.config.multiplex_profiles = True
    second = SimpleNamespace(profile_id=str(secondary), runner=runner, sessions={}, retiring=False)
    runner.session_authorities = registry = SessionAuthorities(default, multiplexed=True)
    registry.add(default, launch)
    registry.add(secondary, second, name="work")
    homes = []
    actual = auxiliary_client.call_llm

    def scoped_model(**kw):
        homes.append(get_hermes_home())
        return actual(**kw)

    monkeypatch.setattr(auxiliary_client, "call_llm", scoped_model)
    await ConnectedAssistance(runner).tick()
    assert homes == [secondary, secondary]
    assert len(calls) == 2 and service.requests
    assert not (default / "connected_assistance").exists()
    assert len(AssistanceStore(secondary).list_findings()) == 1


@pytest.mark.asyncio
async def test_cross_source_scan_cites_two_observed_accounts(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance
    from tests.tools.test_connected_source_capabilities import Reply
    from agent import auxiliary_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, _calls = wire_services(monkeypatch)
    original = service.request

    def two_accounts(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if url.endswith("/accounts"):
            first = response.payload["accounts"][0]
            return Reply({"accounts": [first, first | {"connectionId": "account-2", "alias": "personal"}]})
        if url.endswith("/execute"):
            alias = kwargs["json"]["call"]["arguments"]["connector_alias"]
            return Reply({"result": response.payload["result"] | {"account": alias,
                          "data": service.data | {"id": alias}}})
        return response

    monkeypatch.setattr(service, "request", two_accounts)

    def model(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        if "capabilities" in payload:
            return completion({"reads": [{"capability_id": c["capability_id"], "arguments": {"query": "Atlas"}}
                                          for c in payload["capabilities"]]})
        return completion({"findings": [{"kind": "cross_source", "entity_key": "Atlas", "title": "Atlas review",
            "summary": "Related requests in two accounts", "why_now": "Review requested today", "suggested_action": "review",
            "evidence": [{k: r[k] for k in ("source_id", "record_id", "json_pointer")} |
                         {"excerpt": "Please review Atlas today"} for r in payload["records"]]}]})

    monkeypatch.setattr(auxiliary_client, "call_llm", model)
    runner, _authority = runner_for(tmp_path)
    await ConnectedAssistance(runner).tick()
    rows = AssistanceStore(tmp_path).list_findings()
    assert len(rows) == 1
    assert {e["record_id"] for e in rows[0]["evidence"]} == {"work", "personal"}
    assert len({e["source_id"] for e in rows[0]["evidence"]}) == 2
    assert len([u for _, u, _ in service.requests if u.endswith("/execute")]) == 2


@pytest.mark.asyncio
async def test_large_retained_history_does_not_disable_new_scans(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, calls = wire_services(monkeypatch)
    capability = service.sources().discover()[0]
    records = [{"source_id": capability.source_id, "capability_id": capability.capability_id,
                "record_id": str(i), "content": "retained source " * 500, "data_use": ["no_training"]}
               for i in range(200)]
    AssistanceStore(tmp_path).ingest_scan({"connected": True, "findings": [], "evidence": records})
    runner, _authority = runner_for(tmp_path)
    await ConnectedAssistance(runner).tick()
    assert len(calls) == 2
    assert len(AssistanceStore(tmp_path).list_findings()) == 1
    assert json.loads(calls[0]["messages"][1]["content"])["prior_evidence"]


@pytest.mark.asyncio
async def test_automatic_tick_discovers_and_persists_without_registered_job(tmp_path, monkeypatch):
    from gateway.connected_assistance import ConnectedAssistance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, calls = wire_services(monkeypatch)
    runner, _authority = runner_for(tmp_path)
    runtime = ConnectedAssistance(runner)
    await runtime.tick()
    rows = AssistanceStore(tmp_path).list_findings()
    assert len(rows) == 1 and rows[0]["title"] == "Atlas review"
    assert len(calls) == 2
    assert len([u for _, u, _ in service.requests if u.endswith("/execute")]) == 1
    await ConnectedAssistance(runner).tick()
    assert len(calls) == 2  # persistent cadence, including gateway restart
