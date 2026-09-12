"""Final runtime closure seams; only temporary SQLite and disconnected Slack."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.integration import RouterRuntime
from gateway.work_router.models import CanonicalEvent
from plugins.platforms.slack.adapter import SlackAdapter


def runtime_at(tmp_path):
    config = RouterConfig.from_mapping({
        "enabled": True, "channel_allowlist": ["CTEST"],
        "db_path": str(tmp_path / "router.db"),
        "bot_registry": {n: f"UBOT{i}" for i, n in enumerate(DEFAULT_PROFILES)},
    }).require_ready()
    return RouterRuntime(config, sender=AsyncMock())


@pytest.mark.asyncio
async def test_runtime_recovers_only_after_ready_and_drains_late_children(tmp_path):
    runtime = runtime_at(tmp_path)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    journal = runtime.router.store.ack_journal
    event = CanonicalEvent("EvRecovery", "CTEST", "1.0", text="hello", author_user_id="UH")
    journal.prepare(event, "env")
    journal.begin_wire(event.event_id, "env")
    journal.confirm_wire(event.event_id, "env")
    journal.native(event.event_id, True)
    entered, release, child_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children = []

    async def child():
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert adapter._work_router_stopping
            runtime.router.store.inbox_count()

    async def enqueue(event):
        entered.set()
        await release.wait()
        task = asyncio.create_task(child())
        children.append(task)
        runtime.router._lounge_candidate_tasks.add(task)
        await child_started.wait()
        return True

    runtime.router.enqueue_after_ack = enqueue
    runtime.attach(adapter)
    runtime.attach(adapter)  # idempotent ownership, not a second recovery
    await asyncio.sleep(0.02)
    assert not entered.is_set()
    adapter._running = True
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        stopping = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0.02)
        assert adapter._work_router_stopping and not runtime.closed
        release.set()
        await stopping
        assert children and all(t.done() for t in children)
        assert runtime.closed
    finally:
        release.set()
        if stopping:
            await stopping
        await runtime.stop()


@pytest.mark.asyncio
async def test_ingress_timeout_keeps_fenced_store_and_worker_for_retry(tmp_path):
    runtime = runtime_at(tmp_path)
    adapters = [SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test")) for _ in range(2)]
    for adapter in adapters:
        runtime.attach(adapter)

    async def timed_out(timeout=5.0):
        assert all(a._work_router_stopping for a in adapters)
        return False

    adapters[0].drain_work_router_ingress = timed_out
    try:
        await runtime.stop()
        assert runtime.stopping and not runtime.closed
        assert runtime.adapters == set(adapters)
        assert not runtime.task.done()
        assert runtime.router.store.inbox_count() == 0
    finally:
        adapters[0].drain_work_router_ingress = AsyncMock(return_value=True)
        await runtime.stop()
    assert runtime.closed


@pytest.mark.asyncio
async def test_native_recovery_waits_for_registry_ownership(tmp_path):
    from gateway.work_router.native_sender import NativeRouterSender

    runtime = runtime_at(tmp_path)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._running = True
    registered = False
    runner = SimpleNamespace(_owning_profile=lambda a, p: (registered, None))
    runtime.sender = NativeRouterSender(runner)
    recovered = asyncio.Event()

    async def recover():
        assert runtime.sender.adapter is adapter and registered
        recovered.set()

    adapter.recover_work_router_ingress = recover
    runtime.attach(adapter)
    try:
        await asyncio.sleep(0.06)
        assert not recovered.is_set()
        registered = True
        await asyncio.wait_for(recovered.wait(), 1)
    finally:
        await runtime.stop()
    assert runtime.closed and not runtime._recovery_tasks


@pytest.mark.asyncio
async def test_named_nonmultiplex_target_stays_fail_closed(tmp_path, monkeypatch):
    """Namespace-only override cannot fix key-based native DB ownership."""
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore, _session_key_namespace
    from gateway.work_router.native_sender import NativeRouterSender
    from gateway.work_router.models import RouteAction, RouteDecision, RouterThreadState
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "Hans")
    store = object.__new__(SessionStore)
    store.config = SimpleNamespace(multiplex_profiles=False)
    runner = SimpleNamespace(config=store.config, session_store=store)
    runner._session_key_for_source = lambda s: GatewayRunner._session_key_for_source(runner, s)
    runner._adapter_for_source = lambda s: s._transport_adapter_ref()
    runner._handle_message = AsyncMock()
    sender = NativeRouterSender(runner)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    sender.bind(adapter)
    event = CanonicalEvent("EvTarget", "CTEST", "1.0", text="hello", author_user_id="UH")
    state = RouterThreadState(event.channel_id, event.thread_ts)
    decision = RouteDecision(RouteAction("dispatch", "Hans"), state)
    source = sender._source(event, "Hans")
    key = runner._session_key_for_source(source)
    assert key.startswith("agent:main:")
    # Even a key-only patch would leave native key-based DB lookup ambient.
    hypothetical = _session_key_namespace("Hans") + ":slack:CTEST:1.0"
    assert store._named_profile_for_key(hypothetical) is None
    with pytest.raises(RuntimeError, match="target session namespace unavailable"):
        await sender.execute(event=event, state=state, decision=decision, operation_id="op")
    runner._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_real_ingress_drain_timeout_does_not_cancel_admitted_work(tmp_path):
    runtime = runtime_at(tmp_path)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    runtime.attach(adapter)
    started, release = asyncio.Event(), asyncio.Event()

    async def ingress():
        started.set()
        await release.wait()
        assert adapter._work_router_stopping
        runtime.router.store.inbox_count()

    task = asyncio.create_task(ingress())
    adapter._work_router_ingress_tasks.add(task)
    task.add_done_callback(adapter._work_router_ingress_tasks.discard)
    native_drain = adapter.drain_work_router_ingress

    async def short_drain(timeout=5.0):
        return await native_drain(timeout=0.01)

    adapter.drain_work_router_ingress = short_drain
    await started.wait()
    try:
        await runtime.stop()
        assert not runtime.closed and not task.done()
        assert runtime.router.store.inbox_count() == 0
    finally:
        release.set()
        await task
        await asyncio.gather(runtime.stop(), runtime.stop())
    assert runtime.closed