"""Pinned parent identity, admission receipts, and stale internal-event filtering."""
import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.platforms.event import MessageEvent
from tests.gateway.test_plugin_message_injection import _entry, _runner


def accepted_adapter():
    async def accept(event):
        event._gateway_accepted = True
    return SimpleNamespace(handle_message=AsyncMock(side_effect=accept))


@pytest.mark.asyncio
async def test_rejected_admission_is_not_reported_as_delivered():
    entry = _entry()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter)
    assert not await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="checkpoint", plugin_id="pilot")


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_original_parent_is_preserved_across_async_dispatch(compressed):
    entry = _entry()
    adapter = accepted_adapter()
    runner = _runner(entry, adapter)
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={
        "ended_at": 1, "end_reason": "compression" if compressed else "user_reset"}))
    runner._resolve_compression_lineage_target = AsyncMock(return_value=entry.session_id)
    result = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="checkpoint", plugin_id="pilot",
        expected_session_id="original-parent")
    assert result is compressed
    if compressed:
        event = adapter.handle_message.await_args.args[0]
        assert event.metadata["gateway_session_id"] == "original-parent"
        assert event.metadata["gateway_session_compression"] is True
    else:
        adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_queued_plugin_event_rechecks_original_parent_before_turn(compressed, monkeypatch):
    entry = _entry()
    runner = _runner(entry)
    runner._session_key_for_source = lambda source: entry.session_key
    runner._cache_session_source = lambda *a: None
    runner._is_telegram_topic_lane = lambda source: False
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={
        "ended_at": 1, "end_reason": "compression" if compressed else "user_reset"}))
    runner._resolve_compression_lineage_target = AsyncMock(return_value=entry.session_id)
    monkeypatch.setattr("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", AsyncMock(return_value=True))
    event = MessageEvent(text="checkpoint", source=entry.origin, internal=True,
        allow_gateway_control=False, metadata={"hermes_plugin_injection": True,
            "gateway_session_key": entry.session_key, "gateway_session_id": "original-parent",
            "gateway_session_strict": True, "gateway_session_compression": True})
    assert (await runner._hmwa_resolve_session(event, event.source) is not None) is compressed


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_receipt_callback_observes_admission_not_scheduling(accepted):
    entry = _entry()
    runner = _runner(entry)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=accepted)
    receipts = []
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="checkpoint", plugin_id="pilot",
        expected_session_id=entry.session_id, on_delivery=receipts.append)
    await asyncio.gather(*runner._background_tasks)
    await asyncio.sleep(0)
    assert receipts == [accepted]
    runner._dispatch_plugin_message_injection.assert_awaited_once_with(
        session_key=entry.session_key, content="checkpoint", plugin_id="pilot",
        expected_session_id=entry.session_id, _admission_receipt=ANY)


@pytest.mark.asyncio
async def test_callback_failure_does_not_break_gateway_loop():
    entry = _entry()
    runner = _runner(entry)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(side_effect=RuntimeError("transport failed"))
    receipts = []
    def callback(accepted):
        receipts.append(accepted)
        raise OSError("receipt store failed")
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="checkpoint", plugin_id="pilot", on_delivery=callback)
    await asyncio.gather(*runner._background_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert receipts == [False]


@pytest.mark.asyncio
async def test_only_plugin_internal_events_run_stale_filter(monkeypatch):
    from tests.gateway.test_pre_gateway_dispatch import _make_event, _make_runner
    from gateway.config import Platform
    runner, _ = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = AsyncMock(return_value="should not start")
    seen = []
    async def filter_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"])
            return [{"action": "skip", "reason": "checkpoint already reviewed"}]
        return []
    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", filter_hook)
    event = _make_event("checkpoint")
    event.internal = True
    event.allow_gateway_control = False
    event.metadata = {"hermes_plugin_injection": True, "hermes_plugin_id": "pilot"}
    assert await runner._handle_message(event) is None
    assert seen == [event]
    runner._handle_message_with_agent.assert_not_awaited()

@pytest.mark.asyncio
async def test_sdk_receipt_covers_real_busy_queue_and_owner_pin(tmp_path, monkeypatch):
    from tests.gateway.test_plugin_message_injection import _RoutingAdapter
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(_entry().origin)
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    runner = _runner(entry, adapter)
    runner.session_store = store
    runner._async_session_store = None
    runner._gateway_loop = asyncio.get_running_loop()
    runner._queued_events = {}
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="pilot", key="pilot", source="user"), manager)
    monkeypatch.setattr(context, "_gateway_injection_allowed", lambda: True)
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    receipts = []
    for index in range(33):
        assert context.inject_message(f"checkpoint-{index}", role="tool", session_key=entry.session_key,
            expected_session_id=entry.session_id, on_delivery=receipts.append)
        await asyncio.gather(*runner._background_tasks)
        await asyncio.sleep(0)
    assert receipts == [True] * 32 + [False]
    assert runner._queue_depth(entry.session_key, adapter=adapter) == 32
    assert adapter._pending_messages[entry.session_key].metadata["gateway_session_id"] == entry.session_id
    adapter._message_handler.assert_not_awaited()
    monkeypatch.setattr(context, "_gateway_injection_allowed", lambda: False)
    assert not context.inject_message("denied", session_key=entry.session_key,
        expected_session_id=entry.session_id, on_delivery=receipts.append)
    assert len(receipts) == 33
