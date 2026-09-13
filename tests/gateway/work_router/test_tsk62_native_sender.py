"""BLOCKER1 focused tests: real router/Slack classes, no live Slack or model calls."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import Platform, PlatformConfig
from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.models import CanonicalEvent, RouteAction, RouteDecision, RouterThreadState
from gateway.work_router.native_sender import NativeRouterSender
from gateway.work_router.service import WorkRouter
from gateway.work_router.worker import WorkRouterWorker
from plugins.platforms.slack.adapter import SlackAdapter


class RecordingClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.retry_handlers = [object()]

    async def chat_postMessage(self, **kwargs):
        self.calls.append((kwargs, list(self.retry_handlers)))
        if self.response == "timeout":
            await asyncio.Event().wait()
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def config_at(path):
    raw = {"enabled": True, "channel_allowlist": ["CTEST"],
           "db_path": str(path / "router.db"),
           "bot_registry": {n: f"UBOT{i}" for i, n in enumerate(DEFAULT_PROFILES)}}
    return RouterConfig.from_mapping(raw).require_ready(), raw


@pytest.fixture
def bridge():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._running = True
    adapter.stop_typing = AsyncMock()
    runner = SimpleNamespace()
    runner._adapter_for_source = lambda source: source._transport_adapter_ref()
    runner._owning_profile = lambda obj, platform: (obj is adapter, None)
    runner._session_key_for_source = lambda source: f"agent:{source.profile}:{source.scope_id}:{source.chat_id}:{source.thread_id}"
    runner._is_session_running = lambda key: False
    sender = NativeRouterSender(runner, timeout_seconds=0.02)
    sender.bind(adapter)
    event = CanonicalEvent("EvNative", "CTEST", "1700000000.1", text="한스야 확인해",
                           author_user_id="UHUMAN", mentioned_user_ids=("UBOT1",),
                           metadata={"slack_team_id": "TTEST"})
    state = RouterThreadState(event.channel_id, event.thread_ts)
    decision = RouteDecision(RouteAction("summary", "Demian", content="실제 본문"), state)
    return sender, adapter, event, state, decision


def client_at(adapter, response):
    client = RecordingClient(response)
    adapter._app = SimpleNamespace(client=client)
    adapter._get_client = lambda channel, *, team_id: client
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["summary", "meeting_stopped", "meeting_publish", "dispatch"])
async def test_control_meeting_and_prebuilt_dispatch_are_one_native_post(bridge, kind):
    sender, adapter, event, state, decision = bridge
    decision = replace(decision, action=replace(decision.action, kind=kind))
    client = client_at(adapter, {"ok": True, "ts": "1700000000.2"})
    sender.execute = AsyncMock(side_effect=AssertionError("prebuilt output must not run a model"))
    result = await sender(event=event, state=state, decision=decision, operation_id="op")
    assert result.success and result.message_id == "1700000000.2"
    assert len(client.calls) == 1
    kwargs, retries = client.calls[0]
    assert retries == [] and len(client.retry_handlers) == 1
    assert kwargs["channel"] == event.channel_id and kwargs["thread_ts"] == event.thread_ts
    assert kwargs["text"] == adapter.format_message(decision.action.content)
    adapter.stop_typing.assert_awaited_once()
    sender.execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, {"ok": False, "ts": "1"},
                                       {"ok": True}, {"ok": True, "ts": " "},
                                       {"ok": True, "ts": 123}, "timeout", ConnectionError("lost")])
async def test_native_failures_never_confirm_or_retry(bridge, response):
    sender, adapter, event, state, decision = bridge
    client = client_at(adapter, response)
    result = await sender(event=event, state=state, decision=decision, operation_id="op")
    assert not result.success and not result.message_id and not result.retryable
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_staff_uses_native_profile_scope_and_transport_not_handler_receipt(bridge, tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret
    from hermes_cli import profiles
    sender, adapter, event, state, decision = bridge
    staff = tmp_path / "Hans"
    staff.mkdir()
    (staff / "config.yaml").write_text("{}\n")
    (staff / ".env").write_text("TSK62_TEST_SECRET=staff-only\n")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "Hans")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: staff)
    seen = []

    async def native_handler(native):
        seen.append(native)
        assert get_hermes_home() == staff
        assert get_secret("TSK62_TEST_SECRET") == "staff-only"
        assert sender.runner._adapter_for_source(native.source) is adapter
        assert native.source.profile == "Hans" and native.source.scope_id == "TTEST"
        assert native.source._work_router_owned_final is True
        assert not native.allow_gateway_control
        assert not client.calls  # handler completion is not delivery
        return "staff response"

    sender.runner._handle_message = native_handler
    client = client_at(adapter, {"ok": True, "ts": "2"})
    decision = replace(decision, action=RouteAction("dispatch", "Hans", context="prior context"))
    result = await sender(event=event, state=state, decision=decision, operation_id="op")
    assert result.success and result.message_id == "2"
    assert len(seen) == len(client.calls) == 1
    assert seen[0].channel_context == "prior context"
    assert get_hermes_home() != staff
    # A dict which resembles SendResult returned by the handler cannot confirm.
    sender.runner._handle_message = AsyncMock(return_value={"success": True, "message_id": "forged"})
    result = await sender(event=event, state=state, decision=decision, operation_id="other")
    assert not result.success and len(client.calls) == 1


@pytest.mark.asyncio
async def test_cancel_quarantines_actual_outbox_without_replay(bridge, tmp_path):
    sender, adapter, event, state, decision = bridge
    config, _ = config_at(tmp_path)
    router = WorkRouter(config)
    client = client_at(adapter, asyncio.CancelledError())
    sender.execute = AsyncMock(return_value="staff response")
    try:
        router.ack_gate.complete(event.event_id, success=True)
        await router.enqueue_after_ack(event)
        with pytest.raises(asyncio.CancelledError):
            await router.process_once(event.event_id, sender=sender)
        assert router.store.inbox_status(event.event_id) == "quarantined"
        await router.process_once(event.event_id, sender=sender)
        assert len(client.calls) == 1
    finally:
        router.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response,status", [({"ok": True, "ts": "2"}, "confirmed"),
                                             ({"ok": True}, "quarantined")])
async def test_native_receipt_drives_real_router_outbox(bridge, tmp_path, response, status):
    sender, adapter, event, state, decision = bridge
    config, _ = config_at(tmp_path)
    router = WorkRouter(config)
    client = client_at(adapter, response)
    sender.execute = AsyncMock(return_value="staff response")
    try:
        router.ack_gate.complete(event.event_id, success=True)
        await router.enqueue_after_ack(event)
        result = await router.process_once(event.event_id, sender=sender)
        assert result.status == status
        assert router.store.outbox_status(result.operation_id) == status
        await router.process_once(event.event_id, sender=sender)
        assert len(client.calls) == 1
    finally:
        router.store.close()


@pytest.mark.asyncio
async def test_missing_sender_never_claims_or_prepares(tmp_path):
    config, _ = config_at(tmp_path)
    router = WorkRouter(config)
    try:
        with pytest.raises(RuntimeError, match="sender is not bound"):
            WorkRouterWorker(router, sender=None)
        with pytest.raises(RuntimeError, match="sender is not bound"):
            await router.process_once("missing", sender=None)
        assert router.store.inbox_count() == 0
    finally:
        router.store.close()


@pytest.mark.asyncio
async def test_startup_binds_sender_controls_and_reconnect_without_early_claims(bridge, tmp_path):
    from gateway.work_router.integration import ensure_runtime
    sender, adapter, event, state, decision = bridge
    config, raw = config_at(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"work_router": raw}))
    runtime = ensure_runtime(sender.runner, profile_home=tmp_path)
    try:
        assert isinstance(runtime.sender, NativeRouterSender)
        assert runtime.router._control_sender is runtime.sender
        assert runtime.router._preflight_executor == runtime.sender.execute
        assert not runtime.worker.ready()
        assert ensure_runtime(sender.runner, profile_home=tmp_path) is runtime
        runtime.attach(adapter)
        assert runtime.worker.ready()
        assert runtime.sender.adapter is adapter
        adapter._running = False
        assert not runtime.worker.ready()
    finally:
        await runtime.stop()
    assert runtime.closed


def test_native_stream_consumer_cannot_publish_router_final(bridge):
    from gateway.run_turn_runner import TurnRunner
    sender, adapter, event, state, decision = bridge
    source = sender._source(event, "Hans")
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(source=source)
    assert runner._setup_stream_consumer("slack") == (None, None, None, False)
    wire = source.to_dict()
    assert "_work_router_owned_final" not in wire and "_transport_adapter_ref" not in wire


@pytest.mark.asyncio
async def test_real_gateway_handler_session_to_outbox_transport(tmp_path, monkeypatch):
    """Use the native handler/session/persistence chain; replace only model execution."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig, HomeChannel
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    staff = tmp_path / "Hans"
    staff.mkdir()
    (staff / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "Hans")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: staff)
    runner = GatewayRunner(GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        multiplex_profiles=True,
        platforms={Platform.SLACK: PlatformConfig(
            enabled=True, home_channel=HomeChannel(Platform.SLACK, "CTEST", "Test"),
        )},
    ))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._running = True
    adapter.stop_typing = AsyncMock()
    runner.adapters[Platform.SLACK] = adapter
    sender = NativeRouterSender(runner)
    sender.bind(adapter)
    client = client_at(adapter, {"ok": True, "ts": "native-ts"})
    model_calls = []

    async def model_boundary(**kwargs):
        model_calls.append(kwargs)
        assert kwargs["source"].profile == "Hans"
        assert get_hermes_home() == staff
        assert runner._adapter_for_source(kwargs["source"]) is adapter
        return {"final_response": "native turn answer", "messages": [],
                "completed": True, "api_calls": 1}

    monkeypatch.setattr(runner, "_run_agent", model_boundary)
    event = CanonicalEvent("EvFull", "CTEST", "1700000000.1", text="한스야 확인해",
                           author_user_id="UHUMAN", metadata={"slack_team_id": "TTEST"})
    state = RouterThreadState(event.channel_id, event.thread_ts)
    decision = RouteDecision(RouteAction("dispatch", "Hans"), state)
    try:
        result = await sender(event=event, state=state, decision=decision, operation_id="op")
        assert len(model_calls) == 1
        assert result.success and result.message_id == "native-ts"
        assert len(client.calls) == 1
        source = model_calls[0]["source"]
        entry = runner.session_store.get_or_create_session(source)
        assert entry.session_id == model_calls[0]["session_id"]
        assert "Hans" in model_calls[0]["session_key"]
        assert not runner._is_session_running(model_calls[0]["session_key"])
    finally:
        if runner._session_db is not None:
            await runner._session_db.close()


@pytest.mark.asyncio
async def test_wrong_native_session_namespace_refuses_before_handler(bridge, monkeypatch):
    from hermes_cli import profiles
    sender, adapter, event, state, decision = bridge
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    sender.runner._session_key_for_source = lambda source: "agent:main:slack:CTEST"
    sender.runner._handle_message = AsyncMock(return_value="must not execute")
    decision = replace(decision, action=RouteAction("dispatch", "Hans"))
    with pytest.raises(RuntimeError, match="session namespace"):
        await sender.execute(event=event, state=state, decision=decision, operation_id="op")
    sender.runner._handle_message.assert_not_called()
