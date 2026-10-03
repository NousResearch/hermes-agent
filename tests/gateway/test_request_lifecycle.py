"""Behavior proof for the supported authorized request capability."""
import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli.plugins import get_plugin_manager, PluginContext
from hermes_cli.plugins_manifest import PluginManifest


def event(text="lookup", *, user="owner", thread="thread", profile="default", message="1"):
    return MessageEvent(text=text, message_id=message, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", chat_type="group",
        user_id=user, thread_id=thread, profile=profile))


class Transport:
    def __init__(self):
        self._pending_messages = {}
        self.sent = []

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs))
        return SimpleNamespace(success=True, message_id=str(len(self.sent)))


@pytest.fixture
def scoped(tmp_path):
    token = set_hermes_home_override(tmp_path)
    manager = get_plugin_manager()
    manager._discovered = True
    # Real PluginContext registration (no patch of runtime plugin methods).
    info = PluginManifest(name="test-request", version="1", description="test", path=tmp_path)
    ctx = PluginContext(info, manager)
    yield ctx
    reset_hermes_home_override(token)


def runner(adapter):
    from gateway.run import GatewayRunner
    obj = object.__new__(GatewayRunner)
    obj.config = SimpleNamespace(multiplex_profiles=False)
    obj._delivery_adapter_for = lambda source: adapter
    obj._profile_scope_for_source = lambda source: contextlib.nullcontext()
    obj._async_profile_scope_for_source = lambda source: contextlib.nullcontext()
    obj._session_key_for_source = lambda source: "session"
    obj._reply_anchor_for_event = lambda evt: evt.message_id
    obj._thread_metadata_for_source = lambda source, reply_to=None: {"thread_id": source.thread_id}
    obj._thread_metadata_for_event = lambda evt: {"thread_id": evt.source.thread_id}
    obj._current_session_run_generation = lambda key: 4
    obj._is_session_run_current = lambda key, gen: gen == 4
    obj._is_user_authorized_for_source = lambda source: source.user_id != "intruder"
    obj._admit_bot_message_for_source = lambda source: True
    obj._effective_busy_input_mode = lambda source: "queue"
    obj._effective_busy_text_mode = lambda source: "interrupt"
    obj._draining = False
    obj._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    obj._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    obj._peek_session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=None, busy_ack_ts=0))
    obj._queue_or_replace_pending_event = lambda key, evt: adapter._pending_messages.update({key: evt})
    obj._resolve_busy_steer_or_redirect = AsyncMock(return_value=SimpleNamespace(
        effective_mode="queue", redirected=False, steered=False,
        demoted_for_subagents=False, demoted_for_compression=False))
    obj._send_busy_ack_reply = AsyncMock()
    obj._compose_busy_ack_message = lambda *args, **kwargs: "queued"
    obj._session_state = obj._peek_session_state
    obj._overflow_queue = lambda key: []
    return obj


@pytest.mark.asyncio
async def test_busy_admission_precedes_queue_and_reuses_event(scoped):
    stages = []
    scoped.register_hook("gateway_request_lifecycle", lambda request, stage: stages.append((request, stage)))
    adapter = Transport()
    obj = runner(adapter)
    evt = event()
    await obj._handle_active_session_busy_message(evt, "session")
    assert stages and stages[0][1] == "admitted"
    request = stages[0][0]
    assert await request.send_interim("checking") is True  # no agent yet
    from gateway.request_lifecycle import admit_request, begin_request
    assert await admit_request(obj, evt, "session") is request
    begin_request(obj, evt, 4)
    assert request.run_generation == 4
    assert request.facts.requester_id == "owner"
    assert len([s for _, s in stages if s == "admitted"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_controls_consumed_before_either_guard(scoped, busy):
    from gateway.request_lifecycle import admit_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    scoped.register_hook("gateway_request_control", lambda control, requests: {
        "request_id": requests[0].facts.request_id, "state": {"detail": "expanded"}}
        if control.text == "detail" and requests else None)
    adapter = Transport()
    obj = runner(adapter)
    original = await admit_request(obj, event(), "session")
    if not busy:
        obj._hm_admit_event = AsyncMock(side_effect=lambda evt: (evt, evt.source, False))
        obj._hm_estop_gate = lambda *args: None
        obj._hm_evict_idle_stale_agent = lambda key: None
        obj._is_session_running = lambda key: True
        obj._hm_evict_reaped_agent = lambda key: None
        obj._hm_handle_running_session_message = AsyncMock()
        await obj._handle_message(event("detail", message="2"))
        obj._hm_handle_running_session_message.assert_not_awaited()
    else:
        await obj._handle_active_session_busy_message(event("detail", message="2"), "session")
    assert original.state["detail"] == "expanded"
    assert not adapter._pending_messages
    obj._resolve_busy_steer_or_redirect.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_cancel_and_supersede_reject_delayed_sends(scoped):
    from gateway.request_lifecycle import admit_request, begin_request, finish_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    obj = runner(adapter)
    for terminal in ("final_ready", "cancelled", "superseded", "failed"):
        evt = event(message=terminal)
        request = await admit_request(obj, evt, "session")
        begin_request(obj, evt, 4)
        finish_request(request, terminal)
        assert await request.send_interim("stale") is False
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_foreign_controls_and_internal_unauthorized_admission(scoped):
    from gateway.request_lifecycle import admit_request, consume_control
    calls = []
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    scoped.register_hook("gateway_request_control", lambda control, requests: calls.append(requests))
    obj = runner(Transport())
    await admit_request(obj, event(), "session")
    for evt in (event("detail", user="other"), event("detail", thread="elsewhere"), event("detail", profile="other")):
        assert await consume_control(obj, evt, "session") is False
    assert not any(calls)
    internal = event(); internal.internal = True
    assert await admit_request(obj, internal, "session") is None
    # Real busy guard drops the unauthorized event before invoking the capability.
    await obj._handle_active_session_busy_message(event(user="intruder"), "session")


@pytest.mark.asyncio
async def test_final_wins_before_scheduled_transport_initiates(scoped):
    from gateway.request_lifecycle import admit_request, finish_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    request = await admit_request(runner(adapter), event(), "session")
    send = asyncio.create_task(request.send_interim("late"))
    # Let send_interim schedule the adapter coroutine, without letting that coroutine run.
    await asyncio.sleep(0)
    finish_request(request, "final_ready")
    assert await send is False
    assert adapter.sent == []
    assert request.interim_receipts == []


@pytest.mark.asyncio
async def test_generation_wins_at_actual_send_boundary(scoped):
    from gateway.request_lifecycle import admit_request, begin_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    obj = runner(adapter)
    request = await admit_request(obj, event(), "session")
    begin_request(obj, request._event, 4)
    send = asyncio.create_task(request.send_interim("late"))
    await asyncio.sleep(0)
    obj._is_session_run_current = lambda key, generation: False
    assert await send is False
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_actual_queue_merge_stops_obsolete_narration(scoped):
    from gateway.request_lifecycle import admit_request, begin_request
    from gateway.platforms.base import merge_pending_message_event
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    obj = runner(adapter)
    friday = event("Friday", message="friday")
    saturday = event("Actually Saturday", message="saturday")
    old = await admit_request(obj, friday, "session")
    revised = await admit_request(obj, saturday, "session")
    adapter._pending_messages["session"] = friday
    merge_pending_message_event(adapter._pending_messages, "session", saturday, merge_text=True)
    assert await old.send_interim("checking Friday") is False
    retained = adapter._pending_messages["session"]
    assert retained._gateway_request_context is revised
    assert revised.facts.text == retained.text
    assert revised.facts.admitted_at == old.facts.admitted_at
    begin_request(obj, retained, 4)
    assert revised.run_generation == 4
    assert await revised.send_interim("checking revised request") is True
    replacement = event("new task", message="new")
    fresh = await admit_request(obj, replacement, "session")
    merge_pending_message_event(adapter._pending_messages, "session", replacement)
    assert await revised.send_interim("obsolete") is False
    assert fresh.active


@pytest.mark.asyncio
async def test_correlated_approval_and_clarification_precede_controls(scoped):
    from gateway.request_lifecycle import admit_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    controls = []
    scoped.register_hook("gateway_request_control", lambda **kwargs: controls.append(kwargs))
    obj = runner(Transport())
    await admit_request(obj, event(), "session")
    obj._hm_pending_reply_intercepts = AsyncMock(return_value="")
    assert await obj._handle_active_session_busy_message(event("detail", message="2"), "session")
    assert controls == []
    obj._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    obj._route_plaintext_approval_while_busy = AsyncMock(return_value=True)
    assert await obj._handle_active_session_busy_message(event("detail", message="3"), "session")
    assert controls == []


@pytest.mark.asyncio
async def test_latest_state_formats_final_before_stream_seal(scoped):
    from gateway.request_lifecycle import admit_request, begin_request
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    scoped.register_hook("gateway_request_final", lambda request, response: response + request.state["detail"])
    obj = runner(Transport())
    evt = event()
    request = await admit_request(obj, evt, "session")
    begin_request(obj, evt, 4)
    request.state["detail"] = " latest"
    ctx = TurnContext(source=evt.source, session_key="session", run_generation=4)
    turn = TurnRunner(obj, ctx)
    sealed = []
    class Stream:
        def finish(self, text):
            assert request.stage == "final_ready"
            sealed.append(text)
    result = {"final_response": "answer", "messages": []}
    turn._finish_stream_consumer(result, [], Stream())
    assert sealed == ["answer latest"]
    assert await request.send_interim("late") is False


@pytest.mark.asyncio
async def test_interim_and_final_transport_receipts_are_success_based(scoped):
    from gateway.request_lifecycle import admit_request, finish_request, record_final_delivery
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    obj = runner(Transport())
    request = await admit_request(obj, event(), "session")
    assert await request.send_interim("checking") is True
    receipt = request.interim_receipts[0]
    assert receipt.initiated_at >= request.facts.admitted_at
    assert receipt.delivered_at >= receipt.initiated_at
    finish_request(request, "final_ready")
    record_final_delivery(request, success=False, initiated_at=receipt.initiated_at)
    assert request.final_receipts[-1].delivered_at is None
    record_final_delivery(request, success=True, initiated_at=receipt.initiated_at)
    assert request.final_receipts[-1].delivered_at is not None
    assert request.final_ready_at is not None


@pytest.mark.asyncio
async def test_profile_a_b_a_real_discovery_and_original_transport(tmp_path):
    from gateway.request_lifecycle import admit_request, begin_request, final_response
    from gateway.run import _profile_runtime_scope
    from gateway.session_identity import RoutingIdentity
    import weakref
    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        plugin = home / "plugins" / "request-probe"
        plugin.mkdir(parents=True)
        (home / "config.yaml").write_text("plugins:\n  enabled: [request-probe]\n", encoding="utf-8")
        (plugin / "plugin.yaml").write_text("name: request-probe\nversion: '1'\n", encoding="utf-8")
        (plugin / "__init__.py").write_text('''
from hermes_constants import get_hermes_home

def register(ctx):
    def lifecycle(request, stage):
        request.state.setdefault("scopes", []).append((stage, str(get_hermes_home())))
    ctx.register_hook("gateway_request_lifecycle", lifecycle)
    ctx.register_hook("gateway_request_final", lambda request, response: response + get_hermes_home().name)
''')
    transports = [Transport(), Transport()]
    obj = runner(transports[0])
    obj._profile_scope_for_source = lambda source: _profile_runtime_scope(source._identity.runtime_home)
    obj._delivery_adapter_for = lambda source: source._identity.adapter()
    requests = []
    for idx in (0, 1, 0):
        home, adapter = homes[idx], transports[idx]
        evt = event(profile=home.name, message=str(len(requests)))
        evt.source._identity = RoutingIdentity(home.name, home.name, home, home, transport=weakref.ref(adapter))
        request = await admit_request(obj, evt, "session" + home.name)
        assert request.state["scopes"] == [("admitted", str(home))]
        requests.append(request)
    token = set_hermes_home_override(homes[1])
    try:
        assert await requests[0].send_interim("A delayed") is True
        assert await requests[1].send_interim("B") is True
        assert await requests[2].send_interim("A again") is True
        for request in requests:
            begin_request(obj, request._event, 4)
            assert final_response(obj, request.facts.session_key, 4, "answer ").endswith(request.facts.runtime_profile)
    finally:
        reset_hermes_home_override(token)
    assert [content for _, content, _ in transports[0].sent] == ["A delayed", "A again"]
    assert [content for _, content, _ in transports[1].sent] == ["B"]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    for request in requests:
        assert all(home == request.facts.profile_home for _, home in request.state["scopes"])


@pytest.mark.asyncio
async def test_no_consumer_keeps_original_busy_dispatch(scoped):
    adapter = Transport()
    obj = runner(adapter)
    obj._hm_pending_reply_intercepts = AsyncMock(side_effect=AssertionError("new behavior without consumer"))
    evt = event()
    assert await obj._handle_active_session_busy_message(evt, "session") is True
    assert adapter._pending_messages["session"] is evt
    assert not hasattr(evt, "_gateway_request_context")


from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class NativeTransport(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="synthetic"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id=str(len(self.sent)))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
async def test_real_adapter_guard_consumes_control_without_replaying_work(scoped, mode):
    from gateway.request_lifecycle import admit_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    scoped.register_hook("gateway_request_control", lambda control, requests: {
        "request_id": requests[0].facts.request_id, "state": {"detail": mode}})
    adapter = NativeTransport()
    obj = runner(adapter)
    obj._effective_busy_input_mode = lambda source: mode
    adapter.gateway_runner = obj
    adapter.set_busy_session_handler(obj._handle_active_session_busy_message)
    adapter._active_sessions["session"] = asyncio.Event()
    original = event()
    adapter._canonicalize(original.source)
    request = await admit_request(obj, original, "session")
    await adapter._handle_message_while_active(event("detail", message="2"), "session")
    assert request.state["detail"] == mode
    assert adapter._pending_messages == {}
    obj._resolve_busy_steer_or_redirect.assert_not_awaited()
    assert not adapter.sent


@pytest.mark.asyncio
async def test_native_final_send_receipt_and_adapter_cancellation(scoped):
    from gateway.request_lifecycle import admit_request, finish_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = NativeTransport()
    obj = runner(adapter)
    adapter.gateway_runner = obj
    adapter._final_delivery_adapter = lambda source: adapter
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    evt = event()
    request = await admit_request(obj, evt, "session")
    finish_request(request, "final_ready")
    await adapter.send_final_ledgered(evt, "session", "answer", {}, reply_to=evt.message_id)
    assert request.final_receipts[-1].success
    assert request.final_receipts[-1].delivered_at >= request.final_receipts[-1].initiated_at
    pending = event(message="pending")
    queued = await admit_request(obj, pending, "session")
    adapter._pending_messages["session"] = pending
    await adapter.cancel_session_processing("session")
    assert await queued.send_interim("obsolete queue") is False


@pytest.mark.asyncio
async def test_real_debounce_revision_retains_correct_owner(scoped):
    from gateway.request_lifecycle import admit_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = NativeTransport()
    obj = runner(adapter)
    evt = event("Friday", message="1")
    old = await admit_request(obj, evt, "session")
    await adapter._queue_text_debounce("session", evt)
    revision = event("Actually Saturday", message="2")
    new = await admit_request(obj, revision, "session")
    await adapter._queue_text_debounce("session", revision)
    assert not old.active
    assert adapter._text_debounce["session"].event._gateway_request_context is new
    assert "Saturday" in new.facts.text
    adapter._discard_text_debounce("session")
    assert not new.active


@pytest.mark.asyncio
async def test_real_cold_authorization_and_pending_sentinel_admission(scoped):
    from gateway.run import _AGENT_PENDING_SENTINEL
    scoped.register_hook("gateway_request_lifecycle", lambda request, stage: request.state.update({"seen": stage}))
    adapter = Transport()
    obj = runner(adapter)
    obj._scale_to_zero_note_real_inbound = lambda: None
    obj._hm_pre_gateway_dispatch_hook = AsyncMock(side_effect=lambda evt, source: evt)
    obj._hm_estop_gate = lambda *args: None
    obj._hm_evict_idle_stale_agent = lambda key: None
    obj._hm_evict_reaped_agent = lambda key: None
    obj._is_session_running = lambda key: True
    obj._hm_busy_telegram_grace_queue = lambda *args: False
    obj._peek_session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=_AGENT_PENDING_SENTINEL))
    evt = event()
    await obj._handle_message(evt)
    request = evt._gateway_request_context
    assert request.facts.requester_id == "owner"
    assert request.state["seen"] == "admitted"
    assert request.run_generation is None
    assert adapter._pending_messages["session"] is evt
    assert await request.send_interim("queued preparation") is True
    foreign = event(user="intruder")
    await obj._handle_message(foreign)
    assert not hasattr(foreign, "_gateway_request_context")


@pytest.mark.asyncio
async def test_actual_cross_owner_merge_preserves_fifo_requests(scoped):
    from gateway.request_lifecycle import admit_request, begin_request, finish_request
    from gateway.platforms.base import merge_pending_message_event
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    obj = runner(adapter)
    overflow = []
    obj._session_state = lambda key: SimpleNamespace(conversation=SimpleNamespace(queued_events=overflow))
    obj._overflow_queue = lambda key: overflow
    first = event("owner A request", user="A", message="A")
    second = event("owner B request", user="B", message="B")
    a = await admit_request(obj, first, "session")
    b = await admit_request(obj, second, "session")
    adapter._pending_messages["session"] = first
    merge_pending_message_event(adapter._pending_messages, "session", second, merge_text=True)
    assert adapter._pending_messages["session"] is first
    assert overflow == [second]
    assert first.text == "owner A request" and second.text == "owner B request"
    assert await a.send_interim("A checking") is True
    assert await b.send_interim("B checking") is True
    executing = adapter._pending_messages.pop("session")
    begin_request(obj, executing, 4)
    assert executing.source.user_id == a.facts.requester_id == "A"
    assert a.run_generation == 4 and b.run_generation is None
    finish_request(a, "final_ready")
    executing = obj._promote_queued_event("session", adapter, None)
    begin_request(obj, executing, 4)
    assert executing.source.user_id == b.facts.requester_id == "B"
    assert b.run_generation == 4 and b.active
    assert [kwargs["reply_to"] for _, _, kwargs in adapter.sent] == ["A", "B"]
