"""Native Telegram peer work waits for the human turn; each handoff owns a turn."""

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.run_busy import _BOT_ADMISSION_RECEIPT, _FIFO_HANDOFF_RECEIPT
from gateway.session import SessionSource


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []
        self.timeline = []
        self.blocked_response = None
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()
        self.release_first_send.set()
        self.config.extra["group_sessions_per_user"] = False

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content == self.blocked_response:
            self.first_send_started.set()
            await asyncio.wait_for(self.release_first_send.wait(), 10)
        self.sent.append((content, reply_to))
        self.timeline.append(("send", content))
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def make_runner(adapter, mode="interrupt"):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = GatewayConfig(group_sessions_per_user=False, thread_sessions_per_user=False)
    runner.config.streaming.enabled = False
    runner._sessions = {}
    runner._busy_input_mode = mode
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._restart_requested = False
    runner.session_store = None
    runner.pairing_store = None
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._model = "test-model"
    runner._base_url = None
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    return runner


def make_event(text, message_id, *, is_bot=False, **source_fields):
    return MessageEvent(
        text=text, message_id=message_id, message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group",
            user_id="peer" if is_bot else "human", is_bot=is_bot, **source_fields,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_media", [False, True])
@pytest.mark.parametrize("mode,ingress,peer_count,human_at", [
    *((mode, ingress, count, None) for mode in ("interrupt", "steer", "queue")
      for ingress, count in (("base", 2), ("priority", 2), ("priority", 6))),
    ("queue", "base", 6, 3), ("queue", "priority", 6, 4),
])
async def test_peer_waits_for_completed_response_before_its_own_turn(
    monkeypatch, tmp_path, mode, with_media, ingress, peer_count, human_at,
):
    """Exercise Base.handle_message → runner busy FIFO → real turn drain/delivery."""
    import gateway.run as gateway_run

    adapter = CaptureAdapter()
    runner = make_runner(adapter, mode)
    adapter.config.extra["bot_loop"] = {"max_hops": peer_count, "max_events": peer_count}
    prepare_priority_runner(runner, None)
    runner._run_post_turn_hooks = AsyncMock()
    runner._persist_active_agents = Mock()
    runner._clear_durable_active_turn = AsyncMock()
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter.blocked_response = "answer-1"
    adapter.release_first_send.clear()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "test"})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "display": {"interim_assistant_messages": False},
    })
    runner._run_agent = AsyncMock(wraps=runner._run_agent)
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    calls = []
    agent_controls = []
    human = make_event("human task", "100")
    key = runner._session_key_for_source(human.source)

    class ControlledAgent:
        def __init__(self, **kwargs):
            self.tools = []
            self.interrupt = Mock()
            self.steer = Mock(return_value=True)
            self.redirect = Mock(return_value=True)
            self._supports_active_turn_redirect = True
            agent_controls.append(self)

        def run_conversation(self, message, conversation_history=None, task_id=None):
            calls.append((message, list(conversation_history or [])))
            adapter.timeline.append(("start", message))
            if len(calls) == 1:
                # Publish the actual live agent without waiting for the periodic tracker.
                def mark_started():
                    runner._session_state(key).turn.agent = self
                    started.set()
                loop.call_soon_threadsafe(mark_started)
                assert release.wait(10), "test did not release the first turn"
            else:
                # The recursive drain is still busy: peer responses arriving
                # during it cannot extend the already exhausted chain.
                followup = make_event("loop continuation", str(102 + len(calls)) if human_at is None else "101", is_bot=True)
                followup.allow_gateway_control = False
                asyncio.run_coroutine_threadsafe(adapter.handle_message(followup), loop).result(5)
            answer = f"answer-{len(calls)}"
            return {
                "final_response": answer, "api_calls": 1,
                "messages": [*(conversation_history or []),
                             {"role": "user", "content": message},
                             {"role": "assistant", "content": answer}],
            }

    fake_agent_module = ModuleType("run_agent")
    fake_agent_module.AIAgent = ControlledAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_agent_module)

    async def execute(event, source, session_key, generation):
        message = await runner._prepare_profile_scoped_inbound_message_text(
            event=event, source=event.source, history=[], session_key=key,
        )
        result = await runner._run_agent(
            message=message, context_prompt="", history=[], source=event.source,
            session_id="causal-session", session_key=key, event_message_id=event.message_id,
        )
        return result["final_response"]

    runner._handle_message_with_agent = execute
    adapter.set_message_handler(runner._handle_message)
    await adapter.handle_message(human)
    task = adapter._session_tasks[key]
    peer = make_event("peer result", "101", is_bot=True)
    peer.reply_to_message_id = "90"
    peer.reply_to_text = "original request"
    peer.reply_to_author_name = "Coordinator"
    peer.channel_context = "earlier group context"
    second = make_event("second peer result", "102", is_bot=True)
    second.reply_to_message_id = "91"
    second.reply_to_text = "different original request"
    if with_media:
        attachment = tmp_path / "peer-evidence.txt"
        attachment.write_text("peer attachment evidence", encoding="utf-8")
        second.message_type = MessageType.DOCUMENT
        second.media_urls = [str(attachment)]
        second.media_types = ["text/plain"]
    peers = [peer, second, *(make_event(f"handoff-{n}", str(103 + n), is_bot=True)
                             for n in range(peer_count - 2))]
    for n, queued in enumerate(peers[2:]):
        queued.reply_to_message_id = str(80 + n)
        queued.reply_to_text = f"request-{n}"
        queued.channel_context = f"context-{n}"
    if human_at is not None:
        peers[human_at].source.is_bot = False
        peers[human_at].source.user_id = "human"
    for queued in peers:
        queued.allow_gateway_control = not queued.source.is_bot
    try:
        await asyncio.wait_for(started.wait(), 10)
        deliver = runner._handle_message if ingress == "priority" else adapter.handle_message
        for queued in peers:
            await deliver(queued)
        await deliver(peer)  # identical object is still a duplicate on either ingress
        assert adapter._pending_messages.get(key) is peer
        assert runner._overflow_queue(key) == peers[1:]
        assert peer.text == "peer result"
        assert peer.reply_to_message_id == "90"
        assert second.reply_to_message_id == "91"
        assert not adapter._active_sessions[key].is_set()
        for agent in agent_controls:
            agent.interrupt.assert_not_called()
            agent.steer.assert_not_called()
            agent.redirect.assert_not_called()
        assert adapter.sent == []  # no busy ack can trigger another peer handoff
        release.set()
        await asyncio.wait_for(adapter.first_send_started.wait(), 10)
        assert len(calls) == 1  # completion alone isn't enough: final delivery must finish
    finally:
        release.set()
        adapter.release_first_send.set()
        await asyncio.wait_for(task, 10)
        # Above the recursion cap, Base hands off to fresh background tasks.
        while key in adapter._session_tasks:
            await asyncio.wait_for(adapter._session_tasks[key], 10)

    assert [entry[0] for entry in adapter.timeline] == ["start", "send"] * (peer_count + 1)
    assert [content for content, _ in adapter.sent] == [f"answer-{n}" for n in range(1, peer_count + 2)]
    assert [call.kwargs["event_message_id"] for call in runner._run_agent.await_args_list] == [
        human.message_id, *(queued.message_id for queued in peers),
    ]
    next_message, history = calls[1]
    assert "peer result" in next_message
    assert "original request" in next_message
    assert "earlier group context" in next_message
    assert "second peer result" not in next_message
    assert history[-1] == {"role": "assistant", "content": "answer-1"}
    assert "second peer result" in calls[2][0]
    assert "different original request" in calls[2][0]
    assert calls[2][1][-1] == {"role": "assistant", "content": "answer-2"}
    assert runner._queue_depth(key, adapter=adapter) == 0
    assert max(call.kwargs.get("_interrupt_depth", 0) for call in runner._run_agent.await_args_list) <= runner._MAX_INTERRUPT_DEPTH
    for n, queued in enumerate(peers[2:]):
        assert queued.text == f"handoff-{n}"
        assert queued.reply_to_message_id == str(80 + n)
        assert f"handoff-{n}" in calls[n + 3][0]
        assert f"request-{n}" in calls[n + 3][0]
        assert f"context-{n}" in calls[n + 3][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["clarify", "approval"])
async def test_peer_queue_leaves_authorization_human_prompts_and_other_lanes_intact(monkeypatch, prompt):
    """Boundary capability is supplied by Telegram intake; human replies still bypass FIFO."""
    from dataclasses import replace
    from gateway.run import _AGENT_PENDING_SENTINEL
    from tools import approval, clarify_gateway
    from tools.approval_gateway_wait import _ApprovalEntry

    adapter = CaptureAdapter()
    runner = make_runner(adapter)
    runner.config.multiplex_profiles = True
    adapter.set_message_handler(runner._handle_message)
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    human = make_event("1" if prompt == "clarify" else "yes", "200", profile="default")
    key = runner._session_key_for_source(human.source)
    runner._session_state(key).turn.agent = _AGENT_PENDING_SENTINEL
    adapter._active_sessions[key] = asyncio.Event()
    if prompt == "clarify":
        waiting = clarify_gateway.register("causal-clarify", key, "Choose", ["A", "B"])
    else:
        waiting = _ApprovalEntry({"command": "dangerous test command"})
        monkeypatch.setitem(approval._gateway_queues, key, [waiting])

    peers = []
    try:
        # Native bot slash commands are rejected before Base admission; covered
        # by test_native_bot_admission_is_not_control_or_a_chat_gate_bypass.
        for index, text in enumerate((human.text, "restart gateway")):
            peer = make_event(text, str(201 + index), is_bot=True, profile="default")
            # The sibling intake change stamps this BEFORE Base.handle_message; in
            # particular clarify and /stop bypass this mixin if it is missing.
            peer.allow_gateway_control = False
            await adapter.handle_message(peer)
            peers.append(peer)
            assert not waiting.event.is_set()
            assert peer.text == text
        assert adapter._pending_messages[key] is peers[0]
        assert runner._overflow_queue(key) == peers[1:]
        assert adapter.sent == []

        # Approval vocabulary must also be inert when the busy callback receives
        # an admitted native peer directly, independently of the boundary flag.
        if prompt == "approval":
            direct = make_event("yes", "203", is_bot=True, profile="default")
            assert await runner._handle_active_session_busy_message(direct, key)
            assert not waiting.event.is_set()
            assert direct.text == "yes"
            peers.append(direct)

        # Identical chat IDs in another profile and another chat in this profile
        # must not replace the pending head or borrow this session's overflow.
        other_adapter = CaptureAdapter()
        runner._profile_adapters = {"research": {Platform.TELEGRAM: other_adapter}}
        other_events = [
            replace(peers[0], source=replace(peers[0].source, chat_id="-1002")),
            replace(peers[0], source=replace(peers[0].source, profile="research")),
        ]
        other_keys = []
        for other in other_events:
            other_key = runner._session_key_for_source(other.source)
            other_keys.append(other_key)
            assert await runner._handle_active_session_busy_message(other, other_key)
            target = runner._adapter_for_source(other.source)
            assert target._pending_messages[other_key] is other
            assert runner._queue_depth(other_key, adapter=target) == 1
        assert len({key, *other_keys}) == 3
        assert adapter._pending_messages[key] is peers[0]
        assert runner._overflow_queue(key) == peers[1:]
        assert other_adapter.sent == []

        monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "none")
        unauthorized = make_event("untrusted", "unauthorized", is_bot=True, profile="default")
        assert not runner._is_user_authorized(unauthorized.source)
        await adapter.handle_message(unauthorized)
        assert runner._queue_depth(key, adapter=adapter) == len(peers)
        assert adapter.sent == []
        assert not waiting.event.is_set()

        await adapter.handle_message(human)
        assert waiting.event.is_set()
        if prompt == "clarify":
            assert waiting.response == "A"
        else:
            assert waiting.result == "once"
        assert runner._queue_depth(key, adapter=adapter) == len(peers)
        assert not adapter._active_sessions[key].is_set()
    finally:
        clarify_gateway.clear_session(key)
        approval.unregister_gateway_notify(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["peer_media", "human_media_then_peer", "peer_then_photo",
                                    "human_media_then_peer_then_photo"])
async def test_human_queue_media_cannot_cross_peer_barrier(monkeypatch, layout):
    from gateway.run import _AGENT_PENDING_SENTINEL

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, "queue")
    runner._busy_text_mode = adapter._busy_text_mode = "queue"
    adapter.set_message_handler(AsyncMock())
    peer = make_event("peer handoff", "301", is_bot=True)
    peer.allow_gateway_control = False
    human = make_event("later human", "302")
    head = make_event("earlier human photo", "300") if layout.startswith("human_media") else peer
    media = human if layout == "peer_then_photo" else head
    media.message_type = MessageType.PHOTO
    media.media_urls = ["/tmp/evidence.jpg"]
    media.media_types = ["image/jpeg"]
    if layout == "human_media_then_peer_then_photo":
        human.message_type = MessageType.PHOTO
        human.media_urls = ["/tmp/later-evidence.jpg"]
        human.media_types = ["image/jpeg"]
    key = runner._session_key_for_source(peer.source)
    runner._session_state(key).turn.agent = _AGENT_PENDING_SENTINEL
    adapter._active_sessions[key] = asyncio.Event()
    events = [head, peer, human] if head is not peer else [peer, human]
    originals = [(event.text, event.message_id, list(event.media_urls)) for event in events]
    for event in events:
        await adapter.handle_message(event)
    queued = [adapter._pending_messages.get(key), *(runner._overflow_queue(key) or [])]
    assert queued == events
    assert [(event.text, event.message_id, event.media_urls) for event in events] == originals
    assert not adapter._active_sessions[key].is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("depth_extra", [0, 1])
@pytest.mark.parametrize("peer_first", [False, True])
@pytest.mark.parametrize("rewrite", ["allow", "same", "changed"])
async def test_recursion_cap_restores_mixed_photo_head_without_losing_fifo(monkeypatch, depth_extra, peer_first, rewrite):
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, "queue")
    runner._BUSY_QUEUE_MAX_PENDING = 3
    events = [make_event(f"photo-{n}", str(600 + n), is_bot=bot)
              for n, bot in enumerate((peer_first, not peer_first, True))]
    key = runner._session_key_for_source(events[0].source)
    for n, event in enumerate(events):
        event.message_type = MessageType.PHOTO
        event.media_urls = [f"/tmp/photo-{n}.jpg"]
        event.reply_to_message_id = str(590 + n)
        event.reply_to_text = f"request-{n}"
        event.channel_context = f"context-{n}"
        if not runner._queue_busy_peer_event(event, key):
            runner._queue_or_replace_pending_event(key, event)
    result = {"final_response": "finished", "messages": []}
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, events[0].source, key)
    ctx = SimpleNamespace(source=events[0].source, session_id="cap-test", session_key=key,
                          run_generation=None, _interrupt_depth=runner._MAX_INTERRUPT_DEPTH + depth_extra,
                          history=[], _status_thread_metadata=None, result_holder=[result])
    assert await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event, result, result, None) is result
    queued = [adapter._pending_messages[key], *runner._overflow_queue(key)]
    assert all(actual is expected for actual, expected in zip(queued, events, strict=True))
    assert runner._queue_depth(key, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
    runner._queue_or_replace_pending_event(key, make_event("over capacity", "604", is_bot=True))
    assert runner._queue_depth(key, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
    for n, event in enumerate(events):
        assert event.text == f"photo-{n}"
        assert event.media_urls == [f"/tmp/photo-{n}.jpg"]
        assert event.reply_to_message_id == str(590 + n)
        assert event.reply_to_text == f"request-{n}"
        assert event.channel_context == f"context-{n}"
    from hermes_cli import lifecycle
    rewritten_text = "normalized peer" if rewrite == "changed" else pending_event.text
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: [
        {"action": "allow"} if rewrite == "allow" else {"action": "rewrite", "text": rewritten_text},
    ])
    assert await runner._hm_admit_event(pending_event) is None  # still queued, even for humans
    assert adapter._pending_messages.pop(key) is pending_event
    prepare_priority_runner(runner, None)
    runner._handle_message_with_agent = AsyncMock(return_value="cold peer answer")
    runner._run_post_turn_hooks = AsyncMock()
    runner._persist_active_agents = Mock()
    runner._clear_durable_active_turn = AsyncMock()
    assert await runner._handle_message(pending_event) == "cold peer answer"
    dispatched = runner._handle_message_with_agent.call_args.args[0]
    assert dispatched.text == rewritten_text
    assert (dispatched is pending_event) == (rewrite == "allow")
    assert dispatched.message_id == pending_event.message_id
    assert dispatched.reply_to_message_id == pending_event.reply_to_message_id
    runner._handle_message_with_agent.assert_awaited_once()
    if peer_first:
        assert await runner._handle_message(pending_event) is None
        assert await runner._handle_message(dispatched) is None
    else:
        # A human head owns FIFO precedence only, never peer admission.
        assert runner._trusted_queue_receipt(dispatched, _BOT_ADMISSION_RECEIPT) is None
    assert runner._queue_depth(key, adapter=adapter) == len(events) - 1
    assert runner._overflow_queue(key) == events[1:]


def prepare_priority_runner(runner, agent):
    runner._hm_estop_gate = lambda *args: None
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_evict_idle_stale_agent = Mock()
    runner._hm_evict_reaped_agent = Mock()
    key = runner._session_key_for_source(make_event("", "0").source)
    runner._session_state(key).turn.agent = agent
    return key


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["steer", "interrupt", "queue"])
@pytest.mark.parametrize("layout", ["text", "peer_photo", "human_photo_after_peer", "drain_sentinel", "drain_photo"])
async def test_priority_peer_ingress_obeys_shared_fifo_and_drain(monkeypatch, mode, layout):
    """The runner facade is a separate native ingress, without Base's active guard."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    adapter = CaptureAdapter()
    adapter.config.extra["bot_loop"] = {"max_hops": 2, "max_events": 2}
    runner = make_runner(adapter, mode)
    agent = SimpleNamespace(interrupt=Mock(), steer=Mock(return_value=True),
                            redirect=Mock(return_value=True), _supports_active_turn_redirect=True)
    key = prepare_priority_runner(runner, _AGENT_PENDING_SENTINEL if layout == "drain_sentinel" else agent)
    peer = make_event("peer task", "501", is_bot=True)
    peer.allow_gateway_control = False
    human = make_event("human photo", "502" if layout == "human_photo_after_peer" else "500")
    human.message_type = MessageType.PHOTO
    human.media_urls = ["/tmp/human.jpg"]
    if layout in {"peer_photo", "drain_photo"}:
        peer.message_type = MessageType.PHOTO
        peer.media_urls = ["/tmp/peer.jpg"]
    runner._draining = layout.startswith("drain")
    if layout == "peer_photo":
        adapter._pending_messages[key] = human
    assert await runner._handle_message(peer) is None
    if layout == "human_photo_after_peer":
        await runner._handle_message(human)
    expected = [] if runner._draining else ([human, peer] if layout == "peer_photo" else
                [peer, human] if layout == "human_photo_after_peer" else [peer])
    queued = ([adapter._pending_messages[key]] if key in adapter._pending_messages else [])
    queued += runner._overflow_queue(key) or []
    assert queued == expected
    assert peer.text == "peer task"
    assert human.text == "human photo"
    for verb in (agent.steer, agent.redirect, agent.interrupt):
        verb.assert_not_called()
    assert adapter.sent == []
    if runner._draining:
        runner._draining = False
        assert await runner._hm_admit_event(peer) is not None  # stopping must not charge
    else:
        assert await runner._hm_admit_event(peer) is None  # queued redelivery cannot consume receipt
        while True:
            drained, _ = await runner._run_agent_drain_pending({"final_response": "done"}, adapter, peer.source, key)
            if drained is peer:
                break
        assert await runner._hm_admit_event(peer) is not None  # cold fallback consumes receipt once
        assert await runner._hm_admit_event(peer) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["base", "priority"])
@pytest.mark.parametrize("storage", ["full", "unavailable"])
async def test_full_busy_fifo_does_not_charge_rejected_peer(monkeypatch, ingress, storage):
    from gateway.run import _AGENT_PENDING_SENTINEL

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "human")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, "queue")
    runner._BUSY_QUEUE_MAX_PENDING = 1
    adapter.config.extra["bot_loop"] = {
        "max_hops": 1, "max_events": 1, "window_seconds": 60,
    }
    key = prepare_priority_runner(runner, _AGENT_PENDING_SENTINEL)
    blocker = make_event("human already queued", "700")
    runner._queue_or_replace_pending_event(key, blocker)
    if storage == "unavailable":
        del adapter._pending_messages
    peer = make_event("peer handoff", "701", is_bot=True)
    peer.allow_gateway_control = False

    if ingress == "base":
        assert await runner._handle_active_session_busy_message(peer, key)
    else:
        await runner._handle_message(peer)
    if storage == "unavailable":
        adapter._pending_messages = {}
    else:
        assert adapter._pending_messages.pop(key) is blocker
    replay = make_event("peer handoff", "701", is_bot=True)
    replay.allow_gateway_control = False
    if ingress == "base":
        assert await runner._handle_active_session_busy_message(replay, key)
    else:
        await runner._handle_message(replay)
    assert adapter._pending_messages[key] is replay
    assert runner._trusted_queue_receipt(replay, _BOT_ADMISSION_RECEIPT) is not None

    duplicate = make_event("duplicate", "701", is_bot=True)
    duplicate.allow_gateway_control = False
    if ingress == "base":
        assert await runner._handle_active_session_busy_message(duplicate, key)
    else:
        await runner._handle_message(duplicate)
    assert runner._queue_depth(key, adapter=adapter) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("restart,mode,accepted", [(False, "queue", False), (True, "interrupt", False),
                                                  (True, "queue", True), (True, "steer", True)])
async def test_peer_busy_drain_honors_policy_without_ack(monkeypatch, restart, mode, accepted):
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, mode)
    runner._draining = True
    runner._restart_requested = restart
    peer = make_event("peer handoff", "400", is_bot=True)
    key = runner._session_key_for_source(peer.source)
    assert await runner._handle_active_session_busy_message(peer, key)
    assert runner._queue_depth(key, adapter=adapter) == int(accepted)
    assert adapter.sent == []
    if not accepted:
        runner._draining = False
        assert await runner._hm_admit_event(peer) is not None  # dropped input wasn't charged


@pytest.mark.asyncio
async def test_peer_receipt_is_event_bound_one_use_and_does_not_retain_canceled_work(monkeypatch):
    import copy
    import gc
    import weakref
    from hermes_cli import lifecycle

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, "queue")
    peer = make_event("handoff", "800", is_bot=True)
    key = runner._session_key_for_source(peer.source)
    assert runner._queue_busy_peer_event(peer, key)
    runner._issue_queue_receipt(peer, _FIFO_HANDOFF_RECEIPT)
    clone = copy.copy(peer)  # even copying the dynamic opaque tokens grants no authority
    assert not runner._consume_queue_receipt(clone, _FIFO_HANDOFF_RECEIPT)
    assert await runner._hm_admit_event(clone) is None
    assert await runner._hm_admit_event(peer) is None  # still queued
    assert adapter._pending_messages.pop(key) is peer
    reentrant = []

    def rewrite(*args, **kwargs):
        event = kwargs["event"]
        # A hook sees no reusable admission token, even before its rewrite.
        reentrant.append((runner._consume_queue_receipt(event, _BOT_ADMISSION_RECEIPT),
                          runner._consume_queue_receipt(event, _FIFO_HANDOFF_RECEIPT)))
        runner._queue_busy_peer_event(event, key)
        return [{"action": "rewrite", "text": "normalized"}]

    monkeypatch.setattr(lifecycle, "invoke_hook", rewrite)
    admitted = await runner._hm_admit_event(peer)
    assert admitted is not None and admitted[3:] == (True, True)
    assert reentrant == [(False, False)]
    assert runner._queue_depth(key, adapter=adapter) == 0
    assert await runner._hm_admit_event(admitted[0]) is None
    assert await runner._hm_admit_event(peer) is None

    canceled = make_event("canceled", "801", is_bot=True)
    assert runner._queue_busy_peer_event(canceled, key)
    runner._issue_queue_receipt(canceled, _FIFO_HANDOFF_RECEIPT)
    reference = weakref.ref(canceled)
    adapter._pending_messages.clear()  # /stop or reset can drop a queued event
    del canceled
    gc.collect()
    assert reference() is None
    assert not runner._queued_event_receipts

    adapter.config.extra["bot_loop"] = {"max_hops": 2}
    uncharged = make_event("precedence is not admission", "802", is_bot=True)
    runner._issue_queue_receipt(uncharged, _FIFO_HANDOFF_RECEIPT)
    # Moving a valid precedence token onto the budget attribute cannot upgrade it.
    uncharged.__dict__[_BOT_ADMISSION_RECEIPT] = uncharged.__dict__[_FIFO_HANDOFF_RECEIPT]
    assert await runner._hm_admit_event(uncharged) is None
    assert not runner._queued_event_receipts


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["unauthorized", "skip", "command"])
async def test_peer_receipt_cannot_bypass_current_policy(monkeypatch, policy):
    from hermes_cli import lifecycle

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    adapter = CaptureAdapter()
    runner = make_runner(adapter, "queue")
    peer = make_event("handoff", "900", is_bot=True)
    key = runner._session_key_for_source(peer.source)
    assert runner._queue_busy_peer_event(peer, key)
    assert adapter._pending_messages.pop(key) is peer
    if policy == "unauthorized":
        monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "none")
    else:
        result = {"action": "skip"} if policy == "skip" else {"action": "rewrite", "text": "/stop"}
        monkeypatch.setattr(lifecycle, "invoke_hook", lambda *a, **kw: [result])
    assert await runner._hm_admit_event(peer) is None
    assert not runner._queued_event_receipts
