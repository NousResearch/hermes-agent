"""Platform-surface pending-slot writes publish to background-review admission.

The base adapter and the gateway runner fence every accepted follow-up already
(``test_background_review_followup_admission.py``). Four surfaces write the pending slot
directly instead: raft's busy-session wake merge, yuanbao's message-recall interrupt, the
runner's /stop, /new, /reset tail that re-parks an internal wake (#114456), and the runner's
post-turn drain restoring a dequeued event behind a leftover /steer (#131644). Each write IS the
next live turn, so an automatic review that sampled "no follow-up" a moment earlier must not get
its full-transcript request onto the wire beside it. Rejected and no-op events must leave
admission untouched, or a dropped message would suppress learning forever.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.yuanbao import InboundContext, RecallGuardMiddleware
from gateway.run import _INTERRUPT_REASON_STOP, GatewayRunner
from gateway.session import SessionSource, build_session_key

# The raft plugin registers its Platform member at import time.
from plugins.platforms.raft.adapter import RaftAdapter  # noqa: E402

RAFT = Platform("raft")


class _DepthTrackingLock:
    """RLock that reports whether the caller is inside it, so a test can prove the review fence
    ran under the session's admission lock rather than after it was released."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0

    def __enter__(self):
        self._lock.acquire()
        self._depth += 1
        return self

    def __exit__(self, *_args) -> None:
        self._depth -= 1
        self._lock.release()

    @property
    def held(self) -> bool:
        return self._depth > 0


def _watch_admission(adapter, session_key: str):
    """Instrument one session's admission state -> (state, fence observations)."""
    state = adapter.followup_admission_state(session_key)
    state.lock = tracking = _DepthTrackingLock()
    fenced_under_admission: list[bool] = []
    adapter.register_followup_review_cancel(
        session_key, lambda: fenced_under_admission.append(tracking.held)
    )
    return state, fenced_under_admission


def _raft_adapter() -> RaftAdapter:
    return RaftAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _raft_wake(text: str = "wake") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=RAFT, chat_id="chan", chat_type="dm", user_id="u1"
        ),
    )


def _raft_session_key(adapter: RaftAdapter, event: MessageEvent) -> str:
    """The key ``RaftAdapter.handle_message`` derives for this event."""
    return build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
        profile=adapter._session_key_profile(event.source),
    )


@pytest.mark.asyncio
async def test_raft_busy_wake_merge_publishes_the_followup_fence():
    adapter = _raft_adapter()
    adapter._message_handler = lambda *_args, **_kwargs: None
    event = _raft_wake()
    session_key = _raft_session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    state, fenced_under_admission = _watch_admission(adapter, session_key)

    await adapter.handle_message(event)

    assert adapter._pending_messages[session_key] is event
    assert fenced_under_admission == [True]
    assert state.epoch == 1


@pytest.mark.asyncio
async def test_raft_wake_dropped_without_a_handler_leaves_admission_untouched():
    """No handler means the wake is dropped, not queued — it must not suppress a review."""
    adapter = _raft_adapter()
    adapter._message_handler = None
    event = _raft_wake()
    session_key = _raft_session_key(adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()
    state, fenced_under_admission = _watch_admission(adapter, session_key)

    await adapter.handle_message(event)

    assert session_key not in adapter._pending_messages
    assert fenced_under_admission == []
    assert state.epoch == 0


def _yuanbao_adapter(monkeypatch) -> BasePlatformAdapter:
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***", extra={}), Platform.YUANBAO
    )
    adapter._processing_msg_texts = {}
    adapter._msg_content_cache = {}
    return adapter


def _recall_push(msg_id: str) -> dict:
    return {
        "callback_command": "Group.CallbackAfterRecallMsg",
        "recall_msg_seq_list": [{"msg_id": msg_id}],
        "group_code": "g1",
        "from_account": "acct",
    }


def test_yuanbao_recall_interrupt_publishes_the_followup_fence(monkeypatch):
    adapter = _yuanbao_adapter(monkeypatch)
    session_key = "yuanbao:group:g1"
    adapter._processing_msg_ids = {session_key: "m-1"}
    adapter._active_sessions[session_key] = asyncio.Event()
    state, fenced_under_admission = _watch_admission(adapter, session_key)

    RecallGuardMiddleware()._handle_recall(
        InboundContext(adapter=adapter, push=_recall_push("m-1")),
        "Group.CallbackAfterRecallMsg",
    )

    queued = adapter._pending_messages[session_key]
    assert "MESSAGE RECALLED" in queued.text
    assert fenced_under_admission == [True]
    assert state.epoch == 1
    assert adapter._active_sessions[session_key].is_set()


def test_yuanbao_recall_for_an_idle_message_leaves_admission_untouched(monkeypatch):
    """A recall that matches no in-flight turn patches the transcript instead of queueing a turn."""
    adapter = _yuanbao_adapter(monkeypatch)
    session_key = "yuanbao:group:g1"
    adapter._processing_msg_ids = {session_key: "m-1"}
    adapter._active_sessions[session_key] = asyncio.Event()
    state, fenced_under_admission = _watch_admission(adapter, session_key)

    RecallGuardMiddleware()._handle_recall(
        InboundContext(adapter=adapter, push=_recall_push("m-other")),
        "Group.CallbackAfterRecallMsg",
    )

    assert session_key not in adapter._pending_messages
    assert fenced_under_admission == []
    assert state.epoch == 0


class _CommandAdapter(BasePlatformAdapter):
    """Concrete adapter for the runner's session-command path; processing starts are recorded."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.restarted: list[MessageEvent] = []

    @property
    def name(self) -> str:
        return "telegram"

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id) -> dict:
        return {"id": chat_id, "type": "private"}

    def _start_session_processing(
        self, event, session_key, *, interrupt_event=None
    ) -> bool:
        self.restarted.append(event)
        return True


def _command_gateway():
    adapter = _CommandAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1"
    )
    key = adapter._event_session_key(
        MessageEvent(text="", message_type=MessageType.TEXT, source=source)
    )
    return adapter, runner, source, key


def _accepted(source: SessionSource, text: str, *, internal: bool) -> MessageEvent:
    event = MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, internal=internal
    )
    event._gateway_accepted = True
    return event


async def _stop_pending_sentinel(runner, adapter, source, key) -> None:
    """``/stop`` with no in-flight turn: the one session command that can run beside a review."""
    adapter._active_sessions[key] = asyncio.Event()
    await runner._interrupt_and_clear_session(
        key,
        source,
        interrupt_reason=_INTERRUPT_REASON_STOP,
        invalidation_reason="stop_command_pending",
    )


@pytest.mark.asyncio
async def test_interrupt_promoting_a_parked_wake_publishes_the_followup_fence():
    """/stop discards the human head and re-parks the wake queued behind it. That slot write is
    the next live turn: it is staged BEFORE the overflow pop (a probe never sees both empty) and
    published through the same fence as every other accepted follow-up."""
    adapter, runner, source, key = _command_gateway()
    adapter._pending_messages[key] = _accepted(source, "human head", internal=False)
    wake = _accepted(
        source, "[ASYNC DELEGATION BATCH COMPLETE] 1 task done", internal=True
    )
    slot_staged_before_pop: list[bool] = []

    class _ProbedOverflow(list):
        def remove(self, item) -> None:
            slot_staged_before_pop.append(adapter._pending_messages.get(key) is item)
            super().remove(item)

    runner._session_state(key).conversation.queued_events = _ProbedOverflow([wake])
    state, fenced_under_admission = _watch_admission(adapter, key)

    await _stop_pending_sentinel(runner, adapter, source, key)

    assert adapter._pending_messages[key] is wake
    assert runner._overflow_queue(key) == []
    assert slot_staged_before_pop == [True]
    assert fenced_under_admission == [True]
    assert state.epoch == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parked_internal", [True, False], ids=["wake_already_parked", "human_only"]
)
async def test_interrupt_without_a_new_slot_write_leaves_admission_untouched(
    parked_internal,
):
    """Keeping an already-parked wake, or discarding a human follow-up, queues no new turn."""
    adapter, runner, source, key = _command_gateway()
    parked = _accepted(source, "parked", internal=parked_internal)
    adapter._pending_messages[key] = parked
    state, fenced_under_admission = _watch_admission(adapter, key)

    await _stop_pending_sentinel(runner, adapter, source, key)

    assert (adapter._pending_messages.get(key) is parked) is parked_internal
    assert fenced_under_admission == []
    assert state.epoch == 0


@pytest.mark.asyncio
async def test_leftover_steer_restoring_a_dequeued_event_publishes_the_followup_fence():
    """A /steer that arrived after the last tool batch runs as the next turn and the event the
    post-turn drain had just dequeued goes back to the head of the queue (#131644). That restore
    is an accepted-follow-up slot write like any other — it IS the turn after the steer — so it
    publishes through the fence: a review that sampled the slot empty between the dequeue and the
    restore is fenced here, never left to run beside the restored turn."""
    adapter, runner, source, key = _command_gateway()
    runner._draining = False
    queued = _accepted(source, "queued while busy", internal=False)
    adapter._pending_messages[key] = queued
    state, fenced_under_admission = _watch_admission(adapter, key)

    event, text = await runner._run_agent_drain_pending(
        {"final_response": "done", "pending_steer": "accepted correction"},
        adapter,
        source,
        key,
    )

    assert (event, text) == (None, "accepted correction")
    assert adapter._pending_messages[key] is queued
    assert fenced_under_admission == [True]
    assert state.epoch == 1
