"""Platform-surface pending-slot writes publish to background-review admission.

The base adapter and the gateway runner fence every accepted follow-up already
(``test_background_review_followup_admission.py``). Two platform surfaces write the pending slot
directly instead: raft's busy-session wake merge and yuanbao's message-recall interrupt. Each
write IS the next live turn, so an automatic review that sampled "no follow-up" a moment earlier
must not get its full-transcript request onto the wire beside it. Rejected and no-op events must
leave admission untouched, or a dropped message would suppress learning forever.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.yuanbao import InboundContext, RecallGuardMiddleware
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
