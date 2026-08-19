"""E2E: an orphaned session guard must not busy-ack the next real message.

Unit coverage lives in test_slack_stale_guard_busy_ack.py. This module drives
the actual ``handle_message`` dispatch path so the fix is proven where the bug
was observed — the Level-1 busy branch — rather than only at the healer.

Both directions are asserted:
  1. after a guard is orphaned, the next message runs a NORMAL turn;
  2. while a turn is genuinely running, a follow-up still reaches the busy
     handler (so interrupt/queue keeps working as configured).

Uses the real ``BasePlatformAdapter.__init__`` (same helper shape as
test_session_split_brain_11016.py) so no attribute the dispatch path touches is
accidentally missing from a hand-built stub.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter._busy_text_mode = ""
    adapter.sent_responses = []

    async def _mock_send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._send_with_retry = _mock_send_retry
    return adapter


def _make_event(text="새 작업 시작", chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _session_key(chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return build_session_key(source)


@pytest.mark.asyncio
async def test_orphaned_guard_does_not_busy_ack_the_next_message():
    """The reported symptom: a fresh message must start a turn, not get an ack."""
    adapter = _make_adapter()
    key = _session_key()

    busy_calls = []

    async def busy_handler(event, session_key):
        busy_calls.append(session_key)
        return True

    async def message_handler(event):
        return f"handled:{event.text}"

    adapter.set_busy_session_handler(busy_handler)
    adapter._message_handler = message_handler

    # Reproduce the orphan exactly as _cancel_session_processing leaves it:
    # guard installed, owner task popped, marker recorded.
    adapter._active_sessions[key] = asyncio.Event()
    adapter._note_orphaned_session_guard(key)

    await adapter.handle_message(_make_event())
    for _ in range(40):
        if adapter.sent_responses:
            break
        await asyncio.sleep(0.05)

    assert not busy_calls, (
        "an orphaned guard sent a brand-new message to the busy handler — this "
        "is the '⚡ Interrupting current task' on first contact"
    )
    assert any("handled:새 작업 시작" in r for r in adapter.sent_responses), (
        f"the message never ran a normal turn (sent={adapter.sent_responses})"
    )
    assert key not in adapter._orphaned_session_guards, (
        "the orphan marker must be consumed once healed"
    )


@pytest.mark.asyncio
async def test_live_run_still_routes_followups_to_busy_handler():
    """Counterpart: genuine mid-run follow-ups must still hit the busy path."""
    adapter = _make_adapter()
    key = _session_key()

    busy_calls = []

    async def busy_handler(event, session_key):
        busy_calls.append(session_key)
        return True

    adapter.set_busy_session_handler(busy_handler)

    # handle_message() returns immediately when no message handler is set, so a
    # handler must exist even though the busy branch should short-circuit before
    # reaching it.
    async def message_handler(event):
        return f"handled:{event.text}"

    adapter._message_handler = message_handler
    started = asyncio.Event()

    async def _long():
        started.set()
        await asyncio.sleep(5)

    task = asyncio.create_task(_long())
    await started.wait()
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = task

    try:
        await adapter.handle_message(_make_event("작업 중 추가 메시지"))

        assert busy_calls == [key], (
            "a follow-up during a live run must reach the busy handler so "
            "interrupt/queue behaviour still works"
        )
        assert key in adapter._active_sessions, "a live guard must not be freed"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
