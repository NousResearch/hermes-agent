"""Tool progress after an approval or clarify prompt starts a new bubble below the prompt.

The tool-progress bubble is one message edited in place. An approval card or clarify poll posts a
new message under it; without a reset, tool lines after the prompt kept editing the bubble ABOVE
the prompt, so after a prompt or two the live status was out of view. Also pins that a reset
delivers lines still held back by the edit throttle instead of dropping them (#29371).
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.turn_context import TurnContext
from tests.waiting import async_wait_until
from tools import clarify_gateway as cm

RESET = ("__reset__",)
APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k"}


# --- Prompts close the progress bubble ------------------------------------------------------


class _PlainApprovalAdapter:
    """No native buttons: the approval goes out as plain text."""

    typed_command_prefix = "/"
    send_exec_approval = None

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        return SendResult(success=True, message_id="approval-1")


def _approval_runner(progress_queue):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=_PlainApprovalAdapter(), _status_chat_id="C1", _status_thread_metadata=None,
        session_key="sk-approval", source=SimpleNamespace(chat_id="C1", platform="discord"),
        progress_queue=progress_queue,
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda *a, **k: None
    return runner


def test_approval_prompt_closes_the_progress_bubble():
    q = queue_mod.Queue()
    q.put("💻 terminal: rm -rf /tmp/x")
    _approval_runner(q)._render_approval_sync(dict(APPROVAL))
    items = [q.get_nowait() for _ in range(q.qsize())]
    # The line queued before the prompt stays in the old bubble; everything after starts fresh.
    assert items == ["💻 terminal: rm -rf /tmp/x", RESET]


def test_prompt_without_a_progress_queue_is_a_no_op():
    """tool_progress off (no queue): closing the bubble must not raise."""
    _approval_runner(None)._render_approval_sync(dict(APPROVAL))


class _DeclinedClarifyAdapter(BasePlatformAdapter):
    def __init__(self):
        pass

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
        return None

    async def send_clarify(self, **kwargs):
        return SendResult(success=False, error="egress declined: destination not allowed")

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="t1")

    async def retire_clarify_card(self, clarify_id, notice):
        return None

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def test_clarify_prompt_closes_the_progress_bubble(loop, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    q = queue_mod.Queue()
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=_DeclinedClarifyAdapter(), _status_chat_id="42", _status_thread_metadata=None,
        session_key="sk-clarify", stream_consumer_holder=[None], _loop_for_step=loop, progress_queue=q)
    runner._close_native_stream_boundary = lambda *a, **k: None
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 5)
    runner._clarify_callback_sync([{"qid": "q0", "question": "Pick?", "choices": ["a", "b"], "multi_select": False}])
    assert q.get_nowait() == RESET


# --- The progress loop honours the reset without losing lines -------------------------------


class _EditAdapter(BasePlatformAdapter):
    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str]] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        msg_id = f"m{len(self.sent) + 1}"
        self.sent.append((msg_id, content))
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append((message_id, content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


async def _render(items: list, done) -> _EditAdapter:
    from gateway.run_turn_runner import TurnRunner

    adapter = _EditAdapter()
    q = queue_mod.Queue()
    for item in items:
        q.put(item)
    ctx = TurnContext(progress_queue=q, source=SimpleNamespace(chat_id="C1"))
    ctx._run_still_current = lambda: True
    runner = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx)
    runner._agent_interrupted = lambda: False
    task = asyncio.create_task(runner.send_progress_messages())
    try:
        # Bounded by the 1.5s edit throttle; the loop drains the queue across a few windows.
        await async_wait_until(lambda: done(adapter), timeout=10, what="progress bubbles")
    finally:
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return adapter


def _final_text(adapter: _EditAdapter, msg_id: str) -> str:
    texts = [c for m, c in adapter.sent if m == msg_id] + [c for m, c in adapter.edits if m == msg_id]
    return texts[-1]


@pytest.mark.asyncio
async def test_lines_after_a_reset_land_in_a_new_bubble_and_none_are_dropped():
    # "two" arrives inside the throttle window right before the prompt's reset: it must still
    # reach the first bubble, and "three" (after the prompt) must start a second one.
    adapter = await _render(
        ["one", "two", RESET, "three"],
        lambda a: len(a.sent) == 2 and ("m1", "one\ntwo") in a.edits)
    assert [m for m, _ in adapter.sent] == ["m1", "m2"]
    assert _final_text(adapter, "m1") == "one\ntwo"
    assert _final_text(adapter, "m2") == "three"


@pytest.mark.asyncio
async def test_a_throttled_first_line_is_posted_before_the_reset():
    # Back-to-back prompts: the first line of the middle bubble is still throttled when the next
    # reset arrives. It gets its own bubble instead of vanishing.
    adapter = await _render(["one", RESET, "two", RESET, "three"], lambda a: len(a.sent) == 3)
    assert [c for _, c in adapter.sent] == ["one", "two", "three"]
