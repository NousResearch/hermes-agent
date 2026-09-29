"""Delegated children stay visible on Discord after the dispatching turn ends (#128008).

Drives the real child relay (``_build_child_progress_callback``) into the real ``TurnRunner.progress_callback``
from a worker thread, as the agent does, with a fake Discord adapter recording what reaches the chat.
"""

import asyncio
import queue
from types import SimpleNamespace

import pytest

import gateway.delegated_child_progress as dcp
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tools.delegate_tool_progress import _build_child_progress_callback


class _Adapter:
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, send_id="m1"):
        self.sent, self.edits, self.send_id = [], [], send_id

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id=self.send_id)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)

    def format_tool_preview(self, preview):
        return preview.text

    def latest(self):
        if self.edits:
            return self.edits[-1][2]
        return self.sent[-1][1] if self.sent else None


def _turn(adapter, loop, *, platform=Platform.DISCORD, mode="all", muted=False, chat="thread-1"):
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id=chat), _run_still_current=lambda: False,
        progress_mode=mode, tool_progress_enabled=mode not in {"off", "log"}, progress_queue=queue.Queue(),
        _loop_for_step=loop, _progress_metadata={"thread_id": chat}, mute_notification_reply=muted,
    )
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter, _retain_background_task=lambda t: t)
    turn = TurnRunner(runner, ctx)
    ctx.progress_callback = turn.progress_callback
    return turn


def _child(turn, index=0, count=1, model="claude-opus-5-5", deleg="deleg_a"):
    parent = SimpleNamespace(tool_progress_callback=turn.progress_callback, _delegate_spinner=None, session_id="p")
    return _build_child_progress_callback(
        index, "write add.py", parent, count, subagent_id=f"sa-{index}", parent_id=None, depth=0, model=model,
        toolsets=["terminal"], session_ref={"delegation_id": deleg, "session_id": f"child-{index}"})


async def _settle(adapter, predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate(adapter.latest() or ""):
        assert asyncio.get_running_loop().time() < deadline, adapter.latest()
        await asyncio.sleep(0.02)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(dcp.DelegatedChildProgress, "EDIT_INTERVAL", 0.05)


@pytest.mark.asyncio
async def test_background_child_card_survives_turn_end_and_stays_on_its_route():
    loop = asyncio.get_running_loop()
    adapter = _Adapter()
    first = _turn(adapter, loop)  # _run_still_current() is False: the next message already replaced this turn
    writer, reviewer = _child(first, 0, 2), _child(first, 1, 2, model="gpt-5.5")

    def run_children():
        writer("subagent.start", preview="write add.py")
        reviewer("subagent.start", preview="review add.py")
        writer("tool.started", "terminal", "pytest", {"command": "python3 -m pytest -q tests/<@123>"})
        writer("_thinking", preview="secret chain of thought")
        reviewer("tool.started", "read_file", "add.py", {"path": "add.py"})

    await asyncio.to_thread(run_children)
    await _settle(adapter, lambda text: "Reading" in text)
    card = adapter.latest()
    assert len(adapter.sent) == 1 and adapter.sent[0][0] == "thread-1" and adapter.sent[0][2] == {"thread_id": "thread-1"}
    assert "claude-opus-5-5" in card and "gpt-5.5" in card and "1/2" in card and "2/2" in card
    assert "python3 -m pytest -q" in card and "<@123>" not in card
    assert "secret chain of thought" not in card
    assert first._ctx.progress_queue.empty()  # never the finished turn's own bubble

    # A later turn in another chat must not capture or redirect the earlier batch.
    other = _Adapter(send_id="other")
    _turn(other, loop, chat="thread-2")
    await asyncio.to_thread(lambda: (writer("subagent.complete", preview="done", status="completed", duration_seconds=12),
                                     reviewer("subagent.complete", preview="stopped", status="interrupted")))
    await _settle(adapter, lambda text: "✅" in text and "⏹️" in text)
    assert not other.sent and not other.edits
    assert {mid for _, mid, _ in adapter.edits} == {"m1"} and len(adapter.sent) == 1
    await asyncio.sleep(0.2)
    task = first._child_progress._task
    assert task is not None and task.done()  # terminal: the publisher stops editing


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,platform,muted", [("off", Platform.DISCORD, False), ("log", Platform.DISCORD, False),
                                                 ("all", Platform.DISCORD, True), ("all", Platform.TELEGRAM, False)])
async def test_no_card_when_progress_off_muted_or_not_discord(mode, platform, muted):
    adapter = _Adapter()
    turn = _turn(adapter, asyncio.get_running_loop(), mode=mode, platform=platform, muted=muted)
    child = _child(turn)
    await asyncio.to_thread(lambda: (child("subagent.start", preview="g"), child("tool.started", "terminal", "ls", {"command": "ls"})))
    await asyncio.sleep(0.2)
    assert not adapter.sent and not adapter.edits


@pytest.mark.asyncio
async def test_ambiguous_first_send_is_never_repeated_and_failure_is_contained():
    class _NoId(_Adapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append((chat_id, content, metadata))
            return SendResult(success=True, message_id=None)

    adapter = _NoId()
    turn = _turn(adapter, asyncio.get_running_loop())
    child = _child(turn)
    await asyncio.to_thread(lambda: child("subagent.start", preview="g"))
    await _settle(adapter, lambda text: bool(text))
    await asyncio.to_thread(lambda: [child("tool.started", "terminal", "ls", {"command": f"ls {i}"}) for i in range(4)])
    await asyncio.sleep(0.3)
    assert len(adapter.sent) == 1 and not adapter.edits


@pytest.mark.asyncio
async def test_verbose_keeps_full_terminal_command_and_shutdown_marks_interrupted(monkeypatch):
    adapter = _Adapter()
    turn = _turn(adapter, asyncio.get_running_loop(), mode="verbose")
    child = _child(turn)
    command = "cd /srv/app && python3 -m pytest -q tests/gateway/test_delegated_child_progress.py -k verbose --tb=long"
    await asyncio.to_thread(lambda: (child("subagent.start", preview="g"), child("tool.started", "terminal", "x", {"command": command})))
    await _settle(adapter, lambda text: command in text)
    monkeypatch.setattr(dcp.DelegatedChildProgress, "EDIT_INTERVAL", 30.0)
    await asyncio.to_thread(lambda: child("tool.started", "read_file", "a.py", {"path": "a.py"}))
    task = turn._child_progress._task
    await asyncio.sleep(0.05)
    task.cancel()  # gateway shutdown cancels retained background tasks
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "⏹️" in adapter.latest() and len(adapter.sent) == 1
