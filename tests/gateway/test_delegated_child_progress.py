"""Delegated children stay visible on Discord after the dispatching turn ends (#128008).

Drives the real child relay (``_build_child_progress_callback``) into the real ``TurnRunner.progress_callback``
from a worker thread, as the agent does, with a fake Discord adapter recording what reaches the chat.
"""

import asyncio
import queue
import re
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


def _turn(adapter, loop, *, platform=Platform.DISCORD, mode="all", muted=False, chat="thread-1",
          turn_current=None, stream_consumer=None):
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id=chat),
        _run_still_current=turn_current or (lambda: False),
        progress_mode=mode, tool_progress_enabled=mode not in {"off", "log"}, progress_queue=queue.Queue(),
        _loop_for_step=loop, _progress_metadata={"thread_id": chat}, mute_notification_reply=muted,
        stream_consumer_holder=[stream_consumer] if stream_consumer is not None else [],
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
    assert len(adapter.sent) == 1 and adapter.sent[0][0] == "thread-1"
    assert adapter.sent[0][2] == {"thread_id": "thread-1", "progress": True, "non_conversational": True}
    assert "claude-opus-5-5" in card and "gpt-5.5" in card and "1/2" in card and "2/2" in card
    assert "python3 -m pytest -q tests/<@123>" in card
    # Preserve command text; the explicit progress metadata enforces no mentions
    # on the real serialized wire (tests/test_discord_child_progress_wire.py).
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


# The native-lane contract is exercised with the actual DiscordAdapter and
# GatewayStreamConsumer in tests/test_discord_child_progress_wire.py. A fake
# accepts_tool_progress=True consumer is not a Discord capability proof.


@pytest.mark.asyncio
async def test_oversized_verbose_command_survives_across_chunks_and_never_rewrites_them():
    """A verbose block past the platform cap is split, not dropped: the whole command reaches the chat
    (tail included) and a delivered continuation is never rewritten."""
    adapter = _Adapter()
    turn = _turn(adapter, asyncio.get_running_loop(), mode="verbose")
    child = _child(turn)
    command = "python check.py " + " ".join(f"--case=case{i:04d}" for i in range(230)) + " --required-tail=KEEP_ME"
    await asyncio.to_thread(lambda: (child("subagent.start", preview="g"),
                                     child("tool.started", "terminal", "x", {"command": command})))

    owner = turn._child_progress
    deadline = asyncio.get_running_loop().time() + 5
    while owner._cursor < len(owner._parts) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    delivered = "\n".join([c[1] for c in adapter.sent] + [e[2] for e in adapter.edits])
    bodies = re.findall(r"```\n(.*?)\n```", delivered, re.S)
    assert "".join(bodies) == command  # every character, reassembled exactly
    assert "KEEP_ME" in delivered  # the tail is on screen, not dropped
    assert len(adapter.sent) >= 2  # a head plus a continuation, never a truncated stub
    assert not re.search(r"\(\d+/\d+\)", delivered)  # no chunk indicators inside the command

    before = list(adapter.sent)
    await asyncio.to_thread(lambda: child("tool.started", "read_file", "a.py", {"path": "a.py"}))
    await asyncio.sleep(0.3)
    assert adapter.sent[: len(before)] == before  # append-only: continuations are written once


@pytest.mark.asyncio
async def test_transient_edit_failure_retries_within_budget_then_stops():
    class _Flaky(_Adapter):
        def __init__(self, result):
            super().__init__()
            self.result, self.calls = result, 0

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.calls += 1
            return self.result

    adapter = _Flaky(SendResult(success=False, error="Connection reset by peer",
                                retryable=True, error_kind="transient"))
    turn = _turn(adapter, asyncio.get_running_loop())
    child = _child(turn)
    await asyncio.to_thread(lambda: child("subagent.start", preview="g"))
    await _settle(adapter, lambda text: bool(text))  # the card exists; now an edit must fail
    await asyncio.to_thread(lambda: child("tool.started", "read_file", "a.py", {"path": "a.py"}))
    owner = turn._child_progress
    deadline = asyncio.get_running_loop().time() + 10
    while not owner._dead and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert owner._dead is True
    assert adapter.calls == 1 + dcp.DelegatedChildProgress.MAX_TRANSPORT_RETRIES


@pytest.mark.asyncio
async def test_non_retryable_edit_failure_stops_immediately_without_a_second_card():
    class _Refused(_Adapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            self.calls += 1
            from gateway.platforms.base import classify_send_error
            err = "50013 Cannot execute action on a system message"
            return SendResult(success=False, error=err, error_kind=classify_send_error(None, err.lower()))

    adapter = _Refused()
    turn = _turn(adapter, asyncio.get_running_loop())
    child = _child(turn)
    await asyncio.to_thread(lambda: child("subagent.start", preview="g"))
    await _settle(adapter, lambda text: bool(text))  # the card exists; now an edit must fail
    await asyncio.to_thread(lambda: child("tool.started", "read_file", "a.py", {"path": "a.py"}))
    owner = turn._child_progress
    await asyncio.sleep(0.5)
    assert owner._dead is True and adapter.calls == 1
    assert len(adapter.sent) == 1  # never a second card


@pytest.mark.parametrize("model,family", [
    ("claude-opus-5-5", "Opus"), ("anthropic/claude-opus-5-5", "Opus"), ("claude-fable-5-1", "Fable"),
    ("gpt-6-astra-900k", "Astra"), ("gpt-6-sol-900k", "Sol"), ("claude-sonnet-4-5", "Sonnet"),
    ("gpt-5.5", "Worker"), ("consolidated-model", "Worker"), ("", "Worker"), (None, "Worker"),
])
def test_model_family_is_readable_and_never_guesses(model, family):
    assert dcp._model_family(model) == family


@pytest.mark.asyncio
async def test_labels_are_friendly_deterministic_and_distinct_across_batches():
    adapter = _Adapter()
    turn = _turn(adapter, asyncio.get_running_loop())
    opus_a, legacy = _child(turn, 0, 2), _child(turn, 1, 2, model=None)
    opus_b = _child(turn, 0, 1, deleg="deleg_b")  # second batch: same subagent_id "sa-0", same model
    unknown = _child(turn, 1, 1, model="gpt-5.5", deleg="deleg_b")

    def run():
        for cb in (opus_a, legacy, opus_b, unknown):
            cb("subagent.start", preview="g")
        for cb in (opus_a, legacy, opus_b, unknown):
            cb("tool.started", "read_file", "f.py", {"path": "f.py"})

    await asyncio.to_thread(run)
    owner = turn._child_progress
    await _settle(adapter, lambda text: text.count("[") >= 4)
    labels = {key: c.label for key, c in owner._children.items()}
    # Raw identity is preserved internally; display labels are start-ordered and unique.
    assert labels == {("deleg_a", "sa-0"): "Opus 1", ("deleg_a", "sa-1"): "Worker 2",
                      ("deleg_b", "sa-0"): "Opus 3", ("deleg_b", "sa-1"): "Worker 4"}
    card = adapter.latest()
    for label in labels.values():
        assert f"[{label}] " in card
    assert "[sa-" not in card and "[child]" not in card
    assert "**Opus 1** (claude-opus-5-5)" in card and "**Worker 4** (gpt-5.5)" in card
    assert "**Worker 2** ·" in card  # legacy/missing model: no invented model text
    # A late event re-uses the same lane's label (stable, not recomputed per event).
    await asyncio.to_thread(lambda: opus_b("tool.started", "terminal", "ls", {"command": "ls"}))
    await _settle(adapter, lambda text: text.count("[Opus 3]") == 2)


@pytest.mark.asyncio
async def test_overflow_continuations_never_reply_to_the_previous_bubble():
    class _Refs(_Adapter):
        def __init__(self):
            super().__init__()
            self.reply_tos, self.n = [], 0

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.reply_tos.append(reply_to)
            self.n += 1
            self.sent.append((chat_id, content, metadata))
            return SendResult(success=True, message_id=f"m{self.n}")

    for anchor in (None, "routing-anchor"):
        adapter = _Refs()
        turn = _turn(adapter, asyncio.get_running_loop(), mode="verbose")
        turn._ctx._progress_reply_to = anchor
        child = _child(turn)
        command = "python check.py " + " ".join(f"--case=case{i:04d}" for i in range(230))
        await asyncio.to_thread(lambda: (child("subagent.start", preview="g"),
                                         child("tool.started", "terminal", "x", {"command": command})))
        owner = turn._child_progress
        deadline = asyncio.get_running_loop().time() + 5
        while owner._cursor < len(owner._parts) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        assert len(adapter.sent) >= 2
        # Head and every continuation carry only the turn's own anchor, never a prior bubble id.
        assert adapter.reply_tos == [anchor] * len(adapter.sent)
        assert all(meta == {"thread_id": "thread-1", "progress": True, "non_conversational": True}
                   for _, _, meta in adapter.sent)
