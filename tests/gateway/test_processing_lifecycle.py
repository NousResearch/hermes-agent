"""Processing-lifecycle contracts that every bracketing adapter observes.

Adapters render ``on_processing_start`` and ``on_processing_complete`` as
in-progress and outcome markers (reactions, read receipts, typing cards), so
every started message must complete exactly once, with an outcome that
describes what happened to that message.
"""

import asyncio
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.run import _INTERRUPT_REASON_STOP, GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
)


class LifecycleLogAdapter(HookRecordingAdapter):
    """Records starts, completions and sends in one ordered log."""

    def __init__(self):
        super().__init__()
        self.log: list = []
        self.refused: set = set()

    async def on_processing_start(self, event):
        self.log.append(("start", event.message_id))

    async def on_processing_complete(self, event, outcome):
        self.log.append(("complete", event.message_id, outcome))

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("send", content))
        if any(refused in content for refused in self.refused):
            return SendResult(success=False, error="refused")
        return SendResult(success=True, message_id="sent")


class _ScriptedAgent:
    """Returns one scripted result per call, in order."""

    results: list = []
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        return type(self).results[len(type(self).calls) - 1]


def _interrupted(message):
    return {"final_response": "Operation interrupted.", "messages": [], "api_calls": 1,
            "interrupted": True, "completed": False, "interrupt_message": message}


def _done(text):
    return {"final_response": text, "messages": [], "api_calls": 1}


async def _run_chain(monkeypatch, tmp_path, results, queued_text="the voice note"):
    _ScriptedAgent.calls, _ScriptedAgent.results = [], results
    _install_fake_agent(monkeypatch, tmp_path, _ScriptedAgent)
    adapter = LifecycleLogAdapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text=queued_text, source=_source(), message_id="queued-1")

    async def respond(event):
        result = await runner._run_agent(
            message=event.text, context_prompt="", history=[], source=event.source,
            session_id="sess-lifecycle", session_key=SESSION_KEY, processing_event=event)
        return result["final_response"]

    adapter.set_message_handler(respond)
    await adapter.handle_message(MessageEvent(text="the first turn", source=_source(), message_id="first-1"))
    await asyncio.gather(*adapter._background_tasks)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupt_message", "expected"),
    [
        (
            "the voice note",
            [("start", "first-1"), ("start", "queued-1"), ("send", "done-2"),
             ("complete", "queued-1", ProcessingOutcome.SUCCESS),
             ("complete", "first-1", ProcessingOutcome.SUCCESS)],
        ),
        (
            _INTERRUPT_REASON_STOP,
            [("start", "first-1"), ("complete", "first-1", ProcessingOutcome.CANCELLED),
             ("start", "queued-1"), ("send", "done-2"),
             ("complete", "queued-1", ProcessingOutcome.SUCCESS)],
        ),
    ],
    ids=["superseded-by-input", "stopped"],
)
async def test_interrupted_turn_outcome_depends_on_what_interrupted_it(
    monkeypatch, tmp_path, interrupt_message, expected
):
    """New input supersedes the turn, so the message completes with the chain that answers it; a stop
    cancels it."""
    adapter = await _run_chain(monkeypatch, tmp_path, [_interrupted(interrupt_message), _done("done-2")])

    assert (_ScriptedAgent.calls, adapter.log) == (["the first turn", "the voice note"], expected)


def _priority_runner(monkeypatch, mode):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = LifecycleLogAdapter()
    adapter.platform = Platform.SLACK
    runner = GatewayRunner(config=GatewayConfig())
    runner.adapters[Platform.SLACK] = adapter
    runner._busy_input_mode = mode
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    adapter.set_message_handler(runner._handle_message)
    return runner, adapter


_AGENT_VERBS = {"steer", "redirect", "interrupt"}
_OK, _FAILED, _CANCELLED = ProcessingOutcome.SUCCESS, ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED


def _started(message_id):
    return ("start", message_id)


def _completed(message_id, outcome):
    return ("complete", message_id, outcome)


class _WalkModel:
    """Fake AIAgent behind the real runner. Each model call waits until the test releases it. The
    first ``consumed`` steers or redirects of a call count as read by the model; the rest come back
    as ``pending_steer``, as they do when they arrive after the last tool batch. After an
    ``interrupt`` the call returns an interrupted result whose ``interrupt_message`` is the text."""

    walk: "_WalkRun"
    _supports_active_turn_redirect = True

    def __init__(self, **_kwargs):
        self.tools, self._active_children, self._steers, self._interrupt = [], [], [], None

    def steer(self, text):
        self._steers.append(text)
        return True

    redirect = steer

    def interrupt(self, message=None):
        """An interrupt ends the model call in progress, as it does in the agent."""
        self._interrupt = message
        walk = type(self).walk
        for release in walk.model_release[:len(walk.model_calls)]:
            walk.loop.call_soon_threadsafe(release.set)

    def get_activity_summary(self):
        return {"seconds_since_activity": 0}

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        walk = type(self).walk
        index = len(walk.model_calls)
        walk.model_calls.append(message)
        walk.loop.call_soon_threadsafe(walk.model_started[index].set)
        assert walk.model_release[index].wait(30), "The test did not release the model call"
        if index == walk.failing_call:
            raise RuntimeError("model call failed")
        steers, self._steers = self._steers, []
        interrupt, self._interrupt = self._interrupt, None
        result = {"final_response": f"done-{index + 1}", "messages": [], "api_calls": 1}
        if steers[walk.consumed:]:
            result["pending_steer"] = "\n".join(steers[walk.consumed:])
        if interrupt:
            result.update(interrupted=True, completed=False, interrupt_message=interrupt)
        return result


class _WalkAdapter(LifecycleLogAdapter):
    """With ``hold_ack``, a /steer acknowledgement is not sent until the test releases it. With
    ``hold_completion``, the completion hook for that message does not return."""

    def __init__(self, hold_ack):
        super().__init__()
        self.hold_ack, self.ack_sending, self.ack_release = hold_ack, asyncio.Event(), asyncio.Event()
        self.hold_completion, self.completion_held = None, asyncio.Event()

    async def on_processing_complete(self, event, outcome):
        await super().on_processing_complete(event, outcome)
        if self.hold_completion is not None and event.message_id == self.hold_completion:
            self.completion_held.set()
            await asyncio.Event().wait()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if self.hold_ack and content.startswith("⏩"):
            self.ack_sending.set()
            await self.ack_release.wait()
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


class _WalkCompleteOnlyAdapter(_WalkAdapter):
    """Overrides only on_processing_complete, as A2A, Google Chat and the webhook adapter do."""

    on_processing_start = BasePlatformAdapter.on_processing_start


@dataclass(frozen=True)
class _Walk:
    """One route through the priority path. The adapter is idle while the runner owns the opening
    turn, so the adapter starts every correction itself before the runner handles it."""

    expected: tuple
    verb: str = "steer"            # steer | redirect | interrupt | queue | photo (merged into a queued follow-up)
    leftovers: int = 0             # corrections that the model does not read in the opening turn
    turn: str = "tracked"          # tracked | untracked (no platform message) | hooks-done (completed early)
    ack: str = "returned"          # returned | in-flight (/steer, acknowledgement held until after the drain) |
    #                                late (held until the copy's own model call has started)
    placement: str = "direct"      # direct | fifo | overflow | cap: where the leftover copy waits |
    #                                goal | notification: the slot holds a synthetic prompt without hooks |
    #                                cap-goal: a /goal continuation waits behind the message, and the turn
    #                                ends at the recursion cap |
    #                                orphan: /stop leaves the message in the overflow FIFO, and the next
    #                                message rescues it (exit success | stop-orphan)
    exit: str = "success"          # success | stop | new | reset (before the copy starts) | stop-turn |
    #                                stop-queued | new-queued | reset-queued (during a turn) | refused | exception |
    #                                teardown-turn | teardown-queued (adapter shutdown during a turn)
    adapter: str = "bracketing"    # bracketing | complete-only


@dataclass
class _WalkRun:
    loop: asyncio.AbstractEventLoop
    consumed: int
    failing_call: int = -1
    model_calls: list = field(default_factory=list)
    model_started: list = field(default_factory=lambda: [asyncio.Event() for _ in range(5)])
    model_release: list = field(default_factory=lambda: [threading.Event() for _ in range(5)])


_OPENING = "opening-1"
_C1, _C2, _Q1, _Q2, _P1, _N1 = "corr-1", "corr-2", "queued-1", "queued-2", "photo-1", "next-1"
_S_OPEN = _started(_OPENING)


def _leftover_rows():
    """Leftover steers on a tracked turn, for every placement of the copy and every exit."""
    rows = {}
    for count in (1, 2):
        ids = [_C1, _C2][:count]
        starts = (_S_OPEN, *[_started(i) for i in ids])
        done = lambda outcome: tuple(_completed(i, outcome) for i in ids)  # noqa: E731
        opened = (*starts, _completed(_OPENING, _OK))
        rows[f"direct-{count}-success"] = _Walk((*opened, *done(_OK)), leftovers=count)
        for exit in ("stop", "new", "reset"):
            rows[f"direct-{count}-{exit}"] = _Walk((*opened, *done(_CANCELLED)), leftovers=count, exit=exit)
        rows[f"direct-{count}-refused"] = _Walk((*opened, *done(_FAILED)), leftovers=count, exit="refused")
        rows[f"direct-{count}-exception"] = _Walk((*opened, *done(_FAILED)), leftovers=count, exit="exception")
        queued = (*opened, _started(_Q1))
        rows[f"fifo-{count}-success"] = _Walk(
            (*queued, _completed(_Q1, _OK), *done(_OK)), leftovers=count, placement="fifo")
        rows[f"fifo-{count}-stop"] = _Walk(
            (*queued, _completed(_Q1, _OK), *done(_CANCELLED)), leftovers=count, placement="fifo", exit="stop")
        rows[f"fifo-{count}-stop-queued"] = _Walk(
            (*queued, *done(_CANCELLED), _completed(_Q1, _CANCELLED)), leftovers=count, placement="fifo",
            exit="stop-queued")
        rows[f"cap-{count}-success"] = _Walk((*opened, *done(_OK)), leftovers=count, placement="cap")
        rows[f"cap-{count}-stop"] = _Walk((*opened, *done(_CANCELLED)), leftovers=count, placement="cap", exit="stop")
    one = (_S_OPEN, _started(_C1), _completed(_OPENING, _OK))
    queued = (*one, _started(_Q1))
    rows.update({
        "fifo-1-reset": _Walk((*queued, _completed(_Q1, _OK), _completed(_C1, _CANCELLED)), leftovers=1, placement="fifo",
                              exit="reset"),
        "fifo-1-new-queued": _Walk((*queued, _completed(_C1, _CANCELLED), _completed(_Q1, _CANCELLED)), leftovers=1,
                                   placement="fifo", exit="new-queued"),
        "fifo-1-refused": _Walk((*queued, _completed(_Q1, _OK), _completed(_C1, _FAILED)), leftovers=1, placement="fifo",
                                exit="refused"),
        "fifo-1-exception": _Walk((*queued, _completed(_Q1, _OK), _completed(_C1, _FAILED)), leftovers=1, placement="fifo",
                                  exit="exception"),
        "overflow-1-success": _Walk(
            (*queued, _completed(_Q1, _OK), _started(_Q2), _completed(_Q2, _OK), _completed(_C1, _OK)), leftovers=1,
            placement="overflow"),
        "overflow-1-new-queued": _Walk((*queued, _completed(_C1, _CANCELLED), _completed(_Q1, _CANCELLED)), leftovers=1,
                                       placement="overflow", exit="new-queued"),
        "overflow-1-teardown-queued": _Walk(
            (*queued, _completed(_Q1, _CANCELLED), _completed(_C1, _CANCELLED)), leftovers=1, placement="overflow",
            exit="teardown-queued"),
        "overflow-1-reset-queued": _Walk((*queued, _completed(_C1, _CANCELLED), _completed(_Q1, _CANCELLED)), leftovers=1,
                                         placement="overflow", exit="reset-queued"),
        "cap-1-new": _Walk((*one, _completed(_C1, _CANCELLED)), leftovers=1, placement="cap", exit="new"),
        "cap-1-reset": _Walk((*one, _completed(_C1, _CANCELLED)), leftovers=1, placement="cap", exit="reset"),
        "cap-1-exception": _Walk((*one, _completed(_C1, _FAILED)), leftovers=1, placement="cap", exit="exception"),
    })
    return rows


_CONSUMED = (_S_OPEN, _started(_C1))
_WALKS = {
    # The running turn reads the correction, which completes with that turn's outcome.
    **{f"{verb}-consumed": _Walk((*_CONSUMED, _completed(_C1, _OK), _completed(_OPENING, _OK)), verb=verb)
       for verb in ("steer", "redirect", "interrupt")},
    **{f"{verb}-consumed-stop-turn": _Walk((*_CONSUMED, _completed(_C1, _CANCELLED), _completed(_OPENING, _CANCELLED)),
                                           verb=verb, exit="stop-turn")
       for verb in ("steer", "redirect")},
    "steer-consumed-exception": _Walk((*_CONSUMED, _completed(_C1, _FAILED), _completed(_OPENING, _FAILED)),
                                      exit="exception"),
    # An interrupt's text runs as the next turn, which answers the interrupted turn too.
    "interrupt-stop-queued": _Walk((*_CONSUMED, _completed(_C1, _CANCELLED), _completed(_OPENING, _CANCELLED)),
                                   verb="interrupt", exit="stop-queued"),
    "interrupt-exception": _Walk((*_CONSUMED, _completed(_C1, _FAILED), _completed(_OPENING, _FAILED)),
                                 verb="interrupt", exit="exception"),
    # A started message that waits in the queue completes with its own turn, or when it is dropped.
    "queue": _Walk((*_CONSUMED, _completed(_OPENING, _OK), _completed(_C1, _OK)), verb="queue"),
    # A photo that the priority path merges into a queued follow-up completes with that follow-up.
    "photo-merged": _Walk(
        (_S_OPEN, _started(_P1), _completed(_OPENING, _OK), _started(_Q1), _completed(_P1, _OK),
         _completed(_Q1, _OK)), verb="photo", placement="fifo"),
    # A synthetic slot event has no hooks; the photo merged into it completes with its turn.
    **{f"photo-merged-into-{kind}": _Walk(
        (_S_OPEN, _started(_P1), _completed(_OPENING, _OK), _completed(_P1, _OK)), verb="photo", placement=kind)
       for kind in ("goal", "notification")},
    # At the recursion cap the parked photo merges into the /goal continuation behind it, whose turn
    # the adapter runs later (with its hooks, for an event without a platform message).
    "photo-parked-at-cap-into-goal": _Walk(
        (_S_OPEN, _started(_P1), _completed(_OPENING, _OK), _started(None), _completed(_P1, _OK),
         _completed(None, _OK)), verb="photo", placement="cap-goal"),
    "photo-merged-stop-turn": _Walk(
        (_S_OPEN, _started(_P1), _completed(_P1, _CANCELLED), _completed(_OPENING, _CANCELLED)), verb="photo",
        placement="fifo", exit="stop-turn"),
    # Gateway shutdown or profile teardown cancels the running message and completes the parked one.
    "queue-teardown-turn": _Walk((*_CONSUMED, _completed(_OPENING, _CANCELLED), _completed(_C1, _CANCELLED)),
                                 verb="queue", exit="teardown-turn"),
    # /stop leaves a started message in the overflow FIFO; the next message's turn rescues it first.
    "orphan": _Walk((*_CONSUMED, _completed(_OPENING, _CANCELLED), _started(_N1), _completed(_C1, _OK),
                     _completed(_N1, _OK)), verb="queue", placement="orphan"),
    "orphan-stop": _Walk((*_CONSUMED, _completed(_OPENING, _CANCELLED), _started(_N1), _completed(_N1, _CANCELLED),
                          _completed(_C1, _CANCELLED)), verb="queue", placement="orphan", exit="stop-orphan"),
    "queue-stop-turn": _Walk((*_CONSUMED, _completed(_C1, _CANCELLED), _completed(_OPENING, _CANCELLED)), verb="queue",
                             exit="stop-turn"),
    **_leftover_rows(),
    # The /steer acknowledgement is still being sent when the model returns the steer as leftover.
    "in-flight-success": _Walk((*_CONSUMED, _completed(_OPENING, _OK), _completed(_C1, _OK)), leftovers=1, ack="in-flight"),
    "in-flight-failure": _Walk((*_CONSUMED, _completed(_OPENING, _OK), _completed(_C1, _FAILED)), leftovers=1,
                            ack="in-flight", exit="exception"),
    "in-flight-stop": _Walk((*_CONSUMED, _completed(_OPENING, _OK), _completed(_C1, _CANCELLED)), leftovers=1,
                            ack="in-flight", exit="stop"),
    # The hooks do not track the running turn (it has no platform message).
    **{f"untracked-{verb}-consumed": _Walk((_started(_C1), _completed(_C1, _OK)), verb=verb, turn="untracked")
       for verb in ("steer", "redirect", "interrupt")},
    "untracked-leftover": _Walk((_started(_C1), _completed(_C1, _OK)), leftovers=1, turn="untracked"),
    "untracked-in-flight-success": _Walk((_started(_C1), _completed(_C1, _OK)), leftovers=1, turn="untracked",
                                         ack="in-flight"),
    **{f"untracked-in-flight-{exit}": _Walk((_started(_C1), _completed(_C1, outcome)), leftovers=1, turn="untracked",
                                            ack="in-flight", exit=exit)
       for exit, outcome in (("stop", _CANCELLED), ("refused", _FAILED), ("exception", _FAILED))},
    # The copy waits behind a queued follow-up, and the acknowledgement returns during the copy's turn.
    "untracked-late-ack-fifo-exception": _Walk(
        (_started(_C1), _started(_Q1), _completed(_Q1, _OK), _completed(_C1, _FAILED)), leftovers=1,
        turn="untracked", ack="late", placement="fifo", exit="exception"),
    # The running turn's hooks had already completed, so the correction completes with its handler.
    "hooks-done-leftover": _Walk((_S_OPEN, _completed(_OPENING, _OK), _started(_C1), _completed(_C1, _OK)), leftovers=1,
                                 turn="hooks-done"),
    # The queued lane runs no hooks for an adapter without its own start hook, so its handler
    # completes the correction.
    "complete-only-consumed": _Walk((_completed(_C1, _OK), _completed(_OPENING, _OK)), adapter="complete-only"),
    "complete-only-leftover": _Walk((_completed(_C1, _OK), _completed(_OPENING, _OK)), leftovers=1, adapter="complete-only"),
    "complete-only-queue": _Walk((_completed(_C1, _OK), _completed(_OPENING, _OK)), verb="queue", adapter="complete-only"),
}


async def _walk_lifecycle(monkeypatch, tmp_path, walk: _Walk) -> list:
    from gateway.run_turn_followup_ack import _turn_result_outcome

    run = _WalkRun(asyncio.get_running_loop(), consumed=0 if walk.leftovers else 99)
    _WalkModel.walk = run
    _WalkModel._supports_active_turn_redirect = walk.verb == "redirect"
    _install_fake_agent(monkeypatch, tmp_path, _WalkModel)
    # No interim messages, so no stream consumer: nothing in these routes streams, and a turn that
    # raises would otherwise wait out the consumer's 5 s flush bound.
    (tmp_path / "config.yaml").write_text("display:\n  interim_assistant_messages: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _adapter = _priority_runner(
        monkeypatch,
        {"redirect": "interrupt", "interrupt": "interrupt", "queue": "queue", "photo": "queue"}.get(walk.verb, "steer"))
    adapter = (_WalkCompleteOnlyAdapter if walk.adapter == "complete-only" else _WalkAdapter)(
        hold_ack=walk.ack != "returned")
    adapter.platform = Platform.SLACK
    runner.adapters[Platform.SLACK] = adapter
    adapter.set_message_handler(runner._handle_message)
    monkeypatch.setattr(adapter, "_requeue_backoff_delay", lambda *_args: 3600)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    key = runner._session_key_for_source(source)
    if walk.placement in {"cap", "cap-goal"}:
        runner._MAX_INTERRUPT_DEPTH = 0
    queued_ids = {"fifo": [_Q1], "overflow": [_Q1, _Q2], "orphan": [_Q1]}.get(walk.placement, [])
    copy_call = len(queued_ids) + 1
    if walk.exit == "exception":
        run.failing_call = 0 if not walk.leftovers and walk.verb != "interrupt" else copy_call
    copy_prepared, copy_release = asyncio.Event(), asyncio.Event()
    prepare = runner._prepare_profile_scoped_inbound_message_text

    async def gated_prepare(**kwargs):
        if kwargs["event"].media_urls:
            return kwargs["event"].text or "photo"
        if str(kwargs["event"].message_id).startswith("corr"):
            copy_prepared.set()
            await copy_release.wait()
            if walk.exit == "refused":
                return None
        return await prepare(**kwargs)

    monkeypatch.setattr(runner, "_prepare_profile_scoped_inbound_message_text", gated_prepare)
    opening = MessageEvent(text="opening", source=source, message_id=None if walk.turn == "untracked" else _OPENING)
    tracked = walk.turn != "untracked"

    async def opening_turn():
        """What _process_message_background does around the handler for the opening message."""
        if tracked:
            await adapter._run_processing_hook("on_processing_start", opening)
        try:
            result = await runner._run_agent(message="opening", context_prompt="", history=[], source=source,
                                             session_id="walk", session_key=key, processing_event=opening)
        except asyncio.CancelledError:
            if tracked:
                expected = asyncio.current_task() in adapter._expected_cancelled_tasks
                await adapter._run_processing_hook("on_processing_complete", opening, _CANCELLED if expected else _FAILED)
            raise
        except Exception:
            if tracked:
                await adapter._run_processing_hook("on_processing_complete", opening, _FAILED)
            return
        if tracked:
            await adapter._run_processing_hook("on_processing_complete", opening, _turn_result_outcome(result))

    async def teardown(chain):
        """Gateway shutdown or profile teardown: the adapter's tasks, including the one that runs the
        opening message, are cancelled while the adapter is still connected."""
        adapter._background_tasks.add(chain)
        await runner._bounded_adapter_teardown(adapter, Platform.SLACK)

    async def command(kind, chain):
        """/stop, /new or /reset while a turn runs: the runner's busy-command handler, then the
        adapter's cancellation of the task that runs the turn."""
        adapter._session_tasks[key] = chain
        before = set(adapter._background_tasks)
        event = MessageEvent(text=f"/{kind}", source=source, message_id=f"{kind}-1")
        handler = runner._busy_stop_command if kind == "stop" else runner._busy_new_command
        await handler(event, key, source)
        if not chain.done():
            await adapter.cancel_session_processing(key, discard_pending=False)
        await asyncio.gather(*(set(adapter._background_tasks) - before - {chain}), return_exceptions=True)

    chain = asyncio.create_task(opening_turn())
    try:
        await asyncio.wait_for(run.model_started[0].wait(), 30)

        async def registered():
            while not isinstance(runner._session_state(key).turn.agent, _WalkModel):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(registered(), 30)
        runner._session_state(key).turn.started_ts = time.time()
        if walk.turn == "hooks-done":
            await adapter._run_processing_hook("on_processing_complete", opening, _OK)
        if walk.verb == "photo" or walk.placement == "orphan":
            for queued_id in queued_ids:
                runner._enqueue_fifo(key, MessageEvent(text=queued_id, source=source, message_id=queued_id), adapter)
            queued_ids = []
        if walk.placement in {"goal", "notification"}:
            runner._enqueue_fifo(key, runner._synthetic_prompt_event(
                source, "continue the goal", internal=walk.placement == "notification"), adapter)
        for index in range(1, max(walk.leftovers, 1) + 1):
            text = f"correction {index}" if walk.ack == "returned" else f"/steer correction {index}"
            message = MessageEvent(text=text, source=source, message_id=f"corr-{index}")
            if walk.verb == "photo":
                message = MessageEvent(text="", message_type=MessageType.PHOTO, source=source, message_id=_P1,
                                       media_urls=["/nonexistent/photo-1.jpg"], media_types=["image/jpeg"])
            await adapter.handle_message(message)
            correction_task = adapter._session_tasks[key]
            if walk.ack != "returned":
                await asyncio.wait_for(adapter.ack_sending.wait(), 30)
            else:
                await asyncio.wait_for(correction_task, 30)
        for queued_id in queued_ids:
            runner._enqueue_fifo(key, MessageEvent(text=queued_id, source=source, message_id=queued_id), adapter)
        if walk.placement == "cap-goal":
            runner._enqueue_fifo(key, runner._synthetic_prompt_event(source, "continue the goal"), adapter)
        if walk.exit == "stop-turn" or walk.placement == "orphan":
            await command("stop", chain)
        elif walk.exit == "teardown-turn":
            await teardown(chain)
        run.model_release[0].set()
        if walk.ack == "in-flight":
            await asyncio.wait_for(copy_prepared.wait(), 30)
            adapter.ack_release.set()
            await asyncio.wait_for(correction_task, 30)
        if walk.exit == "teardown-queued":
            await asyncio.wait_for(run.model_started[1].wait(), 30)
            await teardown(chain)
        elif walk.exit.endswith("-queued"):
            await asyncio.wait_for(run.model_started[1].wait(), 30)
            await command(walk.exit.split("-")[0], chain)
        elif walk.exit in {"stop", "new", "reset"} and walk.placement not in {"cap", "cap-goal"}:
            for release in run.model_release[1:copy_call]:
                release.set()
            await asyncio.wait_for(copy_prepared.wait(), 30)
            await command(walk.exit, chain)
        copy_release.set()
        for index, release in enumerate(run.model_release[1:], 1):
            if walk.placement == "orphan":
                break
            if walk.ack == "late" and index == copy_call:
                await asyncio.wait_for(run.model_started[copy_call].wait(), 30)
                adapter.ack_release.set()
                await asyncio.wait_for(correction_task, 30)
            release.set()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(chain, 30)
        if walk.placement == "orphan":
            await adapter.handle_message(MessageEvent(text="next", source=source, message_id=_N1))
            next_task = adapter._session_tasks[key]
            await asyncio.wait_for(run.model_started[1].wait(), 30)
            if walk.exit == "stop-orphan":
                await command("stop", next_task)
            elif walk.exit == "stop-orphan-twice":
                released = runner._release_turn_lease

                def release_turn_lease(session_key, run_generation):
                    adapter.log.append(("turn-lease-released", _N1))
                    return released(session_key, run_generation)

                monkeypatch.setattr(runner, "_release_turn_lease", release_turn_lease)
                adapter.hold_completion = _C1
                await runner._busy_stop_command(MessageEvent(text="/stop", source=source), key, source)
                adapter._expected_cancelled_tasks.add(next_task)
                next_task.cancel()
                await asyncio.wait_for(adapter.completion_held.wait(), 30)
                next_task.cancel()
            for release in run.model_release[1:]:
                release.set()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(next_task, 30)
        if walk.placement in {"cap", "cap-goal"}:
            if walk.exit in {"stop", "new", "reset"}:
                await command(walk.exit, chain)
            else:
                async def slot_turn(_event):
                    if walk.exit == "exception":
                        raise RuntimeError("slot turn failed")
                    return "done"

                adapter.set_message_handler(slot_turn)
                await adapter._process_message_background(adapter._pending_messages.pop(key), key)
        return [entry for entry in adapter.log if entry[0] != "send"]
    finally:
        for release in run.model_release:
            release.set()
        copy_release.set()
        adapter.ack_release.set()
        chain.cancel()
        with suppress(BaseException):
            await chain
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("walk", list(_WALKS.values()), ids=list(_WALKS))
async def test_every_started_message_completes_once_with_the_turn_that_consumed_it(monkeypatch, tmp_path, walk):
    """Invariant: once the adapter has reported a message's start (👀), the message completes exactly
    once (✅, ❌ or cancelled), with the outcome of the turn that read it, FAILURE when that turn
    fails or its text is refused, or CANCELLED when /stop, /new or /reset discards it first. The
    walk covers steer, redirect, interrupt and queue on the priority path; one or two leftover
    steers; tracked and untracked running turns; an acknowledgement still in flight; the leftover
    copy run directly, queued behind other input or parked at the recursion cap; and an adapter
    without its own start hook."""
    log = await _walk_lifecycle(monkeypatch, tmp_path, walk)

    one_lifecycle = ["complete"] if walk.adapter == "complete-only" else ["start", "complete"]
    per_message = {}
    for entry in log:
        per_message.setdefault(entry[1], []).append(entry[0])
    assert (log, {m: kinds for m, kinds in per_message.items() if kinds != one_lifecycle}) == (
        list(walk.expected), {})


class _BlockingSendAdapter(LifecycleLogAdapter):
    """Holds the send of ``blocked_content`` until the test releases it."""

    def __init__(self, blocked_content):
        super().__init__()
        self.blocked_content = blocked_content
        self.send_started, self.send_release = asyncio.Event(), asyncio.Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content == self.blocked_content:
            self.send_started.set()
            await self.send_release.wait()
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_delivery", "queued_outcome"),
    [("delivered", ProcessingOutcome.SUCCESS), ("refused", ProcessingOutcome.FAILURE),
     ("cancelled", ProcessingOutcome.CANCELLED)],
)
@pytest.mark.parametrize("followup", ["platform-message", "hookless-with-photo"])
async def test_each_queued_turn_completes_after_its_own_reply(monkeypatch, tmp_path, final_delivery, queued_outcome, followup):
    """A turn completes once its reply is delivered, before the queued follow-up starts, so cancelling
    the follow-up cannot change the earlier outcome. The terminal follow-up's reply goes out through
    the adapter's final delivery, which decides that follow-up's outcome."""
    _ScriptedAgent.calls, _ScriptedAgent.results = [], [_done("done-1"), _done("done-2")]
    _install_fake_agent(monkeypatch, tmp_path, _ScriptedAgent)
    adapter = _BlockingSendAdapter("done-2")
    if final_delivery == "refused":
        adapter.refused.add("done-2")
    runner = _make_runner(adapter)
    hookless = followup == "hookless-with-photo"
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="follow-up", source=_source(), message_id=None if hookless else "queued-1")
    photo = MessageEvent(
        text="photo", source=_source(), message_id="queued-1", message_type=MessageType.PHOTO,
        media_urls=["/nonexistent/photo.jpg"], media_types=["image/jpeg"])
    if hookless:
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up and photo")

    async def respond(event):
        if hookless:
            await adapter._run_processing_hook("on_processing_start", photo)
            runner._merge_into_pending_slot(adapter, SESSION_KEY, photo)
        result = await runner._run_agent(
            message=event.text, context_prompt="", history=[], source=event.source,
            session_id="sess-lifecycle", session_key=SESSION_KEY, processing_event=event)
        return result["final_response"]

    adapter.set_message_handler(respond)
    await adapter.handle_message(MessageEvent(text="first", source=_source(), message_id="first-1"))
    await asyncio.wait_for(adapter.send_started.wait(), 5)
    before_delivery = [entry for entry in adapter.log if entry[:2] == ("complete", "queued-1")]
    if final_delivery == "cancelled":
        await adapter.cancel_session_processing(SESSION_KEY)
    else:
        adapter.send_release.set()
        await asyncio.gather(*adapter._background_tasks)

    opening = [("start", "first-1"), ("send", "done-1"),
               ("complete", "first-1", ProcessingOutcome.SUCCESS)]
    if hookless:
        opening.insert(1, ("start", "queued-1"))
    else:
        opening.append(("start", "queued-1"))
    terminal = [] if final_delivery == "cancelled" else [("send", "done-2")]
    if final_delivery == "refused":
        from agent.i18n import t

        terminal.append(("send", t("gateway.notify.plain_fallback_prefix", content="done-2")))
    expected = [*opening, *terminal, ("complete", "queued-1", queued_outcome)]
    assert (before_delivery, adapter.log) == ([], expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True])
async def test_hookless_followup_completes_at_its_own_delivery_before_its_successor(monkeypatch, tmp_path, refused):
    _ScriptedAgent.calls, _ScriptedAgent.results = [], [_done("done-1"), _done("done-2"), _done("done-3")]
    _install_fake_agent(monkeypatch, tmp_path, _ScriptedAgent)
    adapter = _BlockingSendAdapter("done-2")
    if refused:
        adapter.refused.add("done-2")
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(text="synthetic", source=_source())
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="synthetic and photo")
    runner._enqueue_fifo(SESSION_KEY, MessageEvent(text="successor", source=_source(), message_id="successor"), adapter)
    photo = MessageEvent(text="photo", source=_source(), message_id="photo", message_type=MessageType.PHOTO,
                         media_urls=["/nonexistent/photo.jpg"], media_types=["image/jpeg"])

    async def respond(event):
        await adapter._run_processing_hook("on_processing_start", photo)
        runner._merge_into_pending_slot(adapter, SESSION_KEY, photo)
        result = await runner._run_agent(message=event.text, context_prompt="", history=[], source=event.source,
                                        session_id="hookless-chain", session_key=SESSION_KEY, processing_event=event)
        return result["final_response"]

    adapter.set_message_handler(respond)
    await adapter.handle_message(MessageEvent(text="opening", source=_source(), message_id="opening"))
    await asyncio.wait_for(adapter.send_started.wait(), timeout=5)
    before_delivery = [entry for entry in adapter.log if entry[:2] == ("complete", "photo")]
    adapter.send_release.set()
    await asyncio.gather(*adapter._background_tasks)
    lifecycle = [entry for entry in adapter.log if entry[0] != "send"]
    assert (before_delivery, lifecycle) == ([], [
        ("start", "opening"), ("start", "photo"), ("complete", "opening", ProcessingOutcome.SUCCESS),
        ("complete", "photo", ProcessingOutcome.FAILURE if refused else ProcessingOutcome.SUCCESS),
        ("start", "successor"), ("complete", "successor", ProcessingOutcome.SUCCESS),
    ])



def _running_slack_turn(runner, *, finished):
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    receiver = MagicMock(_supports_active_turn_redirect=True, _active_children=[])
    receiver.steer.return_value = receiver.redirect.return_value = True
    receiver.get_activity_summary.return_value = {"seconds_since_activity": 0}
    running = MessageEvent(text="running", source=source, message_id="running-1")
    key = runner._session_key_for_source(source)
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.processing_event, turn.started_ts = receiver, running, running, time.time()
    turn.ctx = TurnContext(
        session_key=key, event_message_id="running-1", inbound_message_id="running-1",
        result_holder=[_done("reply") if finished else None])
    return source, key, receiver, running


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "text", "entry"),
    [("steer", "late", "busy"), ("interrupt", "late", "busy"), ("steer", "late", "priority"),
     ("interrupt", "late", "priority"), ("steer", "/steer late", "priority")],
)
async def test_input_after_the_model_result_waits_for_its_own_turn(monkeypatch, mode, text, entry):
    """Once the running turn has its model result, its agent cannot consume more input. Steering or
    redirecting would attach the message to a reply that never saw it, so the message waits in the
    FIFO and runs as its own turn."""
    runner, adapter = _priority_runner(monkeypatch, mode)
    source, key, receiver, _running = _running_slack_turn(runner, finished=True)
    late = MessageEvent(text=text, source=source, message_id="late-1")

    if entry == "busy":
        assert await runner._handle_active_session_busy_message(late, key) is True
    else:
        await runner._handle_message(late)
    queued = adapter._pending_messages.get(key)

    assert (
        [name for name, *_ in receiver.mock_calls if name in _AGENT_VERBS],
        (queued.message_id, queued.text),
    ) == ([], ("late-1", "late"))


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_first", [False, True])
async def test_a_leftover_steer_runs_as_its_own_message(monkeypatch, queued_first):
    """A steer that arrives after the agent's last tool batch comes back as ``pending_steer``. It runs
    as its own turn for the message that sent it, so that message's lifecycle hooks and reply anchor
    apply. A follow-up queued earlier runs first, and the steer waits behind it instead of being
    dropped."""
    runner, adapter = _priority_runner(monkeypatch, "steer")
    source, key, receiver, running = _running_slack_turn(runner, finished=False)
    queued = MessageEvent(text="queued", source=source, message_id="queued-1")
    if queued_first:
        runner._enqueue_fifo(key, queued, adapter)
    late = MessageEvent(text="late", source=source, message_id="late-1")
    assert await runner._handle_active_session_busy_message(late, key) is True
    (admitted,) = receiver.steer.call_args.args

    drained = [await runner._run_agent_drain_pending(
        {"final_response": "reply", "pending_steer": admitted}, adapter, source, key, processing_event=running)]
    if queued_first:
        drained.append(await runner._run_agent_drain_pending(
            {"final_response": "queued reply"}, adapter, source, key, processing_event=queued))

    assert drained == [(queued, "queued")] * queued_first + [(replace(late, text=admitted), admitted)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["failed-result", "handler-error"])
async def test_a_failed_turn_completes_with_its_delivery_outcome(monkeypatch, tmp_path, failure):
    """A failed turn still delivers a failure notice to the user. The completion outcome reports
    that delivery, as it does for every other reply."""
    runner, adapter = _priority_runner(monkeypatch, "queue")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    runner._run_agent = (
        AsyncMock(side_effect=RuntimeError("Provider failed")) if failure == "handler-error"
        else AsyncMock(return_value={"final_response": "Provider unavailable", "messages": [],
                                     "failed": True, "completed": False})
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")

    await adapter.handle_message(MessageEvent(text="first", source=source, message_id="first-1"))
    await asyncio.gather(*adapter._background_tasks)

    assert (
        [entry for entry in adapter.log if entry[0] != "send"],
        len([entry for entry in adapter.log if entry[0] == "send"]) >= 1,
    ) == ([("start", "first-1"), ("complete", "first-1", ProcessingOutcome.SUCCESS)], True)


class _BlockingStartAdapter(LifecycleLogAdapter):
    """The adapter's own start I/O (for example Slack reactions.add) for ``blocked_id`` never returns."""

    def __init__(self, blocked_id):
        super().__init__()
        self.blocked_id, self.in_start = blocked_id, asyncio.Event()

    async def on_processing_start(self, event):
        await super().on_processing_start(event)
        if event.message_id == self.blocked_id:
            self.in_start.set()
            await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_cancel_during_a_follow_ups_own_start_hook_completes_it(monkeypatch):
    """/stop can cancel the chain while the adapter is still reporting a queued follow-up's start. The
    follow-up has started, so it must complete."""
    runner, _adapter = _priority_runner(monkeypatch, "queue")
    adapter = _BlockingStartAdapter("queued-1")
    adapter.platform = Platform.SLACK
    runner.adapters[Platform.SLACK] = adapter
    source, key, _receiver, running = _running_slack_turn(runner, finished=False)
    await adapter._run_processing_hook("on_processing_start", running)
    runner._enqueue_fifo(key, MessageEvent(text="queued", source=source, message_id="queued-1"), adapter)
    result = {"final_response": "reply", "messages": []}
    pending_event, pending = await runner._run_agent_drain_pending(
        result, adapter, source, key, processing_event=running)
    monkeypatch.setattr(runner, "_run_agent_deliver_first_response", AsyncMock(return_value=True))
    turn_ctx = TurnContext(source=source, processing_event=running, session_key=key, session_id="s", history=[])
    chain = asyncio.create_task(runner._run_agent_queued_followup(
        turn_ctx, adapter, pending, pending_event, "reply", result, None))
    adapter._expected_cancelled_tasks.add(chain)
    await asyncio.wait_for(adapter.in_start.wait(), 30)

    chain.cancel()
    with suppress(asyncio.CancelledError):
        await chain

    assert adapter.log == [
        ("start", "running-1"), ("complete", "running-1", ProcessingOutcome.SUCCESS),
        ("start", "queued-1"), ("complete", "queued-1", ProcessingOutcome.CANCELLED),
    ]


class _StalledCompletionAdapter(LifecycleLogAdapter):
    """The platform call behind the completion hook (for example Slack reactions.remove on a
    half-dead connection) does not return. Records what teardown had done when the hook ran."""

    def __init__(self, drain_tasks, flushed):
        super().__init__()
        self.drain_tasks, self.flushed, self.seen = drain_tasks, flushed, []

    async def on_processing_complete(self, event, outcome):
        await super().on_processing_complete(event, outcome)
        self.seen.append(([task.done() for task in self.drain_tasks], list(self.flushed), dict(self._pending_messages)))
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["runner-teardown", "adapter-cancel"])
async def test_teardown_completes_parked_messages_after_cancelling_and_flushing(monkeypatch, entry):
    """A started message parked in the slot completes as CANCELLED at teardown, but only after the
    adapter's tasks are cancelled and the slot is flushed to disk. A completion hook that never
    returns then cannot stop the cancellation or the flush, and no running chain can start the
    message again after it completed."""
    import gateway.shutdown_pending as shutdown_pending

    runner, _adapter = _priority_runner(monkeypatch, "queue")
    drain_tasks, flushed = [], []
    adapter = _StalledCompletionAdapter(drain_tasks, flushed)
    adapter.platform = Platform.SLACK
    runner.adapters[Platform.SLACK] = adapter
    adapter.set_message_handler(runner._handle_message)
    monkeypatch.setattr(adapter, "_requeue_backoff_delay", lambda *_args: 3600)
    source, key, _receiver, _running = _running_slack_turn(runner, finished=False)
    await adapter.handle_message(MessageEvent(text="q1", source=source, message_id="queued-1"))
    await asyncio.wait_for(asyncio.shield(adapter._session_tasks[key]), 30)
    drain_tasks.extend(task for task in adapter._background_tasks if not task.done())
    write_snapshot = shutdown_pending._write_snapshot

    def record_snapshot(owner, session_key, events):
        write_snapshot(owner, session_key, events)
        flushed.extend(event.message_id for event in events)

    monkeypatch.setattr(shutdown_pending, "_write_snapshot", record_snapshot)
    monkeypatch.setattr(runner, "_adapter_disconnect_timeout_secs", lambda: 0.5)

    if entry == "runner-teardown":
        await runner._bounded_adapter_teardown(adapter, Platform.SLACK)
    else:
        teardown = asyncio.create_task(adapter.cancel_background_tasks())
        while not adapter.seen:
            await asyncio.sleep(0.01)
        teardown.cancel()
        with suppress(asyncio.CancelledError):
            await teardown

    assert (len(drain_tasks), adapter.seen[:1], [entry for entry in adapter.log if entry[1] == "queued-1"]) == (
        1, [([True], ["queued-1"], {})],
        [("start", "queued-1"), ("complete", "queued-1", ProcessingOutcome.CANCELLED)],
    )


@pytest.mark.asyncio
async def test_a_second_cancel_during_a_rescued_orphans_completion_still_releases_the_turn(monkeypatch, tmp_path):
    """/stop cancels the turn that runs a rescued orphan, and a second cancellation (another /stop, or
    teardown) lands while the orphan's completion hook awaits the platform. The turn's session slot
    and lease are released before that completion, so the second cancellation cannot skip them."""
    walk = _Walk((), verb="queue", placement="orphan", exit="stop-orphan-twice")

    log = await _walk_lifecycle(monkeypatch, tmp_path, walk)

    assert log == [*_CONSUMED, _completed(_OPENING, _CANCELLED), _started(_N1), _completed(_N1, _CANCELLED),
                   ("turn-lease-released", _N1), _completed(_C1, _CANCELLED)]


@pytest.mark.asyncio
async def test_a_busy_adapter_merge_that_replaces_a_started_parked_message_completes_it(monkeypatch):
    """The runner queued a started location message in the pending slot. A text message then reaches
    the busy adapter, whose own merge replaces an event that has no media and is not text, so the
    location message is dropped and completes as CANCELLED."""
    runner, adapter = _priority_runner(monkeypatch, "queue")
    adapter._requeue_backoff_delay = lambda *_args: 3600
    source, key, _receiver, _running = _running_slack_turn(runner, finished=False)
    location = MessageEvent(text="", message_type=MessageType.LOCATION, source=source, message_id="location-1")
    await adapter.handle_message(location)
    await asyncio.wait_for(asyncio.shield(adapter._session_tasks[key]), 30)
    parked, busy = adapter._pending_messages.get(key), key in adapter._active_sessions

    await adapter.handle_message(MessageEvent(text="text", source=source, message_id="text-1"))
    await asyncio.gather(*[task for task in adapter._background_tasks if task is not adapter._session_tasks.get(key)])

    assert (parked is location, busy, adapter._pending_messages[key].message_id,
            [entry for entry in adapter.log if entry[1] == "location-1"]) == (
        True, True, "text-1",
        [("start", "location-1"), ("complete", "location-1", ProcessingOutcome.CANCELLED)],
    )
    await adapter.cancel_background_tasks()


def _lifecycle(adapter, message_id):
    return [entry[0] if entry[0] == "start" else entry[2] for entry in adapter.log
            if entry[0] != "send" and entry[1] == message_id]


@pytest.mark.asyncio
async def test_teardown_that_overruns_its_bound_completes_a_parked_message_once(monkeypatch):
    """The task that runs a session is slow to unwind (its cancellation path awaits the platform), so
    the cancel step overruns its bound before it flushes and clears the slot. The completion step
    completes the parked message and takes it out of the slot, so the slow task finds nothing to
    run when it finally ends."""
    import gateway.shutdown_flush as shutdown_flush

    runner, adapter = _priority_runner(monkeypatch, "queue")
    monkeypatch.setattr(shutdown_flush, "flush_pending_to_file", lambda pending, reason: None)
    monkeypatch.setattr(runner, "_adapter_disconnect_timeout_secs", lambda: 0.5)
    entered, unwind = asyncio.Event(), asyncio.Event()

    async def handler(event):
        if event.message_id == "running-1":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await unwind.wait()
                raise
        return "done"

    adapter.set_message_handler(handler)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    key = runner._session_key_for_source(source)
    await adapter.handle_message(MessageEvent(text="running", source=source, message_id="running-1"))
    await asyncio.wait_for(entered.wait(), 30)
    owner = adapter._session_tasks[key]
    queued = MessageEvent(text="q1", source=source, message_id="queued-1")
    await adapter._run_processing_hook("on_processing_start", queued)
    runner._enqueue_fifo(key, queued, adapter)

    await runner._bounded_adapter_teardown(adapter, Platform.SLACK)
    unwind.set()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(owner, 30)
    await asyncio.gather(*[task for task in adapter._background_tasks if not task.done()], return_exceptions=True)

    assert (key in adapter._pending_messages, _lifecycle(adapter, "queued-1")) == (False, ["start", _CANCELLED])


@pytest.mark.asyncio
async def test_a_message_started_while_teardown_cancels_tasks_completes(monkeypatch):
    """Teardown waits for another chat's task to unwind while a new message reaches the idle adapter
    for a chat whose turn the runner owns. The adapter starts it and the runner parks it; teardown
    completes it as CANCELLED although it arrived after teardown began."""
    import gateway.shutdown_flush as shutdown_flush

    runner, adapter = _priority_runner(monkeypatch, "queue")
    monkeypatch.setattr(shutdown_flush, "flush_pending_to_file", lambda pending, reason: None)
    adapter._requeue_backoff_delay = lambda *_args: 3600
    source, key, _receiver, _running = _running_slack_turn(runner, finished=False)
    other = SessionSource(platform=Platform.SLACK, chat_id="C2", chat_type="dm", user_id="U2")
    entered, cancelled, unwind = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(event):
        if event.message_id == "other-1":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await unwind.wait()
                raise
        return await runner._handle_message(event)

    adapter.set_message_handler(handler)
    await adapter.handle_message(MessageEvent(text="other", source=other, message_id="other-1"))
    await asyncio.wait_for(entered.wait(), 30)
    teardown = asyncio.create_task(runner._bounded_adapter_teardown(adapter, Platform.SLACK))
    await asyncio.wait_for(cancelled.wait(), 30)
    await adapter.handle_message(MessageEvent(text="next", source=source, message_id="next-1"))
    while adapter._pending_messages.get(key) is None:
        await asyncio.sleep(0.01)
    unwind.set()
    await asyncio.wait_for(teardown, 30)

    assert _lifecycle(adapter, "next-1") == ["start", _CANCELLED]


@pytest.mark.asyncio
async def test_profile_removal_completes_its_started_messages_in_the_runners_fifo(monkeypatch, tmp_path):
    """Multiplex profile removal takes the profile's adapter out of the registry before tearing it
    down. Its started messages in the slot and in the runner's overflow FIFO complete as CANCELLED."""
    import weakref

    import gateway.shutdown_flush as shutdown_flush

    runner, adapter = _priority_runner(monkeypatch, "queue")
    monkeypatch.setattr(shutdown_flush, "flush_pending_to_file", lambda pending, reason: None)
    del runner.adapters[Platform.SLACK]
    runner._profile_adapters = {"work": {Platform.SLACK: adapter}}
    runner._served_profile_homes, runner._served_profile_signatures = {"work": tmp_path}, {}
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    setattr(source, "_transport_adapter_ref", weakref.ref(adapter))
    key = runner._session_key_for_source(source)
    for index in (1, 2):
        event = MessageEvent(text=f"q{index}", source=source, message_id=f"queued-{index}")
        await adapter._run_processing_hook("on_processing_start", event)
        runner._enqueue_fifo(key, event, adapter)

    await runner._unserve_profile("work", tmp_path)

    assert {m: _lifecycle(adapter, m) for m in ("queued-1", "queued-2")} == {
        "queued-1": ["start", _CANCELLED], "queued-2": ["start", _CANCELLED]}
