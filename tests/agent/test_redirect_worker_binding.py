"""The bits a redirect arms are bound to the tool worker that was running when it arrived.

``redirect()`` during tool execution queues the steer and asks every *live* tool worker to
yield, and the terminal backend turns that into a background adoption of the still-running
command. Worker tids are pooled and reused, so the yield (like the interrupt fan-out) must be
raised under the same tracker lock the worker's exit takes to discard its tid and clear its
bits. Snapshot-then-raise outside that lock races the exit:

* raised after the exit's clear -> a yield outlives its own command, and the NEXT command on the
  reused tid is handed to the background at its first poll with a note claiming "The user sent a
  message while this command was running" when no message exists (#144499's false hand-off);
* cleared after the next tool's registration -> a genuine yield for that tool is wiped, so the
  user's message waits behind the command after all (the turn looks stalled).

Each test below drives the exact interleaving with events. The bounded waits are only a window
for the racing side to finish (it expires when the lock holds it off); the assertions read the
bits themselves, never the clock.
"""

from __future__ import annotations

import threading

import pytest

import run_agent
from agent import interrupt_control
from agent.interrupt_control import InterruptControlMixin
from agent.tool_executor import _registered_tool_worker
from tools import interrupt as interrupt_mod


class _Agent(InterruptControlMixin):
    _executing_tools = True
    _interrupt_requested = False
    _pending_steer = None
    _pending_redirect = None
    api_mode = "chat_completions"
    quiet_mode = True

    def __init__(self):
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._execution_thread_id = None
        self._active_children = []
        self._active_children_lock = threading.Lock()


def _start_registered_worker(agent, ready, release, exited, tid_box):
    def body():
        with _registered_tool_worker(agent) as tid:
            tid_box.append(tid)
            ready.set()
            release.wait(5)
        exited.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    assert ready.wait(5), "tool worker never registered"
    return thread


def test_a_yield_armed_after_its_worker_finished_never_reaches_the_reused_tid(monkeypatch):
    """The reported false hand-off: redirect() snapshotted a live tid, the worker exited (and
    cleared its bits), and only then was the yield raised. It must not survive on that tid."""
    agent = _Agent()
    ready, release, exited = threading.Event(), threading.Event(), threading.Event()
    tid_box: list[int] = []
    worker = _start_registered_worker(agent, ready, release, exited, tid_box)
    tid = tid_box[0]

    real_request_yield = interrupt_control._request_yield

    def arm_after_the_worker_exited(target_tid):
        assert target_tid == tid
        release.set()
        # Base code holds no lock here, so the exit wins the race and this arms a dead tid.
        # Fixed code holds the tracker lock, so the exit cannot finish until we return.
        exited.wait(0.5)
        real_request_yield(target_tid)

    monkeypatch.setattr(interrupt_control, "_request_yield", arm_after_the_worker_exited)

    result = {}
    redirector = threading.Thread(
        target=lambda: result.setdefault("accepted", agent.redirect("继续执行")), daemon=True)
    redirector.start()
    redirector.join(5)
    worker.join(5)

    assert result.get("accepted") is True
    assert not worker.is_alive(), "the tool worker's exit deadlocked on the redirect"
    assert not interrupt_mod.is_thread_yield_requested(tid), (
        "a yield armed after its tool worker finished must not survive on the pooled tid: the "
        "next foreground command on it would be adopted as background at its first poll and told "
        "a user message arrived when none did")


def test_an_interrupt_armed_after_its_worker_finished_never_reaches_the_reused_tid(monkeypatch):
    """Same lifetime rule for the interrupt fan-out: a stale bit would make the next tool on the
    reused tid start out 'interrupted' and be killed before it runs."""
    agent = _Agent()
    ready, release, exited = threading.Event(), threading.Event(), threading.Event()
    tid_box: list[int] = []
    worker = _start_registered_worker(agent, ready, release, exited, tid_box)
    tid = tid_box[0]

    real_set_interrupt = interrupt_control._set_interrupt

    def raise_after_the_worker_exited(active, thread_id=None, **kwargs):
        if active and thread_id == tid:
            release.set()
            exited.wait(0.5)
        return real_set_interrupt(active, thread_id, **kwargs)

    monkeypatch.setattr(interrupt_control, "_set_interrupt", raise_after_the_worker_exited)

    stopping = threading.Thread(target=lambda: agent.interrupt("继续执行"), daemon=True)
    stopping.start()
    stopping.join(5)
    worker.join(5)

    assert not worker.is_alive(), "the tool worker's exit deadlocked on the interrupt fan-out"
    assert not interrupt_mod.is_thread_interrupted(tid), (
        "an interrupt armed after its tool worker finished must not survive on the pooled tid: "
        "the next tool there would start already interrupted")


def test_an_exiting_worker_cannot_clear_a_yield_armed_for_the_next_tool_on_its_tid(monkeypatch):
    """The lost-yield direction: the exit's bit clear used to run outside the tracker lock, so a
    redirect that armed the next tool on the reused tid had its genuine yield wiped, parking the
    user's message behind that tool. The next tool's yield must survive the previous exit."""
    agent = _Agent()
    entered = threading.Event()
    armed = threading.Event()
    tid_box: list[int] = []

    real_set_interrupt = run_agent._set_interrupt

    def delayed_clear(active, thread_id=None, **kwargs):
        if active is False:
            entered.set()
            # Fixed code clears inside the tracker lock, so the reuse below cannot arm until this
            # returns; base code clears outside it, so the arm lands first and is wiped here.
            armed.wait(0.5)
        return real_set_interrupt(active, thread_id, **kwargs)

    monkeypatch.setattr(run_agent, "_set_interrupt", delayed_clear)

    def worker_body():
        with _registered_tool_worker(agent) as tid:
            tid_box.append(tid)

    worker = threading.Thread(target=worker_body, daemon=True)
    worker.start()
    assert entered.wait(5), "the worker's exit clear was never reached"

    def reuse_the_tid_for_the_next_tool():
        # The pool reuses the tid: the next tool registers, then a redirect arms its yield.
        with agent._tool_worker_threads_lock:
            agent._tool_worker_threads.add(tid_box[0])
        agent.redirect("a follow-up while the next tool starts")
        armed.set()

    reuse = threading.Thread(target=reuse_the_tid_for_the_next_tool, daemon=True)
    reuse.start()
    reuse.join(5)
    worker.join(5)

    assert armed.is_set(), "the next tool's redirect never completed"
    assert interrupt_mod.is_thread_yield_requested(tid_box[0]), (
        "the previous tool's exit clear wiped the next tool's yield, so a genuine mid-command "
        "message would wait behind that command")


@pytest.mark.parametrize("tid_kind", ["yield", "interrupt"])
def test_the_worker_still_owns_its_own_bit_while_it_runs(tid_kind):
    """Control: the lock scoping must not stop a live worker from being armed at all."""
    agent = _Agent()
    ready, release = threading.Event(), threading.Event()
    tid_box: list[int] = []

    def body():
        with _registered_tool_worker(agent) as tid:
            tid_box.append(tid)
            ready.set()
            release.wait(5)

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    assert ready.wait(5)
    tid = tid_box[0]
    try:
        if tid_kind == "yield":
            assert agent.redirect("look here") is True
            assert interrupt_mod.is_thread_yield_requested(tid)
        else:
            agent.interrupt("stop")
            assert interrupt_mod.is_thread_interrupted(tid)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
