"""A turn a plugin injects vs the session's running background review.

A person's turn supersedes the review at once (foreground priority, #84423). A turn no person asked
for, bound with ``turn_waits_for_background_work``, first lets the running review finish, for a
bounded time that an interrupt of the session agent ends early; whatever still runs after it is
superseded the same way.

Real run handshake (``prepare_background_review_run`` / ``begin_request`` /
``finish_background_review_run``); the review fork is a stand-in that runs until released or
interrupted.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

from agent import background_review as br


class _Review:
    """The admitted review fork: runs until released or interrupted."""

    def __init__(self):
        self.stop = threading.Event()
        self.interrupted = False

    def interrupt(self, message=None):
        self.interrupted = True
        self.stop.set()


def _start_review(session):
    run = br.prepare_background_review_run(session)
    review, outcome = _Review(), []
    assert run is not None and run.begin_request(review)

    def worker():
        review.stop.wait(30)
        outcome.append("cancelled" if review.interrupted else "completed")
        br.finish_background_review_run(session, run)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return SimpleNamespace(run=run, review=review, outcome=outcome, thread=thread)


def _turn_starts(session, *, plugin_injected):
    """The first step of a live turn (``run_conversation``) on its own thread."""
    def start():
        if plugin_injected:
            with br.turn_waits_for_background_work():
                br.cancel_background_review_for_live_turn(session)
        else:
            br.cancel_background_review_for_live_turn(session)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    return thread


def _session():
    return SimpleNamespace(session_id="sess", _interrupt_requested=False)


def test_a_persons_turn_supersedes_the_running_review_at_once():
    session = _session()
    bg = _start_review(session)

    turn = _turn_starts(session, plugin_injected=False)
    turn.join(10)
    bg.thread.join(10)

    assert not turn.is_alive() and bg.outcome == ["cancelled"]


def test_a_plugin_injected_turn_lets_the_running_review_finish():
    session = _session()
    bg = _start_review(session)

    turn = _turn_starts(session, plugin_injected=True)
    turn.join(0.5)
    waiting = turn.is_alive() and not bg.run.cancel_requested.is_set()
    bg.review.stop.set()  # the review finishes on its own
    turn.join(10)
    bg.thread.join(10)

    assert waiting and not turn.is_alive()
    assert bg.outcome == ["completed"] and not bg.run.cancel_requested.is_set()


def test_the_wait_is_bounded_then_the_turn_supersedes_the_review(monkeypatch):
    monkeypatch.setattr(br, "_BACKGROUND_WORK_WAIT_SECONDS", 0.5)
    session = _session()
    bg = _start_review(session)

    turn = _turn_starts(session, plugin_injected=True)
    turn.join(10)
    bg.thread.join(10)

    assert not turn.is_alive() and bg.outcome == ["cancelled"]


def test_an_interrupt_of_the_session_agent_ends_the_wait_early():
    session = _session()
    bg = _start_review(session)

    turn = _turn_starts(session, plugin_injected=True)
    turn.join(0.5)
    waiting = turn.is_alive()
    session._interrupt_requested = True  # a person stopped or overtook the turn
    turn.join(10)
    bg.thread.join(10)

    assert waiting and not turn.is_alive() and bg.outcome == ["cancelled"]


def test_a_persons_turn_after_a_plugin_injected_one_on_the_same_thread_supersedes_at_once():
    """The wait belongs to the injected turn alone: the next turn the same worker starts is a
    person's again."""
    session, started = _session(), []

    def turns():
        with br.turn_waits_for_background_work():
            br.cancel_background_review_for_live_turn(session)  # the injected turn; nothing running
        started.append(_start_review(session))
        br.cancel_background_review_for_live_turn(session)  # a person's turn

    thread = threading.Thread(target=turns, daemon=True)
    thread.start()
    thread.join(10)
    started[0].thread.join(10)

    assert not thread.is_alive() and started[0].outcome == ["cancelled"]
