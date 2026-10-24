"""A review fork that never acknowledges its cancel: the live turn fails loud and escalates, never
fails open. Split from ``test_background_review_compute_overlap.py`` (same agent double and
admission-registry reset)."""

from __future__ import annotations

import threading
import types

import pytest

from agent import background_review as background_review_module
from agent import review_admission
from tests.agent.test_background_review_compute_overlap import (
    _bare_agent,
    _reset_review_admission_state,
)


@pytest.fixture(autouse=True)
def clear_review_admission_state():
    _reset_review_admission_state()
    yield
    _reset_review_admission_state()


def test_unacknowledged_cancel_escalates_without_failing_open(monkeypatch, caplog):
    """The foreground's wait has no fail-open, so a fork wedged in a stalled stream or a
    blocking tool must be unwedged deterministically: after the first bound the waiter logs the
    owner and tears down the fork's clients; after the second it revokes every further provider
    call; and it still returns only once the fork publishes its request exit."""
    from agent.conversation_loop import (
        _reserve_review_input_request,
        _review_input_budget_exhausted,
    )

    monkeypatch.setattr(
        background_review_module, "_CANCEL_ACK_ESCALATION_SECONDS", 0.05, raising=False
    )
    agent = _bare_agent("wedged-review")
    profile_key = review_admission.current_profile_key()
    released_clients = threading.Event()

    class WedgedFork:
        # /refine: no aggregate budget to fall back on.
        _review_input_token_budget = None
        session_prompt_tokens = 0

        def hard_interrupt(self, *_args, **_kwargs):
            pass  # the abort hook is stuck behind the wedge

        def release_clients(self):
            released_clients.set()

    fork = WedgedFork()
    run = background_review_module.prepare_background_review_run(
        agent, session_id=agent.session_id, profile_key=profile_key
    )
    assert run is not None
    assert run.begin_request(fork) is True
    waiter_done = threading.Event()

    def foreground():
        background_review_module.cancel_background_review_for_live_turn(
            agent, session_id=agent.session_id, profile_key=profile_key
        )
        waiter_done.set()

    waiter = threading.Thread(target=foreground, daemon=True)
    with caplog.at_level("WARNING"):
        waiter.start()
        try:
            assert released_clients.wait(timeout=10.0)
            deadline = threading.Event()
            # Bounded poll for the second stage (no timing assertion).
            for _ in range(200):
                if getattr(fork, "_review_revoked", False):
                    break
                deadline.wait(0.05)
            assert getattr(fork, "_review_revoked", False) is True
            # The exit was never published: the waiter is still parked, never in provider work.
            assert waiter_done.is_set() is False
            assert _reserve_review_input_request(fork, 10) is False
            assert _review_input_budget_exhausted(fork) is True
        finally:
            background_review_module.finish_background_review_run(agent, run)
            waiter.join(timeout=10.0)

    assert waiter_done.is_set()
    owner = review_admission.owner_tag(profile_key, agent.session_id)
    for slug in (
        review_admission.REASON_CANCEL_UNACKNOWLEDGED,
        review_admission.REASON_REVIEW_REVOKED,
    ):
        lines = [r.getMessage() for r in caplog.records if slug in r.getMessage()]
        assert len(lines) == 1, (slug, caplog.text)
        assert owner in lines[0]
        assert agent.session_id not in lines[0]


def test_cancel_thread_start_failure_retries_the_interrupt_inline(monkeypatch):
    """Thread exhaustion must not leave an admitted fork un-fenced."""
    interrupted = []

    class Fork:
        def hard_interrupt(self, message=None, *, tool_reason=None):
            interrupted.append(tool_reason)

    class DeadThread:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        background_review_module,
        "threading",
        types.SimpleNamespace(
            Thread=DeadThread, Event=threading.Event, Lock=threading.Lock
        ),
    )

    background_review_module._interrupt_background_review(Fork())

    assert interrupted == ["background review superseded"]
