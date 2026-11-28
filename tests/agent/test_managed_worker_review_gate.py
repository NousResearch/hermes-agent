"""The managed-worker review gate and its drain.

`load_background_review_settings` used to refuse every finite turn; the managed-worker opt-in
narrows that refusal to "finite and not a worker that opted in", and the drain has to mean the
review *finished* rather than "its request phase returned".
"""

from __future__ import annotations

import threading
import time

import pytest

from agent import background_review as br
from gateway.session_finite import finite_turn_scope, managed_worker_turn_scope


@pytest.mark.parametrize(
    "finite, worker, opted_in, expected",
    [
        (True, True, True, True),     # a worker that opted in reviews its turn
        (True, True, False, False),   # a worker without the opt-in does not
        (True, False, True, False),   # chat -q / -z stays off even with the key set
        (False, False, True, True),   # an interactive session is unchanged
    ],
)
def test_gate(monkeypatch, finite, worker, opted_in, expected):
    monkeypatch.setattr(
        br, "_task_block", lambda cfg: {"enabled": True, "managed_workers": opted_in})
    with finite_turn_scope(finite, False), managed_worker_turn_scope(worker):
        assert br.load_background_review_settings()[0] is expected


def test_disabled_config_wins_everywhere(monkeypatch):
    """`enabled: false` outranks the worker opt-in: turning reviews off must mean off."""

    monkeypatch.setattr(
        br, "_task_block", lambda cfg: {"enabled": False, "managed_workers": True})
    with finite_turn_scope(True, False), managed_worker_turn_scope(True):
        assert br.load_background_review_settings()[0] is False


def test_config_failure_does_not_open_finite_turns(monkeypatch):
    """The fail-open path must not become a hole: unreadable config cannot opt a finite turn in."""

    def boom(*_a, **_k):
        raise RuntimeError("no config")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", boom)
    with finite_turn_scope(True, False):
        assert br.load_background_review_settings()[0] is False
    with finite_turn_scope(False, False):
        assert br.load_background_review_settings()[0] is True


class _Agent:
    """The two attributes the drain reads, nothing else."""

    _background_review_thread = None
    _background_review_run = None


def test_wait_joins_the_review_thread():
    agent = _Agent()
    finished = threading.Event()

    def work():
        time.sleep(0.05)
        finished.set()

    agent._background_review_thread = threading.Thread(target=work, daemon=True)
    agent._background_review_thread.start()

    assert br.wait_for_managed_worker_review(agent, timeout_s=5) is True
    assert finished.is_set(), "the drain returned before the review thread finished"


def test_wait_reports_a_review_that_outlives_the_timeout():
    agent = _Agent()
    release = threading.Event()
    agent._background_review_thread = threading.Thread(target=release.wait, daemon=True)
    agent._background_review_thread.start()
    try:
        assert br.wait_for_managed_worker_review(agent, timeout_s=0.05) is False
    finally:
        release.set()
        agent._background_review_thread.join(5)


def test_wait_is_a_no_op_when_nothing_is_in_flight():
    assert br.wait_for_managed_worker_review(_Agent(), timeout_s=0.01) is True
