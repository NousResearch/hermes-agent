"""A background review must announce itself when it starts.

The completion summary was always visible (the ``review.summary`` frame on clients,
``display.review.summary_cli`` on an idle CLI), but the fork's *start* was silent —
a review running under the next turns read as an unexplained latency spike. The
started line now rides the same ``background_review_callback`` rail as the summary.

Like the completion lines, it is gated by ``display.memory_notifications``
(``off`` suppresses both).

The fork itself is stubbed at ``_run_review_fork``: these tests assert on the
announcement, not on a live fork, and keep the real ``run_agent`` module out of
the import path (its bootstrap reads the live Hermes home, which the test
home-IO guard refuses).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg


def _fake_parent(**over) -> SimpleNamespace:
    """The minimal parent-agent surface _run_review_in_thread touches pre-fork."""
    base = dict(
        provider="openai",
        model="gpt-test",
        session_id="s1",
        platform="cli",
        _emit_auxiliary_failure=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
        background_review_callback=None,
        memory_notifications="on",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture()
def fork_stub(monkeypatch):
    """Neutralize the fork: the started line fires before it, and nothing after it runs."""
    monkeypatch.setattr(bg, "_run_review_fork", lambda *a, **k: None)


def _run(agent):
    bg._run_review_in_thread(agent, [{"role": "user", "content": "hi"}], "review please")


def test_started_line_reaches_the_background_review_callback(fork_stub):
    """The fork publishes its start on the same rail the completion summary uses."""
    delivered: list[str] = []
    agent = _fake_parent(background_review_callback=delivered.append)
    _run(agent)
    assert delivered, "started line must reach background_review_callback"
    text = delivered[0]
    assert "review" in text.lower()
    # i18n key resolved, not the bare key — guards the bundled locale catalogs.
    assert text != "display.review.started_callback"
    assert "started" in text.lower()


def test_started_line_reaches_cli_print_even_without_a_callback(fork_stub):
    printed: list[str] = []
    agent = _fake_parent(_safe_print=printed.append)
    _run(agent)
    assert any("started" in line.lower() for line in printed), (
        "idle CLI sessions (no client callback) still get the started line"
    )


def test_memory_notifications_off_suppresses_the_started_line(fork_stub):
    delivered: list[str] = []
    printed: list[str] = []
    agent = _fake_parent(
        background_review_callback=delivered.append,
        _safe_print=printed.append,
        memory_notifications="off",
    )
    _run(agent)
    assert delivered == []
    assert not any("started" in line.lower() for line in printed), (
        "display.memory_notifications: off gates the started line like the summary"
    )


def test_callback_raising_does_not_abort_the_review(fork_stub):
    """A broken client callback must not take the fork down with it."""
    def _boom(_message):
        raise RuntimeError("client callback is broken")

    printed: list[str] = []
    agent = _fake_parent(background_review_callback=_boom, _safe_print=printed.append)
    _run(agent)  # must not raise
    assert any("started" in line.lower() for line in printed)
