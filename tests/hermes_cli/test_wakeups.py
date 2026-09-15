"""Tests for hermes_cli/wakeups.py — agent-scheduled one-shot session wakeups (schedule_wakeup tool)."""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def test_resolve_due_at_clamps_delay_and_rejects_past_when():
    from hermes_cli.wakeups import MAX_DELAY_SECONDS, MIN_DELAY_SECONDS, resolve_due_at

    now = 1_000_000.0
    due, err, note = resolve_due_at(delay="3", now=now)
    assert err is None and due == now + MIN_DELAY_SECONDS and "minimum" in note
    due, err, note = resolve_due_at(delay="30d", now=now)
    assert err is None and due == now + MAX_DELAY_SECONDS and "7-day" in note
    assert resolve_due_at(delay="1h30m", now=now)[0] == now + 5400
    assert resolve_due_at(when="2020-01-01T00:00:00Z", now=time.time())[1] == "`when` is not in the future"
    assert resolve_due_at(now=now)[1]  # neither given
    assert resolve_due_at(delay="5m", when="2030-01-01T00:00:00Z", now=now)[1]  # both given


def test_due_prompt_claims_once_and_abandon_rearms(hermes_home):
    """A due wakeup is removed from the persisted set BEFORE its turn runs (no double fire across
    pollers); a dispatch that never started a turn puts it back so it fires on the next idle poll."""
    from hermes_cli.wakeups import MAX_PER_SESSION, WakeupManager

    mgr = WakeupManager("wk-sid")
    now = time.time()
    wakeup, err, _ = mgr.schedule("re-check CI", delay="10s", reason="ci", now=now - 60)
    assert err is None and wakeup is not None
    later, _, _ = mgr.schedule("far future", delay="1d", now=now)
    assert WakeupManager("wk-sid").load().pending[0].id == wakeup.id  # sorted by due time

    prompt = mgr.due_prompt(now)
    assert prompt and "re-check CI" in prompt and wakeup.id in prompt and "ci" in prompt
    assert WakeupManager("wk-sid").due_prompt(now) is None  # claimed — a second poller sees nothing due
    assert [w.id for w in mgr.load().pending] == [later.id]

    assert mgr.abandon_fire() is True
    assert [w.id for w in mgr.load().pending] == [wakeup.id, later.id]
    assert mgr.abandon_fire() is False  # a claim re-arms exactly once

    assert mgr.cancel(wakeup.id) is True and mgr.cancel(wakeup.id) is False
    for _ in range(MAX_PER_SESSION - 1):
        assert mgr.schedule("x", delay="1h")[1] is None
    assert "maximum" in mgr.schedule("one too many", delay="1h")[1]
