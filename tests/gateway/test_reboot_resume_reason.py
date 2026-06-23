"""Tests for first-class ``reboot_interrupted`` startup auto-resume.

Background
----------
The gateway's startup auto-resume only schedules sessions whose
``resume_reason`` is in ``GatewayRunner._AUTO_RESUME_REASONS``.  A full
*machine reboot* (not just a gateway-process restart) is marked
``reboot_interrupted`` by the external safe-reboot tooling.  That reason was
NOT in the allow-list, so after a real reboot the affected session stayed
``resume_pending`` but was *silently dropped* from startup auto-resume — the
agent never woke on its own; the human had to send a message.

This module pins the fix:

* ``reboot_interrupted`` is recognized (AC-1) and the other three reasons are
  unchanged (INV-2).
* A ``reboot_interrupted`` ``resume_pending`` session IS scheduled at startup
  (AC-3, with teeth: the pre-fix allow-list would NOT schedule it).
* The recovery-note wording for a reboot reads "a machine reboot" (AC-2),
  asserted against the REAL source-of-truth helper (no drift-prone mirror).
* An unrecognized reason is still filtered out AND emits the observability
  warning (AC-4 / AC-11); a recognized reason is scheduled silently (AC-11b).
* N≥3 concurrent reboot sessions all schedule, each claiming its slot once
  (AC-8, thundering-herd correctness).
* A reboot entry aged to a realistic ~3 min post-reboot still schedules; one
  aged past the freshness window does not (AC-10, freshness adequacy).

These drive the REAL ``_schedule_resume_pending_sessions`` against a real
``SessionStore`` on a temp ``HERMES_HOME`` (tmp_path) — same harness as
``tests/gateway/test_restart_cascade.py``.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import (
    _AGENT_PENDING_SENTINEL,
    GatewayRunner,
    _resume_reason_phrase,
)
from gateway.session import SessionSource, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner


# ---------------------------------------------------------------------------
# Harness (mirrors tests/gateway/test_restart_cascade.py)
# ---------------------------------------------------------------------------


def _source(chat_id="123"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1")


def _store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _runner(tmp_path, monkeypatch):
    # Assert the temp HERMES_HOME is actually in effect and is NOT the real
    # store path, so a harness/env mismatch can't silently schedule against
    # the production session store (false-green guard, per spec Phase 1).
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    import gateway.run as _gr

    assert _gr._hermes_home == tmp_path, "temp HERMES_HOME not in effect"
    runner, _adapter = make_restart_runner()
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.session_store = _store(tmp_path)
    assert runner.session_store.sessions_dir == tmp_path
    return runner, _adapter


def _entry(runner, chat_id="123"):
    return runner.session_store.get_or_create_session(_source(chat_id))


# ---------------------------------------------------------------------------
# AC-1 / INV-2 — recognition (the allow-list is a strict superset)
# ---------------------------------------------------------------------------


def test_reboot_interrupted_in_auto_resume_reasons():
    assert "reboot_interrupted" in GatewayRunner._AUTO_RESUME_REASONS


def test_existing_reasons_unchanged_superset():
    # INV-2: the three pre-existing reasons remain recognized; the change is
    # purely additive (a strict superset of the old set).
    old = {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
    assert old <= GatewayRunner._AUTO_RESUME_REASONS
    assert GatewayRunner._AUTO_RESUME_REASONS == old | {"reboot_interrupted"}


# ---------------------------------------------------------------------------
# AC-2 — recovery-note wording (asserted on the real source-of-truth helper)
# ---------------------------------------------------------------------------


def test_reason_phrase_for_reboot():
    assert _resume_reason_phrase("reboot_interrupted") == "a machine reboot"


def test_reason_phrase_existing_unchanged():
    assert _resume_reason_phrase("restart_timeout") == "a gateway restart"
    assert _resume_reason_phrase("shutdown_timeout") == "a gateway shutdown"


def test_reason_phrase_unknown_and_none_fall_back():
    # restart_interrupted has no dedicated phrase -> generic, as before.
    assert _resume_reason_phrase("restart_interrupted") == "a gateway interruption"
    assert _resume_reason_phrase("garbage_reason") == "a gateway interruption"
    assert _resume_reason_phrase(None) == "a gateway interruption"


# ---------------------------------------------------------------------------
# AC-3 — a reboot session IS scheduled (with teeth: pre-fix would NOT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reboot_interrupted_session_is_scheduled(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    assert runner.session_store.mark_resume_pending(
        entry.session_key, "reboot_interrupted"
    )

    assert runner._schedule_resume_pending_sessions() == 1
    # The slot was claimed exactly once.
    assert runner._running_agents.get(entry.session_key) is _AGENT_PENDING_SENTINEL
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
async def test_pre_fix_allowlist_would_not_schedule_reboot(tmp_path, monkeypatch):
    """Teeth: with the OLD allow-list, the same fixture must NOT schedule.

    Proves the bug was real and the fix is load-bearing — not a vacuous green.
    """
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")

    # Simulate the pre-fix state by restoring the old frozenset on this instance.
    monkeypatch.setattr(
        runner,
        "_AUTO_RESUME_REASONS",
        frozenset({"restart_timeout", "shutdown_timeout", "restart_interrupted"}),
    )
    assert runner._schedule_resume_pending_sessions() == 0
    assert entry.session_key not in runner._running_agents


# ---------------------------------------------------------------------------
# AC-4 / AC-11 / AC-11b — observability: warn on UNKNOWN, silent on recognized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_reason_filtered_and_warns(tmp_path, monkeypatch, caplog):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "garbage_reason")

    import logging

    with caplog.at_level(logging.WARNING):
        assert runner._schedule_resume_pending_sessions() == 0

    assert entry.session_key not in runner._running_agents
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "Auto-resume skipped" in r.getMessage()
    ]
    assert warnings, "expected a warning for the unknown-reason drop"
    assert "garbage_reason" in warnings[0]
    assert entry.session_key in warnings[0]


@pytest.mark.asyncio
async def test_recognized_reason_scheduled_without_warning(tmp_path, monkeypatch, caplog):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")

    import logging

    with caplog.at_level(logging.WARNING):
        assert runner._schedule_resume_pending_sessions() == 1

    spam = [
        r.getMessage()
        for r in caplog.records
        if "Auto-resume skipped" in r.getMessage()
    ]
    assert not spam, "a recognized reason must not emit the drop warning"
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
async def test_structural_deferral_is_silent(tmp_path, monkeypatch, caplog):
    """A suspended session (hard wipe) is filtered SILENTLY — not a reason drop.

    Structural deferral (suspended / origin-None) must never trip the
    unknown-reason warning, even though its reason isn't in the allow-list.
    """
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")
    # Force the hard-wipe structural state.
    runner.session_store._entries[entry.session_key].suspended = True

    import logging

    with caplog.at_level(logging.WARNING):
        assert runner._schedule_resume_pending_sessions() == 0

    spam = [
        r.getMessage()
        for r in caplog.records
        if "Auto-resume skipped" in r.getMessage()
    ]
    assert not spam, "a structurally-deferred (suspended) session must be silent"


# ---------------------------------------------------------------------------
# AC-8 — multiplicity / thundering-herd: N reboot sessions all schedule once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_reboot_sessions_all_schedule_once(tmp_path, monkeypatch):
    runner, _adapter = _runner(tmp_path, monkeypatch)
    keys = []
    for i in range(5):
        e = _entry(runner, chat_id=f"chat-{i}")
        runner.session_store.mark_resume_pending(e.session_key, "reboot_interrupted")
        keys.append(e.session_key)

    scheduled = runner._schedule_resume_pending_sessions()
    assert scheduled == 5
    # Each session claimed its slot exactly once (no duplicate AIAgent).
    for k in keys:
        assert runner._running_agents.get(k) is _AGENT_PENDING_SENTINEL
    assert len(runner._resumed_this_boot) == 5
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
async def test_reboot_session_already_running_not_double_scheduled(tmp_path, monkeypatch):
    """A session whose agent is already running is not resumed a second time."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")
    # Pretend it's already in-flight.
    runner._running_agents[entry.session_key] = _AGENT_PENDING_SENTINEL

    assert runner._schedule_resume_pending_sessions() == 0


# ---------------------------------------------------------------------------
# AC-10 — freshness adequacy: realistic post-reboot age schedules; stale doesn't
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realistic_reboot_age_still_schedules(tmp_path, monkeypatch):
    """~3 min post-reboot (well inside the 3600s window) still auto-wakes."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")
    # Age the marker to ~3 minutes ago (a slow-ish real reboot).
    runner.session_store._entries[entry.session_key].last_resume_marked_at = (
        datetime.now() - timedelta(seconds=180)
    )

    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
async def test_stale_reboot_age_past_window_does_not_schedule(tmp_path, monkeypatch):
    """An overnight/multi-hour reboot past the freshness window does NOT
    auto-wake (by design, D-6) — the session stays resume_pending for the
    next inbound message."""
    runner, _adapter = _runner(tmp_path, monkeypatch)
    entry = _entry(runner)
    runner.session_store.mark_resume_pending(entry.session_key, "reboot_interrupted")
    # Age past the 3600s default window.
    runner.session_store._entries[entry.session_key].last_resume_marked_at = (
        datetime.now() - timedelta(seconds=4000)
    )

    assert runner._schedule_resume_pending_sessions() == 0
    # Still resume_pending (recoverable on next message), just not auto-woken.
    assert runner.session_store._entries[entry.session_key].resume_pending is True
