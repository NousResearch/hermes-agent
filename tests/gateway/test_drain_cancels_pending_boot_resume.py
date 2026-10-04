"""Deferred boot-resume turns must never be admitted DURING the next drain.

A boot auto-resume is SCHEDULED synchronously — ``_schedule_resume_pending_sessions``
claims the session slot with ``_AGENT_PENDING_SENTINEL`` and dispatches an internal
event — but the turn BODY runs off-loop on the runner's shared executor and can start
minutes later. Measured in production: scheduled at 10:02:45, turn bodies began at
10:12:36-46, which was 30s INTO a shutdown drain that had started at 10:12:05.

The drain then waited out its whole cap on them (``_drain_active_agents`` gates on
``len(self._running_agents)``, which COUNTS sentinels, while the drain summary reports
``active_at_start`` from ``_snapshot_running_agents()``, which EXCLUDES them — hence a
contradictory ``active_at_start=0, active_now=5``), interrupted them at the cap, skipped
the clean-shutdown marker, and the next boot resumed every affected session a SECOND
time under a re-derived ``shutdown_timeout`` reason.

The contract locked here:

(a) a boot resume SCHEDULED but not STARTED at shutdown is CANCELLED and RE-MARKED
    resumable under its ORIGINAL reason, so the next boot resumes it exactly once;
(b) a boot resume whose turn ALREADY STARTED is drained like normal work;
(c) a turn admitted while shutdown is in progress emits
    ``PHASE=drain_admission key=<key> reason=<why>``.

Hermetic and in-process: the real GatewayRunner methods are driven directly; no gateway
process is spawned.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace


import pytest

from gateway.drain_resume import (
    DISPOSITION_CANCEL,
    DISPOSITION_DRAIN,
    DISPOSITION_GONE,
    BootResumeRegistration,
    classify_boot_resume_at_shutdown,
    drain_admission_reason,
)
from gateway.run import _AGENT_PENDING_SENTINEL, GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner

_KEY = "agent:main:telegram:dm:123456:u1"


def _registration(reason: str = "restart_interrupted") -> BootResumeRegistration:
    return BootResumeRegistration(session_key=_KEY, resume_reason=reason, scheduled_at=0.0)


# ── pure predicate ────────────────────────────────────────────────────────


def test_classify_pending_sentinel_is_cancelled():
    """The incident shape: the slot still holds the sentinel => the turn never began."""
    assert classify_boot_resume_at_shutdown(
        _registration(),
        slot_value=_AGENT_PENDING_SENTINEL,
        pending_sentinel=_AGENT_PENDING_SENTINEL,
        task_done=False,
    ) == DISPOSITION_CANCEL


def test_classify_real_agent_is_drained():
    """A resume turn that genuinely started is ordinary in-flight work."""
    assert classify_boot_resume_at_shutdown(
        _registration(),
        slot_value=object(),
        pending_sentinel=_AGENT_PENDING_SENTINEL,
        task_done=False,
    ) == DISPOSITION_DRAIN


def test_classify_empty_slot_is_already_finished():
    assert classify_boot_resume_at_shutdown(
        _registration(),
        slot_value=None,
        pending_sentinel=_AGENT_PENDING_SENTINEL,
        task_done=True,
    ) == DISPOSITION_GONE


def test_drain_admission_reason_classifies_each_source():
    assert drain_admission_reason(is_boot_resume=True, is_internal=True) == "boot_resume"
    assert drain_admission_reason(is_boot_resume=False, is_internal=True) == "internal_event"
    assert drain_admission_reason(is_boot_resume=False, is_internal=False) == "user_message"


# ── runner-level contract ─────────────────────────────────────────────────


def _runner_with_registry():
    runner, adapter = make_restart_runner()
    marks: list[tuple[str, str]] = []

    def _mark(session_key, reason="restart_timeout"):
        marks.append((session_key, reason))
        return True

    # Patch the SYNC store the real AsyncSessionStore facade delegates to, so the
    # production async path (`await self.async_session_store.mark_resume_pending(...)`)
    # is genuinely exercised rather than stubbed out.
    runner.session_store.mark_resume_pending = _mark
    runner._pending_boot_resumes = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._persist_active_agents = lambda: None
    for name in (
        "_register_pending_boot_resume",
        "_clear_pending_boot_resume",
        "_session_has_pending_boot_resume",
        "_cancel_pending_boot_resumes_for_shutdown",
        "_log_drain_admission",
        "_session_state",
        "_peek_session_state",
        "_release_running_agent_state",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    return runner, adapter, marks


@pytest.mark.asyncio
async def test_pending_resume_is_cancelled_and_remarked_at_shutdown():
    """(a) scheduled-but-unstarted resume => cancelled, slot released, re-marked ONCE
    under the ORIGINAL reason (not a re-derived second-generation one)."""
    runner, _adapter, marks = _runner_with_registry()

    async def _never_runs():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_never_runs())
    # Exactly what _schedule_resume_pending_sessions does.
    runner._session_state(_KEY).turn.agent = _AGENT_PENDING_SENTINEL
    runner._register_pending_boot_resume(_KEY, "restart_interrupted", task)
    assert _KEY in runner._running_agents  # the drain WOULD have waited on this

    assert await runner._cancel_pending_boot_resumes_for_shutdown() == 1

    assert _KEY not in runner._running_agents          # drain no longer counts it
    assert marks == [(_KEY, "restart_interrupted")]     # original reason preserved
    assert not runner._session_has_pending_boot_resume(_KEY)
    # idempotent: a second shutdown pass must not double-mark
    assert await runner._cancel_pending_boot_resumes_for_shutdown() == 0
    assert marks == [(_KEY, "restart_interrupted")]
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_running_resume_turn_is_drained_not_cancelled():
    """(b) an ALREADY-RUNNING resume turn is left alone for the normal drain."""
    runner, _adapter, marks = _runner_with_registry()

    async def _still_running():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_still_running())
    runner._session_state(_KEY).turn.agent = _AGENT_PENDING_SENTINEL
    runner._register_pending_boot_resume(_KEY, "restart_interrupted", task)

    # The turn body starts: the chokepoint clears the registration, sentinel promoted.
    runner._clear_pending_boot_resume(_KEY)
    live_agent = SimpleNamespace(name="live-resume-agent")
    runner._session_state(_KEY).turn.agent = live_agent

    assert await runner._cancel_pending_boot_resumes_for_shutdown() == 0
    assert runner._running_agents.get(_KEY) is live_agent  # drain must wait on it
    assert marks == []                                     # shutdown_mark path owns it
    assert not task.cancelled()
    task.cancel()


@pytest.mark.asyncio
async def test_promoted_slot_wins_even_if_registration_was_not_cleared():
    """Belt-and-braces: a real agent in the slot is drained even when the registry
    entry survived (e.g. the turn chokepoint was bypassed)."""
    runner, _adapter, marks = _runner_with_registry()

    async def _running():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_running())
    runner._register_pending_boot_resume(_KEY, "shutdown_timeout", task)
    runner._session_state(_KEY).turn.agent = SimpleNamespace(name="agent")

    assert await runner._cancel_pending_boot_resumes_for_shutdown() == 0
    assert marks == []
    task.cancel()


def test_drain_admission_line_fires_for_turn_started_during_drain(caplog):
    """(c) the detector names a turn admitted while shutdown is in progress."""
    runner, _adapter, _marks = _runner_with_registry()
    runner._draining = True
    runner._register_pending_boot_resume(_KEY, "restart_interrupted", None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._log_drain_admission(_KEY, SimpleNamespace(internal=True))

    lines = [r.getMessage() for r in caplog.records if "PHASE=drain_admission" in r.getMessage()]
    assert len(lines) == 1, lines
    assert f"key={_KEY}" in lines[0]
    assert "reason=boot_resume" in lines[0]


def test_drain_admission_line_silent_when_not_draining(caplog):
    """No noise on the normal path — the line means 'shutdown was underway'."""
    runner, _adapter, _marks = _runner_with_registry()
    runner._draining = False

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._log_drain_admission(_KEY, SimpleNamespace(internal=True))

    assert not [r for r in caplog.records if "PHASE=drain_admission" in r.getMessage()]


def test_drain_admission_classifies_a_user_message(caplog):
    runner, _adapter, _marks = _runner_with_registry()
    runner._draining = True

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._log_drain_admission("agent:main:telegram:dm:999:u9", SimpleNamespace(internal=False))

    lines = [r.getMessage() for r in caplog.records if "PHASE=drain_admission" in r.getMessage()]
    assert len(lines) == 1
    assert "reason=user_message" in lines[0]


@pytest.mark.asyncio
async def test_cancelled_resume_is_recovered_exactly_once_on_next_boot():
    """The half that distinguishes DEFERRED from DROPPED: after cancel, the durable
    mark the boot path reads is present exactly once, with the original reason."""
    runner, _adapter, marks = _runner_with_registry()

    async def _never():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_never())
    runner._session_state(_KEY).turn.agent = _AGENT_PENDING_SENTINEL
    runner._register_pending_boot_resume(_KEY, "restart_interrupted", task)
    await runner._cancel_pending_boot_resumes_for_shutdown()

    assert len(marks) == 1
    key, reason = marks[0]
    assert key == _KEY
    # NOT re-derived as shutdown_timeout — that re-derivation is what made every
    # affected channel run the resume machinery a second time.
    assert reason == "restart_interrupted"
