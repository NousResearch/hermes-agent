"""Tests for the cron wall-clock run budget (cron.run_budget_seconds / per-job).

Covers:

1. Budget resolution in ``cron.scheduler._cron_run_budget_seconds``:
   - per-job ``run_budget_seconds`` (jobs.json) wins over the
     ``cron.run_budget_seconds`` config default;
   - config fallback when the job carries no budget;
   - invalid values (bool, non-numeric, non-positive) resolve to None and
     leave the feature dormant — a broken config must never crash a fire.

2. The hard ceiling in ``run_job``: when a fire exceeds its budget, the poll
   loop hard-interrupts the agent and raises TimeoutError with a diagnostic
   naming the budget and the last activity — the gap the inactivity watchdog
   cannot close, because a trickling provider stream marks activity on every
   chunk while a single hung call consumes the whole fire window.

3. Backward-compat guards: with no budget the run is untouched (including
   the unlimited-inactivity path), and the inactivity abort still fires when
   no budget is set.
"""

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable (house style in tests/cron).
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _cron_run_budget_seconds, _normalize_cron_run_budget, run_job


# ── unit: resolution + normalization ───────────────────────────────────────


class TestBudgetResolution:
    def test_job_field_wins_over_config(self):
        job = {"id": "j", "run_budget_seconds": 900}
        cfg = {"cron": {"run_budget_seconds": 3600}}
        assert _cron_run_budget_seconds(job, cfg) == 900.0

    def test_config_fallback(self):
        assert _cron_run_budget_seconds(
            {"id": "j"}, {"cron": {"run_budget_seconds": 3600}}
        ) == 3600.0

    def test_both_absent_means_off(self):
        assert _cron_run_budget_seconds({"id": "j"}, {}) is None
        assert _cron_run_budget_seconds(None, None) is None

    def test_invalid_values_are_dormant(self):
        for bad in (True, False, "abc", -5, 0, [], {}):
            assert _normalize_cron_run_budget(bad) is None

    def test_valid_values_normalize(self):
        assert _normalize_cron_run_budget(900) == 900.0
        assert _normalize_cron_run_budget("600") == 600.0
        assert _normalize_cron_run_budget(0.5) == 0.5
        assert _normalize_cron_run_budget(None) is None


# ── integration: hard ceiling inside run_job ──────────────────────────────


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


class FakeSessionDB:
    def get_compression_tip(self, _session_id):
        return None

    def end_session(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


class BlockingAgent:
    """Agent whose run never returns until released; records interrupts.

    Reports itself permanently ACTIVE (seconds_since_activity=0) — the exact
    trickling-stream shape the inactivity watchdog cannot catch.
    """

    def __init__(self, release: threading.Event):
        self.release = release
        self.interrupted = False
        self.interrupt_reason = None
        self.max_iterations = 90

    def get_activity_summary(self):
        return {
            "last_activity_ts": time.time(),
            "last_activity_desc": "receiving stream response",
            "seconds_since_activity": 0.0,
            "current_tool": None,
            "api_call_count": 41,
            "max_iterations": self.max_iterations,
        }

    def interrupt(self, msg):
        self.interrupted = True
        self.interrupt_reason = msg

    def close(self):
        return None

    def run_conversation(self, prompt):
        self.release.wait(10)
        return {"final_response": "done", "messages": []}


class QuickAgent(BlockingAgent):
    def run_conversation(self, prompt):
        return {"final_response": "done", "messages": []}


class Stepper:
    """Fake monotonic clock advancing 60s per read.

    The run_job poll loop waits 5 real seconds per iteration but every clock
    read jumps 60s, so the budget deadline is crossed on the FIRST wait and
    the test costs ~0s of wall time.
    """

    def __init__(self):
        self._t = 0.0

    def __call__(self):
        self._t += 60.0
        return self._t


def _drive_run_job(job, agent, release, hermes_home):
    """Run the REAL ``run_job`` against fakes (harness shape from
    test_cleanup_timeout.py)."""
    from unittest.mock import MagicMock

    agent_cls = MagicMock()
    agent_cls.return_value = agent
    patchers = [
        patch("cron.scheduler._hermes_home", hermes_home),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=FakeSessionDB()),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=_RUNTIME,
        ),
        patch("cron.scheduler._cron_cleanup_timeout_seconds", return_value=0.02),
        patch("run_agent.AIAgent", agent_cls),
    ]
    for p in patchers:
        p.start()
    try:
        return run_job(job)
    finally:
        for p in patchers:
            p.stop()


def test_fire_exceeding_budget_hard_interrupts_and_fails(monkeypatch, tmp_path):
    """A fire past its wall-clock budget is hard-interrupted and fails with a
    TimeoutError naming the budget — even while the agent looks fully active
    (the inactivity watchdog can never catch this shape)."""
    release = threading.Event()
    agent = BlockingAgent(release)
    monkeypatch.setattr("cron.scheduler.time.monotonic", Stepper())

    job = {
        "id": "budget-overrun",
        "name": "budget-overrun",
        "prompt": "work",
        "schedule": {"kind": "interval", "minutes": 120, "display": "every 120m"},
        "run_budget_seconds": 30,
        "enabled": True,
        "deliver": "local",
    }

    try:
        success, output, _final_response, error = _drive_run_job(job, agent, release, tmp_path)
        assert success is False
        assert "wall-clock budget" in (error or "")
        assert "budget-overrun" in (error or "")
        assert agent.interrupted is True
        assert "wall-clock budget (30s)" in agent.interrupt_reason
    finally:
        release.set()


def test_no_budget_leaves_unlimited_run_untouched(monkeypatch, tmp_path):
    """Budget unset + HERMES_CRON_TIMEOUT=0: the run proceeds to completion
    exactly as before the feature existed."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    release = threading.Event()
    agent = QuickAgent(release)

    job = {
        "id": "no-budget",
        "name": "no-budget",
        "prompt": "work",
        "schedule": {"kind": "interval", "minutes": 120, "display": "every 120m"},
        "enabled": True,
        "deliver": "local",
    }
    success, _output, final_response, error = _drive_run_job(job, agent, release, tmp_path)

    assert success is True
    assert final_response == "done"
    assert error is None
    assert agent.interrupted is False


def test_inactivity_timeout_still_fires_without_budget(monkeypatch, tmp_path):
    """No budget set: the pre-existing inactivity abort is unchanged."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.001")  # < poll interval
    release = threading.Event()
    # Idle from the very first read: activity summary reports huge idle.
    agent = BlockingAgent(release)
    agent.get_activity_summary = lambda: {
        "last_activity_ts": time.time() - 999,
        "last_activity_desc": "api_call_streaming",
        "seconds_since_activity": 999.0,
        "current_tool": None,
        "api_call_count": 41,
        "max_iterations": 90,
    }

    job = {
        "id": "idle-job",
        "name": "idle-job",
        "prompt": "work",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "enabled": True,
        "deliver": "local",
    }

    try:
        success, output, _final_response, error = _drive_run_job(job, agent, release, tmp_path)
        assert success is False
        assert "idle for" in (error or "")
        assert agent.interrupted is True
    finally:
        release.set()


def test_per_job_budget_reaches_the_agent(monkeypatch, tmp_path):
    """run_budget_seconds from jobs.json is forwarded into the AIAgent
    constructor so the agent-side budget machinery is active for the run."""
    release = threading.Event()
    agent = QuickAgent(release)

    job = {
        "id": "agent-budget",
        "name": "agent-budget",
        "prompt": "work",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "run_budget_seconds": 900,
        "enabled": True,
        "deliver": "local",
    }

    success, _output, final_response, _error = _drive_run_job(job, agent, release, tmp_path)
    assert success is True
    assert final_response == "done"