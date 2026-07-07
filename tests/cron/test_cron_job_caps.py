"""Tests for per-job execution caps: ``max_turns`` and ``timeout``.

Covers:

* ``create_job`` storing/normalizing ``max_turns`` (turn ceiling) and
  ``timeout`` (wall-clock seconds ceiling).
* ``update_job`` setting and clearing both fields.
* ``cronjob(action='create'/'update')`` tool-level plumbing.
* ``scheduler._resolve_job_max_iterations`` precedence: job field >
  config.yaml ``agent.max_turns`` > top-level ``max_turns`` > 90.
* ``scheduler._resolve_job_wall_clock_limit`` validation.
* Wall-clock watcher: an agent that stays *active* (retry storms touch the
  activity tracker, so the inactivity watcher never fires) is still
  interrupted once the job's wall-clock cap elapses.
"""

from __future__ import annotations

import concurrent.futures
import json
import time

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: data-layer semantics
# ---------------------------------------------------------------------------

def test_create_job_stores_max_turns_and_timeout(hermes_env):
    from cron.jobs import create_job

    job = create_job(
        prompt="say hi",
        schedule="every 5m",
        deliver="local",
        max_turns=30,
        timeout=900,
    )
    assert job["max_turns"] == 30
    assert job["timeout"] == 900.0


def test_create_job_defaults_have_no_caps(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="say hi", schedule="every 5m", deliver="local")
    assert job["max_turns"] is None
    assert job["timeout"] is None


@pytest.mark.parametrize("bad", [0, -5, "abc", 2.5, False])
def test_create_job_invalid_max_turns_normalizes_to_none(hermes_env, bad):
    from cron.jobs import create_job

    job = create_job(
        prompt="say hi", schedule="every 5m", deliver="local", max_turns=bad
    )
    assert job["max_turns"] is None


@pytest.mark.parametrize("bad", [0, -1, "abc", False])
def test_create_job_invalid_timeout_normalizes_to_none(hermes_env, bad):
    from cron.jobs import create_job

    job = create_job(
        prompt="say hi", schedule="every 5m", deliver="local", timeout=bad
    )
    assert job["timeout"] is None


def test_update_job_sets_and_clears_caps(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(prompt="say hi", schedule="every 5m", deliver="local")

    update_job(job["id"], {"max_turns": 25, "timeout": 600})
    updated = get_job(job["id"])
    assert updated["max_turns"] == 25
    assert updated["timeout"] == 600.0

    # 0 clears the cap (back to global defaults).
    update_job(job["id"], {"max_turns": 0, "timeout": 0})
    cleared = get_job(job["id"])
    assert cleared["max_turns"] is None
    assert cleared["timeout"] is None


# ---------------------------------------------------------------------------
# cronjob tool plumbing
# ---------------------------------------------------------------------------

def test_cronjob_create_and_update_plumb_caps(hermes_env):
    from tools.cronjob_tools import cronjob
    from cron.jobs import get_job

    out = json.loads(
        cronjob(
            action="create",
            prompt="say hi",
            schedule="every 5m",
            deliver="local",
            max_turns=30,
            timeout=900,
        )
    )
    assert out["success"] is True
    job = get_job(out["job_id"])
    assert job["max_turns"] == 30
    assert job["timeout"] == 900.0

    cronjob(action="update", job_id=out["job_id"], max_turns=25, timeout=600)
    job = get_job(out["job_id"])
    assert job["max_turns"] == 25
    assert job["timeout"] == 600.0


def test_cronjob_schema_exposes_caps():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert props["max_turns"]["type"] == "integer"
    assert props["timeout"]["type"] == "number"


# ---------------------------------------------------------------------------
# scheduler: per-job max_iterations resolution
# ---------------------------------------------------------------------------

def test_resolve_job_max_iterations_prefers_job_field(hermes_env):
    from cron.scheduler import _resolve_job_max_iterations

    cfg = {"agent": {"max_turns": 60}, "max_turns": 50}
    assert _resolve_job_max_iterations({"max_turns": 30}, cfg) == 30


def test_resolve_job_max_iterations_falls_back_to_config(hermes_env):
    from cron.scheduler import _resolve_job_max_iterations

    assert _resolve_job_max_iterations({}, {"agent": {"max_turns": 60}}) == 60
    assert _resolve_job_max_iterations({}, {"max_turns": 50}) == 50
    assert _resolve_job_max_iterations({}, {}) == 90


@pytest.mark.parametrize("bad", [0, -5, "abc", None])
def test_resolve_job_max_iterations_ignores_invalid_job_value(hermes_env, bad):
    from cron.scheduler import _resolve_job_max_iterations

    assert _resolve_job_max_iterations({"max_turns": bad}, {}) == 90


# ---------------------------------------------------------------------------
# scheduler: per-job wall-clock limit resolution
# ---------------------------------------------------------------------------

def test_resolve_job_wall_clock_limit(hermes_env):
    from cron.scheduler import _resolve_job_wall_clock_limit

    assert _resolve_job_wall_clock_limit({"timeout": 900}) == 900.0
    assert _resolve_job_wall_clock_limit({"timeout": 0}) is None
    assert _resolve_job_wall_clock_limit({"timeout": "abc"}) is None
    assert _resolve_job_wall_clock_limit({}) is None


# ---------------------------------------------------------------------------
# Wall-clock watcher: active agent past the cap is interrupted
# ---------------------------------------------------------------------------

class ActiveFakeAgent:
    """Agent that never goes idle (simulates a retry storm: every retry
    touches the activity tracker) but runs longer than the wall-clock cap."""

    def __init__(self, run_duration=5.0):
        self._run_duration = run_duration
        self._interrupted = False
        self._interrupt_msg = None

    def get_activity_summary(self):
        return {
            "last_activity_ts": time.time(),
            "last_activity_desc": "API error recovery (attempt 2/3)",
            "seconds_since_activity": 0.0,
            "current_tool": None,
            "api_call_count": 5,
            "max_iterations": 90,
        }

    def interrupt(self, msg):
        self._interrupted = True
        self._interrupt_msg = msg

    def run_conversation(self, prompt):
        time.sleep(self._run_duration)
        return {"final_response": "Done", "messages": []}


def test_active_agent_past_wall_clock_cap_is_interrupted():
    """Mirrors the scheduler watcher loop with a per-job wall-clock limit:
    activity alone must not keep a job alive past its wall-clock cap."""
    agent = ActiveFakeAgent(run_duration=5.0)
    _cron_inactivity_limit = 10.0   # never reached — agent is always active
    _wall_clock_limit = 0.3          # job-level cap
    _POLL_INTERVAL = 0.05

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _run_started_at = time.monotonic()
    future = pool.submit(agent.run_conversation, "test prompt")
    _inactivity_timeout = False
    _wall_clock_timeout = False

    result = None
    while True:
        done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
        if done:
            result = future.result()
            break
        if (
            _wall_clock_limit is not None
            and time.monotonic() - _run_started_at >= _wall_clock_limit
        ):
            _wall_clock_timeout = True
            break
        _idle_secs = 0.0
        if hasattr(agent, "get_activity_summary"):
            _act = agent.get_activity_summary()
            _idle_secs = _act.get("seconds_since_activity", 0.0)
        if _idle_secs >= _cron_inactivity_limit:
            _inactivity_timeout = True
            break

    pool.shutdown(wait=False, cancel_futures=True)
    assert result is None
    assert _wall_clock_timeout
    assert not _inactivity_timeout


def test_scheduler_watcher_source_has_wall_clock_branch():
    """The real scheduler watcher must consult the per-job wall-clock limit —
    guards against the cap living only in tests."""
    import inspect
    import cron.scheduler as scheduler

    src = inspect.getsource(scheduler)
    assert "_resolve_job_wall_clock_limit" in src
    assert "_wall_clock_timeout" in src
