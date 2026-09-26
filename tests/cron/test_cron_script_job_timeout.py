"""Per-job ceiling for cron scripts.

A ``no_agent`` job scheduled every 15 minutes ran for 95 minutes because the only ceiling was the
global ``cron.script_timeout_seconds``; each fire slot it overran piled another wedged instance
behind it. The ceiling is now per job: explicit ``timeout_s`` on the job, else the schedule
interval, never above the global.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone

import pytest

from cron import scheduler_script as scheduler_ext


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
# resolve_job_script_timeout: precedence + clamp contract
# ---------------------------------------------------------------------------


def test_interval_job_ceiling_is_its_interval():
    job = {"schedule": {"kind": "interval", "minutes": 15}}
    assert scheduler_ext.resolve_job_script_timeout(job, 7200) == (900, "interval")


def test_ceiling_never_exceeds_global():
    daily = {"schedule": {"kind": "interval", "minutes": 1440}}
    assert scheduler_ext.resolve_job_script_timeout(daily, 7200) == (7200, "interval")
    explicit = {"timeout_s": 99999, "schedule": {"kind": "interval", "minutes": 5}}
    assert scheduler_ext.resolve_job_script_timeout(explicit, 7200) == (7200, "job")


def test_explicit_timeout_overrides_interval_both_ways():
    # A */5 job that legitimately needs ~8 min opts up; a daily job opts down.
    slow = {"timeout_s": 900, "schedule": {"kind": "cron", "expr": "*/5 * * * *"}}
    assert scheduler_ext.resolve_job_script_timeout(slow, 7200) == (900, "job")
    tight = {"timeout_s": "120", "schedule": {"kind": "interval", "minutes": 1440}}
    assert scheduler_ext.resolve_job_script_timeout(tight, 7200) == (120, "job")


@pytest.mark.parametrize("bad", [0, -5, "nope", None, True])
def test_invalid_explicit_timeout_falls_back_to_interval(bad):
    job = {"timeout_s": bad, "schedule": {"kind": "interval", "minutes": 10}}
    assert scheduler_ext.resolve_job_script_timeout(job, 7200) == (600, "interval")


def test_cron_ceiling_is_the_slot_width():
    now = datetime(2026, 9, 24, 14, 7, 30, tzinfo=timezone.utc)
    job = {"schedule": {"kind": "cron", "expr": "*/15 * * * *"}}
    assert scheduler_ext.resolve_job_script_timeout(job, 7200, now=now) == (900, "interval")
    # Irregular expression: the slot containing `now` (09:00 -> 10:00), not
    # the 23 h gap after it.
    irregular = {"schedule": {"kind": "cron", "expr": "0 9,10 * * *"}}
    at = datetime(2026, 9, 24, 9, 30, tzinfo=timezone.utc)
    assert scheduler_ext.resolve_job_script_timeout(irregular, 7200, now=at) == (3600, "interval")


def test_short_interval_gets_floor():
    job = {"schedule": {"kind": "cron", "expr": "* * * * *"}}
    got, _ = scheduler_ext.resolve_job_script_timeout(job, 7200)
    assert got == scheduler_ext.MIN_DERIVED_SCRIPT_TIMEOUT


@pytest.mark.parametrize(
    "schedule",
    [{"kind": "once", "run_at": "2026-09-24T00:00:00Z"}, {"kind": "cron", "expr": "bogus"}, None, "x"],
)
def test_unknown_schedule_uses_global(schedule):
    assert scheduler_ext.resolve_job_script_timeout({"schedule": schedule}, 3600) == (3600, "global")


# ---------------------------------------------------------------------------
# End to end: the scheduler kills a TERM-ignoring script AT its interval
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_term_ignoring_script_is_killed_at_its_interval(hermes_env, monkeypatch, caplog):
    import cron.scheduler as sched
    from cron import scheduler_script as ext

    # 0.05 min = 3 s interval; drop the 60 s floor so the test stays fast.
    monkeypatch.setattr(ext, "MIN_DERIVED_SCRIPT_TIMEOUT", 0)
    monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", 7200)

    pidfile = hermes_env / "grandchild.pid"
    script = hermes_env / "scripts" / "wedge.sh"
    script.write_text(
        "trap '' TERM\n"
        f"sleep 120 & echo $! > {pidfile}\n"
        "wait\n",
        encoding="utf-8",
    )
    job = {
        "id": "wedge1",
        "name": "wedge-job",
        "schedule": {"kind": "interval", "minutes": 0.05},
        "script": "wedge.sh",
        "no_agent": True,
    }

    caplog.set_level("WARNING", logger=sched.logger.name)
    t0 = time.monotonic()
    ok, output = sched._run_job_script_with_claim_heartbeat(job, "wedge.sh")
    elapsed = time.monotonic() - t0

    assert ok is False
    assert "timed out after 3s" in output
    # Killed at the interval (3 s) + TERM->KILL escalation, not the 7200 s global.
    assert elapsed < 3 + 8, elapsed
    grandchild = int(pidfile.read_text().strip())
    deadline = time.monotonic() + 3
    while _pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(grandchild), "process group survivor"
    phase = [r.getMessage() for r in caplog.records if "PHASE=cron_script_timeout" in r.getMessage()]
    assert phase and "job=wedge-job" in phase[0] and "elapsed=3" in phase[0], phase


@pytest.mark.skipif(os.name != "posix", reason="POSIX")
def test_script_within_interval_still_succeeds(hermes_env, monkeypatch):
    import cron.scheduler as sched

    script = hermes_env / "scripts" / "quick.sh"
    script.write_text("echo done\n", encoding="utf-8")
    job = {"id": "q", "name": "quick", "schedule": {"kind": "interval", "minutes": 15}}
    ok, output = sched._run_job_script_with_claim_heartbeat(job, "quick.sh")
    assert ok is True and output == "done"


# ---------------------------------------------------------------------------
# Every script call site carries the per-job ceiling (sibling paths)
# ---------------------------------------------------------------------------


def _capture_run_job_script(monkeypatch):
    import cron.scheduler_script as script_mod

    calls = []

    def fake(script_path, workdir=None, cancel_event=None, **kwargs):
        calls.append(kwargs)
        return True, ""

    monkeypatch.setattr(script_mod, "_run_job_script", fake)
    return calls


def test_prompt_path_without_prerun_passes_job_ceiling(hermes_env, monkeypatch):
    from cron.scheduler_prompt import _build_job_prompt

    calls = _capture_run_job_script(monkeypatch)
    job = {"id": "p1", "name": "prompt-job", "prompt": "x", "script": "s.sh",
           "schedule": {"kind": "interval", "minutes": 10}}
    _build_job_prompt(job)
    assert calls == [{"timeout_seconds": 600, "job_name": "prompt-job"}]


def test_monitor_script_passes_job_ceiling(hermes_env, monkeypatch):
    from cron.monitor import _run_monitor_source

    calls = _capture_run_job_script(monkeypatch)
    job = {"id": "m1", "name": "monitor-job", "monitor_script": "m.sh", "timeout_s": 45,
           "schedule": {"kind": "interval", "minutes": 30}}
    _run_monitor_source(job)
    assert calls == [{"timeout_seconds": 45, "job_name": "monitor-job"}]


def test_timeout_s_is_an_authored_field():
    # Profile distributions refresh authored fields; an operator-set ceiling must travel with
    # the job definition rather than be dropped as scheduler-owned state.
    from cron.job_definition import JOB_DEFINITION_FIELDS

    assert scheduler_ext.JOB_SCRIPT_TIMEOUT_KEY in JOB_DEFINITION_FIELDS
