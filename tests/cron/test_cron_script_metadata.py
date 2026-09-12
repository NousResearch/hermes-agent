"""Regression tests for per-script cron dispatch-metadata env injection.

A ``no_agent`` cron script had no way to know it was running late, so a report
delivered hours after an outage read as if it were on time. The overlay exposes
upstream's OWN dispatch classification to the script instead of inventing a
second one.

The single source of truth is ``job["last_dispatch"]``, stamped by
``cron.jobs`` immediately before a run (``scheduled_at`` / ``dispatched_at`` /
``lateness_seconds`` / ``kind`` in on_time|late|catch_up). ``_cron_script_env``
only forwards it; it never recomputes lateness and never falls back to
``next_run_at`` — which is already advanced on the tick path and, on a manual
one-shot, is the long-past original ``run_at``. Either fallback would fabricate
a catch-up for a run a person triggered by hand.

These tests prove:

* the metadata reaches a ``no_agent`` script's environment,
* the env agrees with the record the scheduler persisted (one classifier, one
  verdict),
* the REAL built-in tick still delivers the original due time,
* a manual run of an overdue one-shot gets NO metadata and NO label,
* a direct ``_run_job_script`` call injects no metadata AND strips inherited
  parent metadata, so pre-existing callers are unchanged.

Patch where production reads — ``cron.scheduler_script`` — not the
``cron.scheduler`` facade that re-exports the dispatch seam.
"""

import json
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_CRON_KEYS = (
    "HERMES_CRON_JOB_ID",
    "HERMES_CRON_SCHEDULED_AT",
    "HERMES_CRON_STARTED_AT",
    "HERMES_CRON_LATENESS_SECONDS",
    "HERMES_CRON_DISPATCH_KIND",
)


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


# Dumps every cron metadata var as a JSON line so run_job delivers them verbatim
# (no_agent forwards script stdout as the final message).
_DUMP_SCRIPT = textwrap.dedent(
    """\
    import json, os
    keys = ({keys})
    print(json.dumps({{k: os.environ.get(k) for k in keys}}))
    """
).format(keys=", ".join(repr(k) for k in _CRON_KEYS))


def _dispatch_stamp(scheduled_at: str, *, lateness: float, kind: str) -> dict:
    """The record ``cron.jobs`` stamps onto a job right before a dispatch."""
    return {
        "scheduled_at": scheduled_at,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
        "lateness_seconds": lateness,
        "kind": kind,
    }


def _write_script(home: Path, name: str, body: str) -> str:
    path = home / "scripts" / name
    path.write_text(body)
    return name


def test_cron_env_reaches_no_agent_script(cron_env):
    """A cron-dispatched no_agent script sees the full dispatch metadata."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    job = create_job(
        prompt=None, schedule="every 5m", script="dump.py",
        no_agent=True, deliver="local",
    )
    # A scheduled dispatch: the ticker stamps the due time it fired for.
    due = job["next_run_at"]
    job["last_dispatch"] = _dispatch_stamp(due, lateness=18000.0, kind="catch_up")

    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None

    payload = json.loads(final_response.strip().splitlines()[-1])
    assert payload["HERMES_CRON_JOB_ID"] == job["id"]
    # Scheduled-at is the ORIGINAL due time this dispatch fired for.
    assert payload["HERMES_CRON_SCHEDULED_AT"] == due
    # Started-at is the dispatch stamp, not a fresh wall-clock read.
    assert payload["HERMES_CRON_STARTED_AT"] == job["last_dispatch"]["dispatched_at"]
    # The classification is forwarded verbatim, never recomputed here.
    assert payload["HERMES_CRON_LATENESS_SECONDS"] == "18000.0"
    assert payload["HERMES_CRON_DISPATCH_KIND"] == "catch_up"


def test_no_scheduled_provenance_means_no_metadata_at_all(cron_env):
    """run_job on a job with no dispatch stamp is a manual/direct run.

    ``next_run_at`` is deliberately NOT a fallback: it is already advanced on
    the tick path, and on a manual one-shot it is the long-past original
    run_at, which would fabricate a catch-up for a hand-triggered run.
    """
    from cron.jobs import create_job
    from cron.scheduler import run_job

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    job = create_job(
        prompt=None, schedule="every 5m", script="dump.py",
        no_agent=True, deliver="local",
    )
    assert job["next_run_at"]  # present, and still must not be used

    success, doc, final_response, error = run_job(job)
    assert success is True

    payload = json.loads(final_response.strip().splitlines()[-1])
    assert all(payload[key] is None for key in _CRON_KEYS)


def test_scheduled_at_comes_from_the_dispatch_stamp(cron_env):
    """_cron_script_env reads last_dispatch, not next_run_at."""
    from cron.scheduler_script import _cron_script_env

    job = {
        "id": "job-xyz",
        "next_run_at": "2026-07-13T12:00:00+03:00",       # already advanced
        "last_dispatch": {
            "scheduled_at": "2026-07-13T09:00:00+03:00",
            "dispatched_at": "2026-07-13T11:00:00+03:00",
            "lateness_seconds": 7200.0,
            "kind": "catch_up",
        },
    }
    env = _cron_script_env(job)
    assert env["HERMES_CRON_JOB_ID"] == "job-xyz"
    assert env["HERMES_CRON_SCHEDULED_AT"] == "2026-07-13T09:00:00+03:00"
    assert env["HERMES_CRON_STARTED_AT"] == "2026-07-13T11:00:00+03:00"
    assert env["HERMES_CRON_LATENESS_SECONDS"] == "7200.0"
    assert env["HERMES_CRON_DISPATCH_KIND"] == "catch_up"


@pytest.mark.parametrize("stamp", [
    None,
    "",
    "   ",
    12345,
    {},
    {"scheduled_at": "2026-07-13T09:00:00+03:00"},          # no dispatched_at
    {"dispatched_at": "2026-07-13T11:00:00+03:00"},         # no scheduled_at
    {"scheduled_at": "", "dispatched_at": ""},
])
def test_env_is_empty_without_usable_dispatch_provenance(cron_env, stamp):
    """No trusted stamp -> no metadata. Never fall back, never guess.

    Both timestamps come from the same upstream record; a half-written one is
    not provenance, and inventing a wall-clock "now" here is exactly the
    guessing this contract exists to remove.
    """
    from cron.scheduler_script import _cron_script_env

    job = {"id": "job-xyz", "next_run_at": "2026-07-13T09:00:00+03:00"}
    if stamp is not None:
        job["last_dispatch"] = stamp
    assert _cron_script_env(job) == {}


def test_advance_next_run_does_not_mutate_in_memory_job(cron_env):
    """The due job handed to run_job keeps its original next_run_at even after
    advance_next_run rolls the persistent schedule forward — so the metadata
    reflects the real 'should have run at' time, not the advanced one."""
    from cron.jobs import create_job, advance_next_run, get_job

    job = create_job(prompt="x", schedule="every 5m")
    original = job["next_run_at"]

    advanced = advance_next_run(job["id"])
    assert advanced is True
    # Persistent store rolled forward...
    assert get_job(job["id"])["next_run_at"] != original
    # ...but the in-memory dict (what run_job/_cron_script_env read) did not.
    assert job["next_run_at"] == original


def test_direct_script_call_injects_no_metadata(cron_env):
    """A plain _run_job_script call (no scoped dispatch) must not expose any
    HERMES_CRON_* var — pre-existing/manual callers are unchanged."""
    from cron.scheduler_script import _run_job_script

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    ok, output = _run_job_script("dump.py")
    assert ok is True

    payload = json.loads(output.strip().splitlines()[-1])
    assert all(payload[key] is None for key in _CRON_KEYS)


def test_direct_script_call_strips_inherited_parent_metadata(cron_env, monkeypatch):
    """Stale parent markers must not leak into a direct/manual script run."""
    from cron.scheduler_script import _run_job_script

    monkeypatch.setenv("HERMES_CRON_JOB_ID", "stale-parent-job")
    monkeypatch.setenv("HERMES_CRON_SCHEDULED_AT", "2020-01-01T00:00:00+00:00")
    monkeypatch.setenv("HERMES_CRON_STARTED_AT", "2020-01-02T00:00:00+00:00")
    monkeypatch.setenv("HERMES_CRON_LATENESS_SECONDS", "999999.0")
    monkeypatch.setenv("HERMES_CRON_DISPATCH_KIND", "catch_up")
    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)

    ok, output = _run_job_script("dump.py")
    assert ok is True
    payload = json.loads(output.strip().splitlines()[-1])
    assert all(payload[key] is None for key in _CRON_KEYS)


def test_scoped_cron_context_reaches_script(cron_env):
    """The real runner consumes dispatch metadata without a new public kwarg."""
    from cron.scheduler_script import _CRON_SCRIPT_ENV_CONTEXT, _run_job_script

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    token = _CRON_SCRIPT_ENV_CONTEXT.set({
        "HERMES_CRON_JOB_ID": "abc",
        "HERMES_CRON_SCHEDULED_AT": "2026-07-13T09:00:00+03:00",
        "HERMES_CRON_STARTED_AT": "2026-07-13T09:00:01+03:00",
    })
    try:
        ok, output = _run_job_script("dump.py")
    finally:
        _CRON_SCRIPT_ENV_CONTEXT.reset(token)
    assert ok is True
    payload = json.loads(output.strip().splitlines()[-1])
    assert payload["HERMES_CRON_JOB_ID"] == "abc"
    assert payload["HERMES_CRON_SCHEDULED_AT"] == "2026-07-13T09:00:00+03:00"


def test_dispatch_runner_scopes_metadata_and_restores_the_context(cron_env):
    """The dispatch seam sets the context for the run and clears it after."""
    from cron.scheduler_script import _CRON_SCRIPT_ENV_CONTEXT, _run_job_script_for_dispatch

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    job = {
        "id": "scoped-job",
        "last_dispatch": _dispatch_stamp(
            "2026-07-13T09:00:00+03:00", lateness=7200.0, kind="catch_up"),
    }
    ok, output = _run_job_script_for_dispatch(job, "dump.py")
    assert ok is True
    payload = json.loads(output.strip().splitlines()[-1])
    assert payload["HERMES_CRON_JOB_ID"] == "scoped-job"
    assert payload["HERMES_CRON_DISPATCH_KIND"] == "catch_up"
    # Scoped, not global: nothing survives the call.
    assert _CRON_SCRIPT_ENV_CONTEXT.get() is None


def test_heartbeat_wrapper_preserves_legacy_script_runner_signature(monkeypatch):
    import cron.scheduler_script as scheduler_script

    observed = []

    def legacy_runner(script_path, workdir=None, cancel_event=None):
        observed.append((script_path, workdir, cancel_event))
        return True, "ok"

    # Patch where production reads: _run_job_script_for_dispatch resolves the
    # runner in this module's namespace, not through the scheduler facade.
    monkeypatch.setattr(scheduler_script, "_run_job_script", legacy_runner)
    result = scheduler_script._run_job_script_with_claim_heartbeat(
        {
            "id": "legacy-seam",
            "next_run_at": "2026-08-27T00:00:00+03:00",
            "schedule": {"kind": "interval", "minutes": 60},
        },
        "probe.py",
        workdir="/tmp",
    )
    assert result == (True, "ok")
    assert observed == [("probe.py", "/tmp", None)]
    assert scheduler_script._CRON_SCRIPT_ENV_CONTEXT.get() is None


# --------------------------------------------------------------------------- #
# The production ordering: due stamp -> advance -> claim -> run
#
# These drive the REAL built-in tick, not a reconstruction of it.
# --------------------------------------------------------------------------- #

def _due_recurring_job(home: Path, *, hours_late: float, script: bool = True):
    """One recurring job whose due time is ``hours_late`` hours in the past."""
    from cron.jobs import create_job, update_job

    if script:
        _write_script(home, "dump.py", _DUMP_SCRIPT)
    job = create_job(
        prompt=None if script else "x",
        schedule="every 60m",
        script="dump.py" if script else None,
        no_agent=bool(script),
        deliver="telegram:-100:1",
    )
    due = (datetime.now(timezone.utc) - timedelta(hours=hours_late)).isoformat()
    update_job(job["id"], {"next_run_at": due})
    return job["id"], due


def _tick_capturing_delivery(monkeypatch):
    import cron.scheduler as scheduler

    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_result",
        lambda job, content, adapters=None, loop=None, **kwargs: delivered.append(content),
    )
    scheduler.tick()
    return delivered


def _persisted(home: Path, job_id: str) -> dict:
    raw = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    jobs = raw["jobs"] if isinstance(raw, dict) and "jobs" in raw else raw
    return next(j for j in jobs if j["id"] == job_id)


def _backdate_on_disk(home: Path, job_id: str, due: str) -> None:
    """Rewrite jobs.json so ``job_id`` is overdue, bypassing the update guard.

    ``update_job``/``create_job`` refuse a one-shot in the past (the 120s grace
    window), but that is exactly the record an outage leaves: scheduled while
    the server was up, still pending when it came back.
    """
    path = home / "cron" / "jobs.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    jobs = raw["jobs"] if isinstance(raw, dict) and "jobs" in raw else raw
    for job in jobs:
        if job["id"] == job_id:
            job["next_run_at"] = due
            if isinstance(job.get("schedule"), dict):
                job["schedule"]["run_at"] = due
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_tick_advances_next_run_at_before_execution(cron_env, monkeypatch):
    """Guard the guard: prove these tests drive the hazard they claim to."""
    job_id, due = _due_recurring_job(cron_env, hours_late=5)
    _tick_capturing_delivery(monkeypatch)
    persisted = _persisted(cron_env, job_id)
    assert persisted["next_run_at"] != due
    assert datetime.fromisoformat(persisted["next_run_at"]) > datetime.fromisoformat(due)


def test_builtin_tick_delivers_original_due_time_to_the_script(cron_env, monkeypatch):
    """HERMES_CRON_SCHEDULED_AT must survive advance_next_runs + the claim."""
    job_id, due = _due_recurring_job(cron_env, hours_late=5)
    delivered = _tick_capturing_delivery(monkeypatch)

    assert len(delivered) == 1
    payload = json.loads(delivered[0].splitlines()[-1])
    assert payload["HERMES_CRON_JOB_ID"] == job_id
    assert payload["HERMES_CRON_SCHEDULED_AT"] == due
    assert payload["HERMES_CRON_STARTED_AT"]


def test_script_env_agrees_with_the_persisted_dispatch_record(cron_env, monkeypatch):
    """One classifier, one verdict.

    The env a script reads and the record the scheduler persisted come from the
    same stamp, so a reader can never see a "catch_up" banner over a job whose
    stored history says it ran on time (or the reverse).
    """
    job_id, _due = _due_recurring_job(cron_env, hours_late=5)
    delivered = _tick_capturing_delivery(monkeypatch)
    assert len(delivered) == 1

    payload = json.loads(delivered[0].splitlines()[-1])
    stamp = _persisted(cron_env, job_id)["last_dispatch"]
    assert payload["HERMES_CRON_SCHEDULED_AT"] == stamp["scheduled_at"]
    assert payload["HERMES_CRON_STARTED_AT"] == stamp["dispatched_at"]
    assert payload["HERMES_CRON_DISPATCH_KIND"] == stamp["kind"] == "catch_up"
    assert payload["HERMES_CRON_LATENESS_SECONDS"] == str(stamp["lateness_seconds"])


def test_builtin_tick_delivers_the_due_time_of_an_on_time_run(cron_env, monkeypatch):
    """An on-time dispatch still carries its own scheduled_at."""
    _job_id, due = _due_recurring_job(cron_env, hours_late=0)
    delivered = _tick_capturing_delivery(monkeypatch)
    assert len(delivered) == 1
    payload = json.loads(delivered[0].splitlines()[-1])
    assert payload["HERMES_CRON_SCHEDULED_AT"] == due


def test_missed_one_shot_is_retired_not_delivered_late(cron_env, monkeypatch):
    """A one-shot past its grace window never fires, so it never needs a label.

    ``get_due_jobs`` retires a one-shot more than ``ONESHOT_GRACE_SECONDS``
    (120s) past due rather than dispatching it — far below the catch-up
    threshold. So on the scheduled path a one-shot is either on time or gone,
    and the delayed label is a recurring-job concern by construction. This is
    pinned because it is the reason the one-shot lane needs no catch-up
    handling, not an accident.
    """
    from cron.jobs import ONESHOT_GRACE_SECONDS, create_job

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    due = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    job = create_job(
        prompt=None, schedule=soon, script="dump.py",
        no_agent=True, deliver="telegram:-100:1",
    )
    assert ONESHOT_GRACE_SECONDS < 2 * 60 * 60
    # The on-disk state an outage leaves: scheduled while up, still pending
    # when the server came back.
    _backdate_on_disk(cron_env, job["id"], due)

    delivered = _tick_capturing_delivery(monkeypatch)
    assert delivered == []
    raw = json.loads((cron_env / "cron" / "jobs.json").read_text(encoding="utf-8"))
    jobs = raw["jobs"] if isinstance(raw, dict) and "jobs" in raw else raw
    assert [j for j in jobs if j["id"] == job["id"]] == []


def test_tick_leaves_claim_and_completion_bookkeeping_intact(cron_env, monkeypatch):
    """Carrying the dispatch metadata is additive: it must not disturb the claim."""
    job_id, _due = _due_recurring_job(cron_env, hours_late=5)
    _tick_capturing_delivery(monkeypatch)
    persisted = _persisted(cron_env, job_id)
    assert persisted["last_status"] == "ok"
    assert persisted["last_run_at"]
    assert not persisted.get("fire_claim")


# --------------------------------------------------------------------------- #
# Manual runs: the operator triggered it, so it is not a catch-up
# --------------------------------------------------------------------------- #

def test_manual_run_of_overdue_one_shot_gets_no_metadata_and_no_label(
    cron_env, monkeypatch,
):
    """The real `cronjob run` path on a one-shot that has been overdue for days.

    ``claim_job_for_fire`` does not advance a one-shot's ``next_run_at``, so a
    ``next_run_at`` fallback would hand this run cron metadata and a "delayed"
    banner for something the operator just triggered by hand.
    """
    import cron.scheduler as scheduler
    from cron.jobs import create_job, update_job
    from tools.cronjob_tools import _execute_job_now

    _write_script(cron_env, "dump.py", _DUMP_SCRIPT)
    soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    due = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
    job = create_job(
        prompt=None, schedule=soon, script="dump.py",
        no_agent=True, deliver="telegram:-100:1",
    )
    update_job(job["id"], {"next_run_at": due})

    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_result",
        lambda job, content, adapters=None, loop=None, **kwargs: delivered.append(content),
    )

    result = _execute_job_now(dict(job, next_run_at=due))
    assert result["claimed"] is True

    assert len(delivered) == 1
    payload = json.loads(delivered[0].splitlines()[-1])
    assert all(payload[key] is None for key in _CRON_KEYS)
