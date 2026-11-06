"""Dispatch metadata reaches a scheduled script and does not leak into a direct run."""

import json
import textwrap

import pytest

from cron.scheduler_script import (
    _CRON_SCRIPT_ENV_CONTEXT,
    _cron_script_env,
    _run_job_script,
    _run_job_script_with_claim_heartbeat,
)

_KEYS = (
    "HERMES_CRON_JOB_ID",
    "HERMES_CRON_SCHEDULED_AT",
    "HERMES_CRON_STARTED_AT",
    "HERMES_CRON_LATENESS_SECONDS",
    "HERMES_CRON_DISPATCH_KIND",
)

_DUMP = textwrap.dedent(
    f"""\
    import json, os
    keys = {_KEYS!r}
    print(json.dumps({{k: os.environ.get(k) for k in keys}}))
    """
)


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    return hermes_home


def _write(home, name="dump.py"):
    path = home / "scripts" / name
    path.write_text(_DUMP)
    return name


def test_direct_call_strips_inherited_metadata(cron_env, monkeypatch):
    from cron.scheduler_script import _run_job_script as run

    for key in _KEYS:
        monkeypatch.setenv(key, "stale")
    _write(cron_env)
    ok, output = run("dump.py")
    assert ok is True
    payload = json.loads(output.strip().splitlines()[-1])
    assert all(payload[key] is None for key in _KEYS)


def test_heartbeat_wrapper_forwards_last_dispatch(cron_env):
    _write(cron_env)
    job = {
        "id": "job-1",
        "schedule": {"kind": "interval", "minutes": 5},
        "last_dispatch": {
            "scheduled_at": "2026-07-13T09:00:00+03:00",
            "dispatched_at": "2026-07-13T11:00:00+03:00",
            "lateness_seconds": 7200.0,
            "kind": "catch_up",
        },
    }
    ok, output = _run_job_script_with_claim_heartbeat(job, "dump.py")
    assert ok is True
    payload = json.loads(output.strip().splitlines()[-1])
    assert payload["HERMES_CRON_JOB_ID"] == "job-1"
    assert payload["HERMES_CRON_SCHEDULED_AT"] == "2026-07-13T09:00:00+03:00"
    assert payload["HERMES_CRON_DISPATCH_KIND"] == "catch_up"
    assert _CRON_SCRIPT_ENV_CONTEXT.get() is None


def test_env_helper_does_not_guess():
    assert _cron_script_env({"id": "x", "next_run_at": "2026-01-01T00:00:00+00:00"}) == {}
    assert _cron_script_env({"id": "x", "last_dispatch": {"scheduled_at": "a"}}) == {}
