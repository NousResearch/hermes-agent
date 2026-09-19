"""Cron run history falls back to the execution ledger for scriptless-session jobs.

A `no_agent` cron job runs a shell script: it appends to the execution ledger but
never opens an agent session. The dashboard's run panel read sessions only, so
every such job displayed "No runs yet" no matter how often it had fired.

These are behaviour contracts, not snapshots: they assert the relationship
between what a job leaves behind and what the endpoint reports.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the cron modules bound to it."""
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    from cron import executions as executions_mod
    from cron import jobs as jobs_mod
    importlib.reload(jobs_mod)
    importlib.reload(executions_mod)
    return home, jobs_mod, executions_mod


def _script_job(jobs_mod, name="nightly-backup"):
    """A no_agent job: the shape that leaves ledger rows and no sessions."""
    return jobs_mod.create_job(
        schedule="0 4 * * *", prompt=None, script="backup.sh", no_agent=True, name=name,
    )


def test_ledger_rows_surface_when_no_session_exists(cron_env):
    """A job with executions and no sessions reports its executions, not nothing."""
    _home, jobs_mod, executions_mod = cron_env
    job = _script_job(jobs_mod)

    rec = executions_mod.create_execution(job["id"], source="builtin")
    executions_mod.finish_execution(rec["id"], success=True)

    from hermes_cli.web_routers import cron as cron_router
    result = cron_router._list_cron_job_runs_sync(job["id"])

    assert len(result["runs"]) == 1, "an executed job must not report an empty history"
    row = result["runs"][0]
    assert row["_cron_execution"] is True
    assert row["_cron_status"] == "completed"


def test_run_rows_carry_every_field_the_frontend_requires(cron_env):
    """SessionInfo treats these as required; a missing key renders as NaN."""
    _home, jobs_mod, executions_mod = cron_env
    job = _script_job(jobs_mod)
    rec = executions_mod.create_execution(job["id"], source="builtin")
    executions_mod.finish_execution(rec["id"], success=True)

    from hermes_cli.web_routers import cron as cron_router
    row = cron_router._list_cron_job_runs_sync(job["id"])["runs"][0]

    for field in ("id", "started_at", "last_active", "ended_at", "is_active",
                  "title", "preview", "source", "message_count", "tool_call_count",
                  "input_tokens", "output_tokens"):
        assert field in row, f"SessionInfo requires {field!r}"
    assert isinstance(row["started_at"], (int, float))
    assert isinstance(row["message_count"], int)


def test_a_job_name_resolves_to_the_id_the_ledger_indexes(cron_env):
    """The panel passes whatever the user sees; the ledger is keyed by id.

    Without name resolution both stores are queried with a key neither indexes,
    which is the exact shape of the original "No runs yet" bug.
    """
    _home, jobs_mod, executions_mod = cron_env
    job = _script_job(jobs_mod, name="by-name-job")
    rec = executions_mod.create_execution(job["id"], source="builtin")
    executions_mod.finish_execution(rec["id"], success=True)

    from hermes_cli.web_routers import cron as cron_router
    by_name = cron_router._list_cron_job_runs_sync("by-name-job")
    by_id = cron_router._list_cron_job_runs_sync(job["id"])

    assert len(by_name["runs"]) == len(by_id["runs"]) == 1
    assert by_name["runs"][0]["id"] == by_id["runs"][0]["id"]


def test_a_failed_execution_shows_its_error(cron_env):
    """The panel is where a silent nightly failure becomes visible."""
    _home, jobs_mod, executions_mod = cron_env
    job = _script_job(jobs_mod)
    rec = executions_mod.create_execution(job["id"], source="builtin")
    executions_mod.finish_execution(rec["id"], success=False, error="disk full")

    from hermes_cli.web_routers import cron as cron_router
    row = cron_router._list_cron_job_runs_sync(job["id"])["runs"][0]

    assert row["_cron_status"] == "failed"
    assert "disk full" in row["preview"]


def test_a_job_that_never_ran_still_reports_empty(cron_env):
    """The fallback must not invent history."""
    _home, jobs_mod, _executions_mod = cron_env
    job = _script_job(jobs_mod, name="never-ran")

    from hermes_cli.web_routers import cron as cron_router
    assert cron_router._list_cron_job_runs_sync(job["id"])["runs"] == []
