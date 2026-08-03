"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

import json
from unittest.mock import patch

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


def test_create_job_no_agent_requires_script(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


def test_update_job_roundtrips_no_agent_flag(hermes_env):
    from cron.jobs import create_job, update_job, get_job

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local")

    update_job(job["id"], {"no_agent": False})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False

    update_job(job["id"], {"no_agent": True})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool: API-layer validation
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_no_agent_without_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "no_agent=True requires a script" in result.get("error", "")


# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        target="scheduler",
        no_agent=True,
        deliver="local",
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


def test_no_agent_script_runs_through_terminal_backend(hermes_env, monkeypatch):
    """A script-only cron job uses the same backend as the profile terminal."""
    from cron.jobs import create_job
    from cron import scheduler

    calls = []

    def fake_terminal(*, command, timeout, workdir, **_kwargs):
        calls.append((command, timeout, workdir))
        return '{"output": "coop scraped\\n", "exit_code": 0, "error": null}'

    monkeypatch.setattr(scheduler, "terminal_tool", fake_terminal)
    job = create_job(
        prompt=None,
        schedule="every 1h",
        script="/workspace/scrape_coop.py",
        target="backend",
        no_agent=True,
        deliver="local",
    )

    success, _doc, final_response, error = scheduler.run_job(job)

    assert success is True
    assert final_response == "coop scraped"
    assert error is None
    assert calls == [
        (
            "python3 /workspace/scrape_coop.py",
            min(scheduler._get_script_timeout(), scheduler.FOREGROUND_MAX_TIMEOUT),
            "/workspace",
        )
    ]


def test_no_agent_backend_script_keeps_backend_workdir(hermes_env, monkeypatch):
    """A backend workdir is not tested against the scheduler host filesystem."""
    from cron.jobs import create_job
    from cron import scheduler

    calls = []

    def fake_terminal(*, command, timeout, workdir, **_kwargs):
        calls.append((command, timeout, workdir))
        return '{"output": "ok", "exit_code": 0, "error": null}'

    monkeypatch.setattr(scheduler, "terminal_tool", fake_terminal)
    job = create_job(
        prompt=None,
        schedule="every 1h",
        script="/workspace/tasks/scrape_coop.py",
        workdir="/workspace",
        target="backend",
        no_agent=True,
        deliver="local",
    )

    success, _doc, _final_response, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert calls[0][2] == "/workspace"


def test_new_script_job_defaults_to_backend_target(hermes_env):
    """New script jobs follow the agent's terminal backend by default."""
    from cron.jobs import create_job

    job = create_job(
        prompt=None,
        schedule="every 1h",
        script="/workspace/worker.py",
        no_agent=True,
        deliver="local",
    )


    assert job["target"] == "backend"


def test_backend_script_runner_rejects_relative_path(hermes_env, monkeypatch):
    """Stored jobs cannot bypass the backend absolute-path API contract."""
    from cron import scheduler

    monkeypatch.setattr(
        scheduler,
        "terminal_tool",
        lambda **kwargs: pytest.fail("terminal backend must not run for a relative path"),
    )

    success, output = scheduler._run_job_script_in_backend("worker.py")

    assert success is False
    assert output == "Backend script path must be absolute: 'worker.py'."


def test_legacy_script_job_without_target_uses_scheduler_compat_path(hermes_env, monkeypatch):
    """Persisted jobs from before targets retain scheduler-host execution."""
    from cron.jobs import create_job
    from cron import scheduler

    calls = []

    def fake_scheduler_script(script_path, workdir=None):
        calls.append((script_path, workdir))
        return True, "scheduler output"

    monkeypatch.setattr(scheduler, "_run_job_script", fake_scheduler_script)
    monkeypatch.setattr(
        scheduler,
        "terminal_tool",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("backend must not run")),
    )
    job = create_job(
        prompt=None,
        schedule="every 1h",
        script="/workspace/legacy.py",
        no_agent=True,
        deliver="local",
    )
    job.pop("target")

    success, doc, _final_response, error = scheduler.run_job(job)
    assert success is True
    assert "scheduler output" in doc
    assert error is None
    assert calls == [("/workspace/legacy.py", None)]


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


def test_run_job_script_nul_path_fails_cleanly(hermes_env):
    """Sibling of the lifecycle-guard ingestion fix: a NUL-bearing script
    value can survive to fire time (the creation-time guard treats it as
    "nothing to scan"), and ``Path.expanduser()`` raises ValueError — not
    OSError — on it. The scheduler must fail the run with a report, not
    crash with an unhandled exception."""
    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("~user\x00bad.sh")
    assert ok is False
    assert "Blocked" in output
