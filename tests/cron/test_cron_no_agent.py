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
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


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


def test_run_job_script_uses_shared_bash_resolver_and_safe_path(hermes_env, monkeypatch):
    """Cron .sh execution must not pick WSL bash or pass raw Windows paths."""
    from cron.scheduler import _run_job_script
    import tools.environments.local as local_mod

    script = hermes_env / "scripts" / "watch.sh"
    script.write_text("#!/bin/bash\necho ok\n")
    git_bash = "C:/Program Files/Git/bin/bash.exe"
    captured = {}

    class Result:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    monkeypatch.setattr(
        "cron.scheduler.shutil.which",
        lambda _name: r"C:\Windows\System32\bash.exe",
    )
    monkeypatch.setattr(local_mod, "_find_bash", lambda: git_bash)
    monkeypatch.setattr(local_mod, "_bash_safe_path", lambda path: f"SAFE:{path}")

    def fake_run(argv, **_kwargs):
        captured["argv"] = argv
        return Result()

    monkeypatch.setattr("cron.scheduler.subprocess.run", fake_run)

    ok, output = _run_job_script(str(script))

    assert ok is True
    assert output == "ok"
    assert captured["argv"] == [git_bash, f"SAFE:{script}"]


def test_run_job_script_reports_unusable_bash_without_spawning(
    hermes_env, monkeypatch
):
    from cron.scheduler import _run_job_script
    import tools.environments.local as local_mod

    script = hermes_env / "scripts" / "watch.sh"
    script.write_text("#!/bin/bash\necho ok\n")

    def no_bash():
        raise RuntimeError("no git bash")

    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("must not spawn WSL bash")

    monkeypatch.setattr(local_mod, "_find_bash", no_bash)
    monkeypatch.setattr("cron.scheduler.subprocess.run", unexpected_spawn)

    ok, output = _run_job_script(str(script))

    assert ok is False
    assert "bash not found" in output
