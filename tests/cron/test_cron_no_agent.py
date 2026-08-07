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
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", target="scheduler", no_agent=True, deliver="local")

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
    import tools.terminal_tool

    calls = []
    waits = []

    def fake_terminal(**kwargs):
        calls.append(kwargs)
        if kwargs["command"].startswith("test -f "):
            return '{"output": "", "exit_code": 0, "error": null}'
        return '{"output": "Background process started", "session_id": "proc_cron", "exit_code": 0, "error": null}'

    def fake_wait(session_id, timeout, *_args):
        waits.append((session_id, timeout))
        return True, "coop scraped"

    monkeypatch.setattr(scheduler, "terminal_tool", fake_terminal)
    monkeypatch.setattr(scheduler, "_wait_for_backend_process", fake_wait, raising=False)
    monkeypatch.setattr(scheduler, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr(tools.terminal_tool, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr("tools.cronjob_tools._validate_backend_script", lambda script, workdir=None: None)
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
    assert calls[0] == {
        "command": "test -f /workspace/scrape_coop.py",
        "timeout": 30,
        "workdir": "/workspace",
    }
    assert calls[1]["command"] == "python3 /workspace/scrape_coop.py"
    assert calls[1]["background"] is True
    assert calls[1]["workdir"] == "/workspace"
    assert waits == [("proc_cron", scheduler._get_script_timeout())]


def test_no_agent_backend_script_keeps_backend_workdir(hermes_env, monkeypatch):
    """A backend workdir is not tested against the scheduler host filesystem."""
    from cron.jobs import create_job
    from cron import scheduler
    import tools.terminal_tool

    calls = []

    def fake_terminal(**kwargs):
        calls.append(kwargs)
        if kwargs["command"].startswith("test -f "):
            return '{"output": "", "exit_code": 0, "error": null}'
        return '{"output": "Background process started", "session_id": "proc_workdir", "exit_code": 0, "error": null}'

    monkeypatch.setattr(scheduler, "terminal_tool", fake_terminal)
    monkeypatch.setattr(scheduler, "_wait_for_backend_process", lambda *_args: (True, "ok"), raising=False)
    monkeypatch.setattr(scheduler, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr(tools.terminal_tool, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr("tools.cronjob_tools._validate_backend_script", lambda script, workdir=None: None)
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
    assert calls[0]["workdir"] == "/workspace"
    assert calls[1]["workdir"] == "/workspace"
    assert calls[1]["background"] is True


def test_new_script_job_defaults_to_backend_target(hermes_env, monkeypatch):
    """New script jobs follow the agent's terminal backend by default."""
    from cron.jobs import create_job
    import tools.terminal_tool

    monkeypatch.setattr(tools.terminal_tool, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr("tools.cronjob_tools._validate_backend_script", lambda script, workdir=None: None)

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
    (hermes_env / "scripts" / "legacy.py").write_text("print('legacy')\n")
    job = create_job(
        prompt=None,
        schedule="every 1h",
        script="legacy.py",
        target="scheduler",
        no_agent=True,
        deliver="local",
    )
    job.pop("target")

    success, doc, _final_response, error = scheduler.run_job(job)
    assert success is True
    assert "scheduler output" in doc
    assert error is None
    assert calls == [("legacy.py", None)]


def test_backend_process_timeout_kills_tracked_process(monkeypatch):
    """Cron owns the long script deadline and terminates a timed-out backend process."""
    from types import SimpleNamespace
    from cron import scheduler
    import tools.process_registry

    waits = []
    kills = []
    fake_registry = SimpleNamespace(
        wait=lambda session_id, timeout: waits.append((session_id, timeout)) or {"status": "timeout"},
        kill_process=lambda session_id, source: kills.append((session_id, source)) or {"status": "killed"},
    )
    ticks = iter((0.0, 0.0, 3600.0))
    monkeypatch.setattr(tools.process_registry, "process_registry", fake_registry)
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: next(ticks))

    success, output = scheduler._wait_for_backend_process("proc_cron", 3600)

    assert success is False
    assert output == "Script timed out after 3600s"
    assert waits == [("proc_cron", 180)]
    assert kills == [("proc_cron", "cron.script_timeout")]


def test_backend_process_output_is_redacted_before_cron_consumes_it(monkeypatch):
    """Tracked backend output must retain terminal-tool secret redaction."""
    from types import SimpleNamespace
    from cron import scheduler
    import tools.process_registry

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    fake_registry = SimpleNamespace(
        wait=lambda *_args, **_kwargs: {"status": "exited", "exit_code": 0, "output": secret},
        read_log=lambda *_args, **_kwargs: {"output": secret},
    )
    monkeypatch.setattr(tools.process_registry, "process_registry", fake_registry)

    success, output = scheduler._wait_for_backend_process("proc_redact", 3600)

    assert success is True
    assert secret not in output


def test_backend_script_start_failure_redacts_output(hermes_env, monkeypatch):
    """Backend launch errors must not bypass terminal-style secret redaction."""
    from cron import scheduler

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    responses = iter((
        '{"output": "", "exit_code": 0, "error": null}',
        '{"output": "", "exit_code": 1, "error": "' + secret + '"}',
    ))
    monkeypatch.setattr(scheduler, "terminal_tool", lambda **_kwargs: next(responses))

    success, output = scheduler._run_job_script_in_backend("/workspace/worker.py")

    assert success is False
    assert secret not in output


def test_backend_process_timeout_reports_cleanup_failure(monkeypatch):
    """A failed termination remains visible instead of pretending the process stopped."""
    from types import SimpleNamespace
    from cron import scheduler
    import tools.process_registry

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    fake_registry = SimpleNamespace(
        wait=lambda *_args, **_kwargs: {"status": "timeout"},
        kill_process=lambda *_args, **_kwargs: {"status": "error", "error": secret},
    )
    ticks = iter((0.0, 0.0, 3600.0))
    monkeypatch.setattr(tools.process_registry, "process_registry", fake_registry)
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: next(ticks))

    success, output = scheduler._wait_for_backend_process("proc_cron", 3600)

    assert success is False
    assert "cleanup failed:" in output
    assert secret not in output


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
