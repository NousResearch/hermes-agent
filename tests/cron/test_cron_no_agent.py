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
import pathlib
import subprocess
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


def test_run_job_no_agent_reloads_dotenv_before_script(hermes_env, monkeypatch):
    """Regression: a standalone cron tick process starts without home-channel
    vars in its environment, and the agent path's per-run dotenv reload never
    executes for no_agent jobs — delivery home channels stayed unresolved.
    run_job must load .env at the top of the no_agent branch."""
    import hermes_cli.env_loader as env_loader
    from cron.jobs import create_job
    from cron.scheduler import run_job

    loaded_homes: list = []

    def fake_load(*, hermes_home=None, project_env=None):
        loaded_homes.append(hermes_home)
        return []

    monkeypatch.setattr(env_loader, "load_hermes_dotenv", fake_load)

    script_path = hermes_env / "scripts" / "probe.sh"
    script_path.write_text('#!/bin/bash\necho "ok"\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="probe.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert loaded_homes, "load_hermes_dotenv was not called on the no_agent path"
    assert str(loaded_homes[0]) == str(hermes_env)


def test_timed_out_no_agent_script_delivery_is_not_mislabeled_as_provider_failure(
    hermes_env, monkeypatch,
):
    """A watchdog timeout happens before any LLM/provider call.

    The delivery summary must preserve that process-level failure taxonomy and
    must not claim a provider fallback was attempted or exhausted.
    """
    from cron.jobs import create_job
    import cron.scheduler as scheduler

    (hermes_env / "scripts" / "slow.py").write_text("import time; time.sleep(999)\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="slow.py",
        no_agent=True,
        deliver="telegram",
        name="slow watchdog",
    )
    delivered = []

    # The script runner uses Popen + a polling loop (cancel/timeout aware),
    # so simulate a process that never finishes: communicate() always times
    # out and the script deadline is shrunk to keep the test fast.
    class _NeverFinishes:
        returncode = None
        pid = 0
        stdout = None
        stderr = None

        def __init__(self, *_args, **_kwargs):
            pass

        def poll(self):
            return None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="slow.py", timeout=timeout)

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="slow.py", timeout=timeout)

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(scheduler.subprocess, "Popen", _NeverFinishes)
    monkeypatch.setattr(scheduler, "_get_script_timeout", lambda: 1)
    monkeypatch.setattr(
        scheduler,
        "_terminate_cron_script_process",
        lambda proc: setattr(proc, "returncode", -15),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content),
    )

    assert scheduler.run_one_job(job) is True
    assert len(delivered) == 1
    assert "script timed out" in delivered[0].lower()
    assert "provider" not in delivered[0].lower()
    assert "fallback" not in delivered[0].lower()


def test_agent_provider_timeout_delivery_keeps_fallback_guidance(hermes_env, monkeypatch):
    """Provider timeout classification remains available to agent-backed jobs."""
    from cron.jobs import create_job
    import cron.scheduler as scheduler

    job = create_job(
        prompt="Summarize the overnight logs.",
        schedule="every 5m",
        deliver="telegram",
        name="provider-backed report",
    )
    delivered = []

    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (
            False,
            "# Cron Job: provider-backed report\n\nprovider request timed out\n",
            "",
            "ReadTimeout: provider request timed out after fallback attempts",
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content),
    )

    assert scheduler.run_one_job(job) is True
    assert len(delivered) == 1
    assert "provider timeout" in delivered[0].lower()
    # Chain wording is now honest (#85508): exhausted when configured,
    # "no fallback chain configured" guidance otherwise.
    assert "fallback chain" in delivered[0].lower()


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

    def fake_scheduler_script(script_path, workdir=None, cancel_event=None):
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


def test_backend_process_cancellation_kills_tracked_process(monkeypatch):
    """Backend scripts stop promptly when the cron fire loses its claim."""
    import threading
    from types import SimpleNamespace
    from cron import scheduler
    import tools.process_registry

    waits = []
    kills = []
    fake_registry = SimpleNamespace(
        wait=lambda session_id, timeout: waits.append((session_id, timeout)) or {"status": "timeout"},
        kill_process=lambda session_id, source: kills.append((session_id, source)) or {"status": "killed"},
    )
    monkeypatch.setattr(tools.process_registry, "process_registry", fake_registry)

    cancel_event = threading.Event()
    cancel_event.set()
    success, output = scheduler._wait_for_backend_process(
        "proc_cancelled", 3600, cancel_event=cancel_event,
    )

    assert success is False
    assert output == "Backend script was cancelled"
    assert waits == []
    assert kills == [("proc_cancelled", "cron.script_cancelled")]


def test_backend_target_forwards_cancellation_to_backend_runner(monkeypatch):
    """The target router preserves cancellation for non-local backends."""
    import threading
    from cron import scheduler

    cancel_event = threading.Event()
    captured = {}
    monkeypatch.setattr(scheduler, "get_effective_terminal_backend", lambda: "docker")
    monkeypatch.setattr(
        scheduler,
        "_run_job_script_in_backend",
        lambda script_path, workdir=None, cancel_event=None: captured.update(
            script_path=script_path, workdir=workdir, cancel_event=cancel_event,
        ) or (True, "backend output"),
    )

    success, output = scheduler._run_job_script_for_target(
        {"target": "backend"}, "/workspace/worker.py", workdir="/workspace",
        cancel_event=cancel_event,
    )

    assert (success, output) == (True, "backend output")
    assert captured == {
        "script_path": "/workspace/worker.py",
        "workdir": "/workspace",
        "cancel_event": cancel_event,
    }


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
    crash with an unhandled exception.

    Regression (#86829): the assertion pins the *eager rejection* contract
    — the specific "NUL byte" report is only produced by the pre-check
    added in the fix. On Linux the legacy guard would swallow the
    expanduser() ValueError and report a generic invalid-path message, so
    a bare "Blocked" assertion could not tell the fixed code from the
    unfixed code; on Windows the unfixed code crashes outright."""
    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("~user\x00bad.sh")
    assert ok is False
    assert "NUL byte" in output


def test_run_job_script_nul_rejected_before_any_path_call(hermes_env, monkeypatch):
    """The eager NUL check must run before ``Path(...)`` is ever constructed.

    On Windows ``expanduser()`` never expands ``~user`` and never raises,
    so without the pre-check the NUL surfaces later from ``resolve()`` /
    ``exists()`` — outside the guard's try — and the uncaught ValueError
    crashes the scheduler (#86829). Stubbing ``Path`` with a hard failure
    proves the rejection happens before any pathlib call on every
    platform, not just the ones where expanduser happens to raise."""
    import cron.scheduler as scheduler_module

    def boom(*_args, **_kwargs):
        raise AssertionError("Path must not be touched for a NUL-bearing script path")

    monkeypatch.setattr(scheduler_module, "Path", boom)
    ok, output = scheduler_module._run_job_script("nul\x00byte.sh")
    assert ok is False
    assert "NUL byte" in output


def test_run_job_script_accepts_pathlike_script_path(hermes_env):
    """The eager NUL guard must not crash on a non-str script_path.

    ``"\x00" in script_path`` raises TypeError for a pathlib.Path (not
    iterable), so a Path passed by a future caller would crash the
    scheduler at the guard itself. The guard coerces with str() first;
    a valid Path must still run the script end-to-end (regression for
    the #86832 review point)."""
    from cron.scheduler import _run_job_script

    script = hermes_env / "scripts" / "probe.py"
    script.write_text('print("pathlike ok")\n', encoding="utf-8")

    ok, output = _run_job_script(pathlib.Path(script))
    assert ok is True
    assert "pathlike ok" in output


# ---------------------------------------------------------------------------
# _summarize_cron_failure_for_delivery: mode-aware failure attribution
# ---------------------------------------------------------------------------
#
# The summarizer classified failures by substring-matching the error prose and
# mapped any hit onto a provider-shaped explanation. For a no_agent job that is
# structurally impossible — run_job short-circuits before any model is reached —
# so a script whose own text happened to contain "timed out", "429" or
# "authentication" had its failure attributed to a provider it never called.
#
# Observed in practice: _run_job_script reports a timeout as "Script timed out
# after {n}s: {path}", which was delivered to chat as "provider timeout. Fallback
# chain was exhausted or unavailable." for a job that never opened a socket.
#
# The summarizer had no direct test coverage — the only test referencing it
# mocks it out and asserts on its arguments — which is why this shipped.


@pytest.mark.parametrize(
    "error",
    [
        "Script timed out after 900s: /home/u/.hermes/scripts/nightly.sh",
        "Script failed: curl returned 429 from api.example.com",
        "Script failed: gpg authentication failed for key",
        "Script failed: ReadTimeout contacting localhost",
    ],
)
def test_no_agent_failure_never_blamed_on_a_provider(error):
    """A script job's failure must never be reported as a provider/fallback failure."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "nightly-job", "no_agent": True, "script": "nightly.sh"}
    msg = _summarize_cron_failure_for_delivery(job, error)

    assert "provider" not in msg.lower()
    assert "fallback chain" not in msg.lower()
    # The operator must be pointed at what actually failed.
    assert "script" in msg.lower()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("ReadTimeout: provider did not respond", "provider timeout"),
        ("HTTP 429 rate limit exceeded", "provider rate limit"),
        ("HTTP 401 authentication failed", "provider authentication error"),
    ],
)
def test_agent_job_provider_classification_unchanged(error, expected):
    """Regression guard: agent-mode jobs keep the provider-shaped summaries."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "daily-digest", "no_agent": False}
    assert expected in _summarize_cron_failure_for_delivery(job, error)
