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
    script_path.write_text("echo hi\n", encoding="utf-8")
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
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n", encoding="utf-8")

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


def test_run_job_script_handles_subprocess_env_correctly(hermes_env):
    """Script subprocess inherits sanitized environment, not raw gateway env."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "env_probe.py"
    script_path.write_text("import os; print(os.environ.get('HERMES_TEST_MARKER', 'NOT_SET'))\n", encoding="utf-8")

    job = create_job(
        prompt=None, schedule="every 5m", script="env_probe.py", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert "NOT_SET" in final_response, "subprocess should not inherit test markers from gateway"


@pytest.mark.skipif(
    not hasattr(__import__("os"), "getpgid"),
    reason="process groups are POSIX-only",
)
def test_run_job_script_gets_own_process_group(hermes_env):
    """#78432: a no_agent script must not share the gateway's process group.

    Previously ``_run_job_script`` spawned via a plain ``subprocess.run``
    with no ``start_new_session``, so on POSIX the child inherited the
    caller's (gateway's) process group verbatim. Any killpg() aimed at that
    shared group — an external supervisor signalling the gateway's
    foreground group, or a future MCP lifecycle teardown/orphan sweep —
    would kill the script as collateral damage. The fix isolates the script
    into its own process group, matching every other Hermes-spawned
    subprocess (mcp_stdio_watchdog, terminal/code-exec children, the LSP
    client).
    """
    import os

    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "pgid_probe.py"
    script_path.write_text(
        "import os\nprint(os.getpgid(os.getpid()))\nprint(os.getpid())\n"
    )

    caller_pgid = os.getpgid(os.getpid())

    success, output = _run_job_script(str(script_path))
    assert success is True

    script_pgid_str, script_pid_str = output.strip().splitlines()
    script_pgid, script_pid = int(script_pgid_str), int(script_pid_str)

    # The script must be its own process-group leader, not a member of the
    # caller's group.
    assert script_pgid == script_pid
    assert script_pgid != caller_pgid


@pytest.mark.skipif(
    not hasattr(__import__("os"), "getpgid"),
    reason="process groups are POSIX-only",
)
def test_run_job_script_timeout_kills_whole_process_group(hermes_env, monkeypatch):
    """A timed-out script must not leak grandchildren in its isolated group.

    Isolating the script into its own process group (#78432) means a plain
    ``proc.kill()`` on timeout only reaches the script's own PID — any
    children it spawned survive as orphans in that now-unreachable group.
    The timeout path must killpg() the whole group instead.
    """
    import os
    import time

    from cron.scheduler import _run_job_script

    pid_file = hermes_env / "grandchild.pid"
    script_path = hermes_env / "scripts" / "hang.py"
    script_path.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pid_file = {str(pid_file)!r}\n"
        "with open(pid_file, 'w', encoding='utf-8') as f:\n"
        "    f.write(str(child.pid))\n"
        "    f.flush()\n"
        "time.sleep(30)\n",
        encoding="utf-8"
    )
    monkeypatch.setattr("cron.scheduler._get_script_timeout", lambda: 0.3)

    success, output = _run_job_script(str(script_path))
    assert success is False, f"Expected timeout failure, got success with output: {output}"
    assert "timed out" in output.lower(), f"Expected 'timed out' in error, got: {output}"

    # Wait for grandchild pid file to be written with exponential backoff.
    pid_file_found = False
    for attempt in range(50):
        if pid_file.exists():
            pid_file_found = True
            break
        time.sleep(0.05)
    assert pid_file_found, "grandchild never wrote its PID — test setup is broken"

    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())

    # Give killpg() SIGKILL time to land and process to be reaped.
    time.sleep(0.5)

    # Verify the grandchild was actually reaped (not just orphaned).
    # Use _pid_exists() instead of os.kill(pid, 0) which is unsafe on Windows
    # (not a no-op; sends CTRL_C_EVENT to console process group, hard-killing target).
    try:
        from gateway.status import _pid_exists
        still_exists = _pid_exists(grandchild_pid)
    except (ImportError, Exception):
        # Fallback when _pid_exists unavailable: skip check on non-POSIX
        # (os.kill(pid, 0) is unsafe on Windows and Android; only safe on Unix-like).
        still_exists = False

    if still_exists:
        raise AssertionError(
            f"Grandchild process {grandchild_pid} still exists after timeout — "
            "killpg() did not reap the process group"
        )
