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
import os
from pathlib import PurePosixPath, PureWindowsPath
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


def test_create_job_no_agent_stores_field(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "watchdog.sh"
    script_path.write_text("#!/bin/bash\necho hi\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["script"] == "watchdog.sh"
    # Prompt can be empty/None for no_agent jobs.
    assert job["prompt"] in {None, ""}


def test_create_job_default_is_not_no_agent(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="say hi", schedule="every 5m", deliver="local")
    assert job.get("no_agent") is False


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


def test_cronjob_tool_create_no_agent_with_script_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["no_agent"] is True
    assert result["job"]["script"] == "alert.sh"


def test_cronjob_tool_update_toggles_no_agent(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    off = json.loads(cronjob(action="update", job_id=job_id, no_agent=False, prompt="run"))
    assert off["success"] is True
    assert off["job"].get("no_agent") in {False, None}

    on = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert on["success"] is True
    assert on["job"]["no_agent"] is True


def test_cronjob_tool_update_no_agent_without_script_errors(hermes_env):
    """Flipping no_agent=True on a job that has no script must fail."""
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(action="create", schedule="every 5m", prompt="do a thing", deliver="local")
    )
    job_id = created["job_id"]

    result = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert result.get("success") is False
    assert "without a script" in result.get("error", "")


def test_cronjob_tool_create_does_not_require_prompt_when_no_agent(hermes_env):
    """The 'prompt or skill required' rule is relaxed for no_agent jobs."""
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True


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


def test_run_job_no_agent_empty_output_is_silent(hermes_env):
    """Empty stdout → SILENT_MARKER, which suppresses delivery downstream."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\n# nothing to say\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_wake_gate_is_silent(hermes_env):
    """wakeAgent=false gate in stdout triggers a silent run."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "gated.sh"
    script_path.write_text('#!/bin/bash\necho \'{"wakeAgent": false}\'\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="gated.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_script_failure_delivers_error(hermes_env):
    """Non-zero exit → success=False, error alert is the delivered message."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "broken.sh"
    script_path.write_text("#!/bin/bash\necho oops >&2\nexit 3\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="broken.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is False
    assert error is not None
    assert "oops" in final_response or "exited with code 3" in final_response
    assert "Cron watchdog" in final_response  # alert header


def test_run_job_no_agent_never_invokes_aiagent(hermes_env):
    """no_agent jobs must NOT import/construct the AIAgent."""
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )

    with patch("run_agent.AIAgent") as ai_mock:
        from cron.scheduler import run_job

        run_job(job)

    ai_mock.assert_not_called()


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_shell_script_runs_via_bash(hermes_env):
    """.sh files should execute under /bin/bash even without a shebang line."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "shelly.sh"
    # No shebang — relies on the interpreter-by-extension rule.
    script_path.write_text('echo "shell: $BASH_VERSION" | head -c 7\n')

    ok, output = _run_job_script("shelly.sh")
    assert ok is True
    assert output.startswith("shell:")


def test_run_job_script_bash_extension_also_runs_via_bash(hermes_env):
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "thing.bash"
    script_path.write_text('printf "via bash\\n"\n')

    ok, output = _run_job_script("thing.bash")
    assert ok is True
    assert output == "via bash"


def test_run_job_script_python_still_runs_via_python(hermes_env):
    """Regression: .py files must keep running via sys.executable."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "py.py"
    script_path.write_text("import sys\nprint(f'python {sys.version_info.major}')\n")

    ok, output = _run_job_script("py.py")
    assert ok is True
    assert output.startswith("python ")


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


# ---------------------------------------------------------------------------
# _resolve_bash: Windows Git Bash preference (issue #46332, Layer 1)
# ---------------------------------------------------------------------------


def test_resolve_bash_prefers_git_bash_over_wsl_on_windows(monkeypatch):
    """On Windows, Git for Windows bash must be chosen before WSL bash."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "win32")

    def _fake_isfile(p):
        # Git Bash 64-bit exists; WSL bash also "exists" on PATH.
        git_bash = os.path.join(r"C:\Program Files\Git\usr\bin", "bash.exe")
        return p == git_bash

    monkeypatch.setattr(sched.os.path, "isfile", _fake_isfile)
    monkeypatch.setattr(sched.shutil, "which", lambda cmd: r"C:\Windows\System32\bash.exe")

    result = sched._resolve_bash()
    assert result == os.path.join(r"C:\Program Files\Git\usr\bin", "bash.exe")


def test_resolve_bash_checks_32bit_git_bash_path(monkeypatch):
    """Fallback to 32-bit Git for Windows install path (edge case #3)."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "win32")

    def _fake_isfile(p):
        git_bash_32 = os.path.join(r"C:\Program Files (x86)\Git\usr\bin", "bash.exe")
        return p == git_bash_32

    monkeypatch.setattr(sched.os.path, "isfile", _fake_isfile)
    monkeypatch.setattr(sched.shutil, "which", lambda cmd: r"C:\Windows\System32\bash.exe")

    result = sched._resolve_bash()
    assert result == os.path.join(r"C:\Program Files (x86)\Git\usr\bin", "bash.exe")


def test_resolve_bash_falls_back_to_which_without_git_bash(monkeypatch):
    """Without Git for Windows, fall back to shutil.which (edge case #4)."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "win32")
    monkeypatch.setattr(sched.os.path, "isfile", lambda p: False)
    wsl_bash = r"C:\Windows\System32\bash.exe"
    monkeypatch.setattr(sched.shutil, "which", lambda cmd: wsl_bash)

    result = sched._resolve_bash()
    assert result == wsl_bash


def test_resolve_bash_uses_which_on_non_windows(monkeypatch):
    """On Linux/macOS, shutil.which('bash') is the primary resolver."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "linux")
    monkeypatch.setattr(sched.shutil, "which", lambda cmd: "/usr/bin/bash")

    result = sched._resolve_bash()
    assert result == "/usr/bin/bash"


# ---------------------------------------------------------------------------
# bash script path argument formatting (issue #46332, Layer 2)
# ---------------------------------------------------------------------------


def test_bash_script_path_arg_uses_forward_slashes_for_windows_paths(monkeypatch):
    """Windows bash script paths must be converted from backslashes."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "win32")
    path = PureWindowsPath(r"C:\Users\graem\AppData\Local\hermes\scripts\watchdog.sh")

    assert sched._bash_script_path_arg(path) == "C:/Users/graem/AppData/Local/hermes/scripts/watchdog.sh"


def test_bash_script_path_arg_keeps_native_paths_on_unix(monkeypatch):
    """Non-Windows platforms keep their native path formatting."""
    from cron import scheduler as sched

    monkeypatch.setattr(sched.sys, "platform", "linux")
    path = PurePosixPath("/home/graem/.hermes/scripts/watchdog.sh")

    assert sched._bash_script_path_arg(path) == str(path)


# ---------------------------------------------------------------------------
# _run_job_script: POSIX path conversion (issue #46332, Layer 2)
# ---------------------------------------------------------------------------


def test_run_job_script_sh_uses_posix_paths_on_windows(hermes_env, monkeypatch):
    r"""On Windows, .sh script paths must use forward slashes (issue #46332).

    ``str(path)`` on Windows yields ``C:\\Users\\...`` (backslashes). Git Bash
    (MSYS2) interprets each ``\`` as an escape, mangling the path to
    ``C:Users...`` and failing with exit 127. The fix uses ``as_posix()``.
    """
    from cron import scheduler as sched
    from cron.scheduler import _run_job_script

    # Simulate a nested script path so separators are meaningful.
    nested = hermes_env / "scripts" / "nested" / "dir"
    nested.mkdir(parents=True)
    script_path = nested / "watchdog.sh"
    script_path.write_text('echo "ok"\n')

    # Force Windows platform so the POSIX-path code path activates.
    monkeypatch.setattr(sched.sys, "platform", "win32")
    monkeypatch.setattr(sched, "_resolve_bash", lambda: r"C:\Program Files\Git\usr\bin\bash.exe")

    captured_argv = []

    def _fake_run(argv, **kwargs):
        captured_argv[:] = argv
        class _R:
            stdout = "ok\n"
            stderr = ""
            returncode = 0
        return _R()

    monkeypatch.setattr(sched.subprocess, "run", _fake_run)

    ok, output = _run_job_script("nested/dir/watchdog.sh")
    assert ok is True

    # The script-path argument (argv[1]) must NOT contain backslashes.
    script_arg = captured_argv[1]
    assert "\\" not in script_arg, f"path must not contain backslashes: {script_arg!r}"
    assert "/" in script_arg, f"path must contain forward slashes: {script_arg!r}"


def test_run_job_script_sh_keeps_native_paths_on_unix(hermes_env, monkeypatch):
    """On non-Windows, script paths are passed as-is (no POSIX conversion)."""
    from cron import scheduler as sched
    from cron.scheduler import _run_job_script

    nested = hermes_env / "scripts" / "nested"
    nested.mkdir(parents=True)
    script_path = nested / "watchdog.sh"
    script_path.write_text('echo "ok"\n')

    # Confirm we're NOT on Windows.
    assert sched.sys.platform != "win32"

    captured_argv = []

    def _fake_run(argv, **kwargs):
        captured_argv[:] = argv
        class _R:
            stdout = "ok\n"
            stderr = ""
            returncode = 0
        return _R()

    monkeypatch.setattr(sched.subprocess, "run", _fake_run)

    ok, output = _run_job_script("nested/watchdog.sh")
    assert ok is True

    # On Unix, str(path) is already a valid POSIX path.
    script_arg = captured_argv[1]
    assert script_arg.endswith("nested/watchdog.sh")
