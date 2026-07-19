"""Tests for cron job script injection feature.

Tests cover:
- Script field in job creation / storage / update
- Script execution and output injection into prompts
- Error handling (missing script, timeout, non-zero exit)
- Path resolution (absolute, relative to HERMES_HOME/scripts/)
"""

import json
import os
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Clear cached module-level paths
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestJobScriptField:
    """Test that the script field is stored and retrieved correctly."""

    def test_create_job_with_script(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="Analyze the data",
            schedule="every 30m",
            script="/path/to/monitor.py",
        )
        assert job["script"] == "/path/to/monitor.py"

        loaded = get_job(job["id"])
        assert loaded["script"] == "/path/to/monitor.py"

    def test_create_job_without_script(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("script") is None

    def test_create_job_empty_script_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", script="  ")
        assert job.get("script") is None

    def test_update_job_add_script(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("script") is None

        updated = update_job(job["id"], {"script": "/new/script.py"})
        assert updated["script"] == "/new/script.py"

    def test_update_job_clear_script(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Hello", schedule="every 1h", script="/some/script.py")
        assert job["script"] == "/some/script.py"

        updated = update_job(job["id"], {"script": None})
        assert updated.get("script") is None


def test_cronjob_tool_rejects_stale_past_one_shot(cron_env, monkeypatch):
    from tools.cronjob_tools import cronjob

    now = datetime(2026, 3, 18, 4, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    stale = (now - timedelta(minutes=5)).isoformat()

    result = json.loads(cronjob(action="create", prompt="Too late", schedule=stale))

    assert result["success"] is False
    assert "past and cannot be scheduled" in result["error"]


class TestRunJobScript:
    """Test the _run_job_script() function."""

    def test_successful_script(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello from script")\n')

        success, output = _run_job_script(str(script))
        assert success is True
        assert output == "hello from script"

    def test_script_relative_path(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "relative.py"
        script.write_text('print("relative works")\n')

        success, output = _run_job_script("relative.py")
        assert success is True
        assert output == "relative works"

    def test_script_not_found(self, cron_env):
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("nonexistent_script.py")
        assert success is False
        assert "not found" in output.lower()

    def test_script_nonzero_exit(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "fail.py"
        script.write_text(textwrap.dedent("""\
            import sys
            print("partial output")
            print("error info", file=sys.stderr)
            sys.exit(1)
        """))

        success, output = _run_job_script(str(script))
        assert success is False
        assert "exited with code 1" in output
        assert "error info" in output

    def test_script_subprocess_env_sanitized(self, cron_env, monkeypatch):
        """Cron scripts must not inherit Hermes provider env (SECURITY.md §2.3)."""
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST
        from cron.scheduler import _run_job_script

        # sorted() so the probed var is deterministic across runs
        # (frozenset iteration order varies with PYTHONHASHSEED).
        blocked_var = sorted(_HERMES_PROVIDER_ENV_BLOCKLIST)[0]
        monkeypatch.setenv(blocked_var, "must_not_leak")

        script = cron_env / "scripts" / "env_probe.py"
        script.write_text(
            textwrap.dedent(
                f"""\
                import os
                key = {blocked_var!r}
                print("PRESENT" if os.environ.get(key) else "ABSENT")
                """
            )
        )

        success, output = _run_job_script("env_probe.py")
        assert success is True
        assert output == "ABSENT"

    def test_windows_uv_venv_python_script_bypasses_launcher(self, cron_env, tmp_path, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        venv = tmp_path / "venv"
        venv_scripts = venv / "Scripts"
        site_packages = venv / "Lib" / "site-packages"
        base = tmp_path / "base"
        venv_scripts.mkdir(parents=True)
        site_packages.mkdir(parents=True)
        base.mkdir()
        venv_python = venv_scripts / "python.exe"
        base_python = base / "python.exe"
        venv_python.write_text("", encoding="utf-8")
        base_python.write_text("", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text(f"home = {base}\nuv = true\n", encoding="utf-8")

        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            kwargs["stdout"].write(b"ok\n")
            return SimpleNamespace(
                wait=lambda timeout=None: 0,
                kill=lambda: None,
            )

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(sched_mod.sys, "executable", str(venv_python))
        monkeypatch.setattr(sched_mod, "windows_detach_flags", lambda: 0x09000208)
        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [str(base_python), str(script.resolve())]
        assert captured["kwargs"]["creationflags"] == 0x09000208
        assert captured["kwargs"]["stdin"] == sched_mod.subprocess.DEVNULL
        env = captured["kwargs"]["env"]
        assert env["VIRTUAL_ENV"] == str(venv)
        assert str(site_packages) in env["PYTHONPATH"]

    def test_windows_pythonw_script_uses_sibling_python_for_captured_output(self, cron_env, tmp_path, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        venv = tmp_path / "venv"
        venv_scripts = venv / "Scripts"
        venv_scripts.mkdir(parents=True)
        pythonw = venv_scripts / "pythonw.exe"
        python = venv_scripts / "python.exe"
        pythonw.write_text("", encoding="utf-8")
        python.write_text("", encoding="utf-8")

        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            # Undecodable byte: the runner must decode utf-8 with
            # errors="replace" (parity with the old encoding/errors kwargs).
            kwargs["stdout"].write("ok \u00e9\n".encode("utf-8") + b"\xff")
            return SimpleNamespace(
                wait=lambda timeout=None: 0,
                kill=lambda: None,
            )

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(sched_mod.sys, "executable", str(pythonw))
        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok \u00e9\n\ufffd".strip()
        assert captured["argv"] == [str(python), str(script.resolve())]

    def test_non_windows_script_preserves_default_text_decoding(self, cron_env, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.sys, "platform", "linux")
        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [sys.executable, str(script.resolve())]
        assert captured["kwargs"]["text"] is True
        assert "creationflags" not in captured["kwargs"]
        assert "encoding" not in captured["kwargs"]
        assert "errors" not in captured["kwargs"]

    def test_script_empty_output(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "empty.py"
        script.write_text("# no output\n")

        success, output = _run_job_script(str(script))
        assert success is True
        assert output == ""

    def test_script_timeout(self, cron_env, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        # Use a very short timeout
        monkeypatch.setattr(sched_mod, "_SCRIPT_TIMEOUT", 1)

        script = cron_env / "scripts" / "slow.py"
        script.write_text("import time; time.sleep(30)\n")

        success, output = _run_job_script(str(script))
        assert success is False
        assert "timed out" in output.lower()

    def test_script_json_output(self, cron_env):
        """Scripts can output structured JSON for the LLM to parse."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "json_out.py"
        script.write_text(textwrap.dedent("""\
            import json
            data = {"new_prs": [{"number": 42, "title": "Fix bug"}]}
            print(json.dumps(data, indent=2))
        """))

        success, output = _run_job_script(str(script))
        assert success is True
        parsed = json.loads(output)
        assert parsed["new_prs"][0]["number"] == 42


class TestBuildJobPromptWithScript:
    """Test that script output is injected into the prompt."""

    def test_script_output_injected(self, cron_env):
        from cron.scheduler import _build_job_prompt

        script = cron_env / "scripts" / "data.py"
        script.write_text('print("new PR: #123 fix typo")\n')

        job = {
            "prompt": "Report any notable changes.",
            "script": str(script),
        }
        prompt = _build_job_prompt(job)
        assert "## Script Output" in prompt
        assert "new PR: #123 fix typo" in prompt
        assert "Report any notable changes." in prompt

    def test_script_error_injected(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {
            "prompt": "Report status.",
            "script": "nonexistent_monitor.py",
        }
        prompt = _build_job_prompt(job)
        assert "## Script Error" in prompt
        assert "not found" in prompt.lower()
        assert "Report status." in prompt

    def test_no_script_unchanged(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {"prompt": "Simple job."}
        prompt = _build_job_prompt(job)
        assert "## Script Output" not in prompt
        assert "Simple job." in prompt



class TestCronjobToolScript:
    """Test the cronjob tool's script parameter."""

    def test_create_with_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
        ))
        assert result["success"] is True
        assert result["job"]["script"] == "monitor.py"

    def test_update_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script="new_script.py",
        ))
        assert update_result["success"] is True
        assert update_result["job"]["script"] == "new_script.py"

    def test_clear_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="some_script.py",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script="",
        ))
        assert update_result["success"] is True
        assert "script" not in update_result["job"]

    def test_list_shows_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="data_collector.py",
        )

        list_result = json.loads(cronjob(action="list"))
        assert list_result["success"] is True
        assert len(list_result["jobs"]) == 1
        assert list_result["jobs"][0]["script"] == "data_collector.py"


class TestScriptPathContainment:
    """Regression tests for path containment bypass in _run_job_script().

    Prior to the fix, absolute paths and ~-prefixed paths bypassed the
    scripts_dir containment check entirely, allowing arbitrary script
    execution through the cron system.
    """

    def test_absolute_path_outside_scripts_dir_blocked(self, cron_env):
        """Absolute paths outside ~/.hermes/scripts/ must be rejected."""
        from cron.scheduler import _run_job_script

        # Create a script outside the scripts dir
        outside_script = cron_env / "outside.py"
        outside_script.write_text('print("should not run")\n')

        success, output = _run_job_script(str(outside_script))
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_absolute_path_tmp_blocked(self, cron_env):
        """Absolute paths to /tmp must be rejected."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("/tmp/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_tilde_path_blocked(self, cron_env):
        """~ prefixed paths must be rejected (expanduser bypasses check)."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("~/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_tilde_traversal_blocked(self, cron_env):
        """~/../../../tmp/evil.py must be rejected."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("~/../../../tmp/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_traversal_still_blocked(self, cron_env):
        """../../etc/passwd style traversal must still be blocked."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("../../etc/passwd")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_path_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths within the scripts dir should still work."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "good.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("good.py")
        assert success is True
        assert output == "ok"

    def test_subdirectory_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths to subdirectories within scripts/ should work."""
        from cron.scheduler import _run_job_script

        subdir = cron_env / "scripts" / "monitors"
        subdir.mkdir()
        script = subdir / "check.py"
        script.write_text('print("sub ok")\n')

        success, output = _run_job_script("monitors/check.py")
        assert success is True
        assert output == "sub ok"

    def test_absolute_path_inside_scripts_dir_allowed(self, cron_env):
        """Absolute paths that resolve WITHIN scripts/ should work."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "abs_ok.py"
        script.write_text('print("abs ok")\n')

        success, output = _run_job_script(str(script))
        assert success is True
        assert output == "abs ok"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlink_escape_blocked(self, cron_env, tmp_path):
        """Symlinks pointing outside scripts/ must be rejected."""
        from cron.scheduler import _run_job_script

        # Create a script outside the scripts dir
        outside = tmp_path / "outside_evil.py"
        outside.write_text('print("escaped")\n')

        # Create a symlink inside scripts/ pointing outside
        link = cron_env / "scripts" / "sneaky.py"
        link.symlink_to(outside)

        success, output = _run_job_script("sneaky.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()


class TestCronjobToolScriptValidation:
    """Test API-boundary validation of cron script paths in cronjob_tools."""

    def test_create_with_absolute_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="/home/user/evil.py",
        ))
        assert result["success"] is False
        assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()

    def test_create_with_tilde_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="~/monitor.py",
        ))
        assert result["success"] is False
        assert "relative" in result["error"].lower() or "absolute" in result["error"].lower()

    def test_create_with_traversal_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="../../etc/passwd",
        ))
        assert result["success"] is False
        assert "escapes" in result["error"].lower() or "traversal" in result["error"].lower()

    def test_create_with_relative_script_allowed(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
        ))
        assert result["success"] is True
        assert result["job"]["script"] == "monitor.py"

    def test_update_with_absolute_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script="/tmp/evil.py",
        ))
        assert update_result["success"] is False
        assert "relative" in update_result["error"].lower() or "absolute" in update_result["error"].lower()

    def test_update_clear_script_allowed(self, cron_env, monkeypatch):
        """Clearing a script (empty string) should always be permitted."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script="",
        ))
        assert update_result["success"] is True
        assert "script" not in update_result["job"]

    def test_windows_absolute_path_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="C:\\Users\\evil\\script.py",
        ))
        assert result["success"] is False


class TestRunJobEnvVarCleanup:
    """Test that run_job() env vars are cleaned up even on early failure."""

    def test_env_vars_cleaned_on_early_error(self, cron_env, monkeypatch):
        """Origin env vars must be cleaned up even if run_job fails early."""
        # Ensure env vars are clean before test
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
        ):
            monkeypatch.delenv(key, raising=False)

        # Build a job with origin info that will fail during execution
        # (no valid model, no API key — will raise inside try block)
        job = {
            "id": "test-envleak",
            "name": "env-leak-test",
            "prompt": "test",
            "schedule_display": "every 1h",
            "origin": {
                "platform": "telegram",
                "chat_id": "12345",
                "chat_name": "Test Chat",
            },
        }

        from cron.scheduler import run_job

        # Expect it to fail (no model/API key), but env vars must be cleaned
        try:
            run_job(job)
        except Exception:
            pass

        # Verify env vars were cleaned up by the finally block
        assert os.environ.get("HERMES_SESSION_PLATFORM") is None
        assert os.environ.get("HERMES_SESSION_CHAT_ID") is None
        assert os.environ.get("HERMES_SESSION_CHAT_NAME") is None


class TestWindowsDetachedRunner:
    """Behavioral tests for ``_run_script_windows_detached`` (PR #43252).

    These launch REAL subprocesses. On POSIX the detach-flag helpers return
    0 by design ("no-ops on non-Windows"), so the launch mechanics --
    file-backed capture, constructor-only retry, timeout, at-most-once --
    are exercised identically on every platform; on native Windows the real
    creationflags are applied. Native job-object lifecycle behaviour is
    covered separately by tests/cron/test_windows_detach_native.py.
    """

    @staticmethod
    def _runner():
        from cron.scheduler import _run_script_windows_detached
        return _run_script_windows_detached

    def test_captures_stdout_stderr_and_returncode(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            "import sys\n"
            "print('to stdout')\n"
            "print('to stderr', file=sys.stderr)\n"
            "sys.exit(3)\n"
        )
        rc, out, err = self._runner()(
            [sys.executable, str(script)],
            timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 3
        assert out.strip() == "to stdout"
        assert err.strip() == "to stderr"

    def test_unicode_output_utf8(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.buffer.write('héllo 世界\\n'.encode('utf-8'))\n",
            encoding="utf-8",
        )
        rc, out, _ = self._runner()(
            [sys.executable, str(script)],
            timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 0
        assert out.strip() == "héllo 世界"

    def test_output_larger_than_pipe_buffer(self, tmp_path):
        # 2 MiB >> any anonymous-pipe buffer. A pipe-backed capture without
        # a live reader would deadlock or truncate; the file sink must not.
        script = tmp_path / "s.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.write('x' * (2 * 1024 * 1024))\n"
        )
        rc, out, _ = self._runner()(
            [sys.executable, str(script)],
            timeout=60, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 0
        assert len(out) == 2 * 1024 * 1024

    def test_empty_output(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("pass\n")
        rc, out, err = self._runner()(
            [sys.executable, str(script)],
            timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert (rc, out, err) == (0, "", "")

    def test_timeout_kills_child_and_never_retries(self, tmp_path, monkeypatch):
        import subprocess as sp
        from cron import scheduler as sched_mod

        spawn_count = {"n": 0}
        real_popen = sp.Popen

        def counting_popen(*args, **kwargs):
            spawn_count["n"] += 1
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(sched_mod.subprocess, "Popen", counting_popen)

        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)\n")
        with pytest.raises(sp.TimeoutExpired):
            self._runner()(
                [sys.executable, str(script)],
                timeout=1, cwd=str(tmp_path), env=os.environ.copy(),
            )
        assert spawn_count["n"] == 1

    def test_breakaway_denied_retries_constructor_exactly_once(
        self, tmp_path, monkeypatch
    ):
        """WinError 5 from the constructor -> one retry without breakaway,
        with every non-flag argument identical."""
        from cron import scheduler as sched_mod

        calls = []

        class _BreakawayDenied(OSError):
            winerror = 5  # ERROR_ACCESS_DENIED, as raised by CreateProcessW

        monkeypatch.setattr(sched_mod, "windows_detach_flags", lambda: 0x09000208)
        monkeypatch.setattr(
            sched_mod, "windows_detach_flags_without_breakaway", lambda: 0x08000208
        )
        monkeypatch.setattr(sched_mod, "_parent_in_job", lambda: True)

        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            if len(calls) == 1:
                raise _BreakawayDenied(5, "Access is denied")
            kwargs["stdout"].write(b"second attempt ran\n")
            from types import SimpleNamespace
            return SimpleNamespace(wait=lambda timeout=None: 0, kill=lambda: None)

        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)

        rc, out, _ = self._runner()(
            ["exe", "arg"], timeout=5, cwd=str(tmp_path), env={"A": "1"},
        )
        assert rc == 0
        assert out.strip() == "second attempt ran"
        assert len(calls) == 2
        assert calls[0][1]["creationflags"] == 0x09000208
        assert calls[1][1]["creationflags"] == 0x08000208
        # Everything except creationflags must be identical across attempts.
        for key in ("stdin", "stdout", "stderr", "cwd", "env"):
            assert calls[0][1][key] is calls[1][1][key] or (
                calls[0][1][key] == calls[1][1][key]
            )
        assert calls[0][0] == calls[1][0] == ["exe", "arg"]

    def test_non_breakaway_oserror_never_retries(self, tmp_path, monkeypatch):
        from cron import scheduler as sched_mod

        calls = []

        def fake_popen(argv, **kwargs):
            calls.append(argv)
            raise FileNotFoundError(2, "No such file")

        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)
        with pytest.raises(FileNotFoundError):
            self._runner()(
                ["missing-exe"], timeout=5, cwd=str(tmp_path), env={},
            )
        assert len(calls) == 1

    def test_post_spawn_oserror_never_spawns_second_child(
        self, tmp_path, monkeypatch
    ):
        """The Codex D1 scenario: the child performs a real side effect,
        then the post-spawn wait phase raises OSError. The rejected package
        launched a second child here; the reworked runner must not."""
        import subprocess as sp
        from cron import scheduler as sched_mod

        side_effect = tmp_path / "side_effect_count.txt"
        script = tmp_path / "s.py"
        script.write_text(
            "from pathlib import Path\n"
            f"p = Path({str(side_effect)!r})\n"
            "n = int(p.read_text()) if p.exists() else 0\n"
            "p.write_text(str(n + 1))\n"
            "print('completed run', n + 1)\n"
        )

        spawn_count = {"n": 0}
        real_popen = sp.Popen

        class _FaultyWaitPopen:
            def __init__(self, *args, **kwargs):
                spawn_count["n"] += 1
                self._proc = real_popen(*args, **kwargs)

            def wait(self, timeout=None):
                self._proc.wait(timeout=timeout)  # let the child finish...
                raise OSError(22, "Invalid argument")  # ...then inject fault

            def kill(self):
                self._proc.kill()

        monkeypatch.setattr(sched_mod.subprocess, "Popen", _FaultyWaitPopen)

        with pytest.raises(OSError):
            self._runner()(
                [sys.executable, str(script)],
                timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
            )
        assert spawn_count["n"] == 1
        assert side_effect.read_text() == "1"


    def test_access_denied_outside_job_never_retries(self, tmp_path, monkeypatch):
        """WinError 5 when the parent is in no job object cannot be
        breakaway denial (e.g. AV block, unexecutable file) — the runner
        must propagate rather than retry with weaker flags."""
        from cron import scheduler as sched_mod

        calls = []

        class _AccessDenied(OSError):
            winerror = 5

        monkeypatch.setattr(sched_mod, "_parent_in_job", lambda: False)

        def fake_popen(argv, **kwargs):
            calls.append(argv)
            raise _AccessDenied(5, "Access is denied")

        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)
        with pytest.raises(OSError):
            self._runner()(
                ["blocked-exe"], timeout=5, cwd=str(tmp_path), env={},
            )
        assert len(calls) == 1

    def test_dual_large_streams_no_deadlock(self, tmp_path):
        """1 MiB on BOTH streams simultaneously; a single-threaded pipe
        reader would deadlock, the file sinks must not."""
        script = tmp_path / "s.py"
        script.write_text(
            "import sys\n"
            "for _ in range(64):\n"
            "    sys.stdout.write('o' * 16384)\n"
            "    sys.stderr.write('e' * 16384)\n"
        )
        rc, out, err = self._runner()(
            [sys.executable, str(script)],
            timeout=60, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 0
        assert len(out) == 64 * 16384 and set(out) == {"o"}
        assert len(err) == 64 * 16384 and set(err) == {"e"}

    def test_grandchild_holding_stdout_does_not_block_return(self, tmp_path):
        """A script that daemonizes a worker (which inherits the output
        handle) must return when the DIRECT child exits. Current main's
        pipe capture blocks until pipe EOF here — a latent false-timeout
        (and, on Windows, a post-kill communicate() hang)."""
        import time as _time

        script = tmp_path / "s.py"
        script.write_text(
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print('done')\n"
        )
        t0 = _time.monotonic()
        rc, out, _ = self._runner()(
            [sys.executable, str(script)],
            timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 0
        assert out.strip() == "done"
        assert _time.monotonic() - t0 < 10, "returned only after grandchild exit"

    def test_nul_bytes_and_invalid_utf8_do_not_crash(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.buffer.write(b'a\\x00b\\xff\\xfec\\n')\n"
        )
        rc, out, _ = self._runner()(
            [sys.executable, str(script)],
            timeout=30, cwd=str(tmp_path), env=os.environ.copy(),
        )
        assert rc == 0
        assert "a\x00b" in out and "c" in out  # replaced, not raised

    def test_run_job_script_routes_win32_to_detached_runner(
        self, cron_env, monkeypatch
    ):
        """On win32, _run_job_script must use the detached runner and pass
        it the sanitized environment."""
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

        blocked_var = sorted(_HERMES_PROVIDER_ENV_BLOCKLIST)[0]
        monkeypatch.setenv(blocked_var, "must_not_leak")

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        captured = {}

        def fake_detached(argv, *, timeout, cwd, env):
            captured["argv"] = argv
            captured["env"] = env
            return 0, "ok\n", ""

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(
            sched_mod, "_run_script_windows_detached", fake_detached
        )
        # Keep interpreter resolution out of scope for this routing test.
        monkeypatch.setattr(
            sched_mod,
            "_windows_cron_python_invocation",
            lambda exe: (exe, {}),
        )

        success, output = _run_job_script("probe.py")
        assert success is True
        assert output == "ok"
        assert blocked_var not in captured["env"]
