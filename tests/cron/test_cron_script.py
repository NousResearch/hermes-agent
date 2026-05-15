"""Tests for cron job script injection feature.

Tests cover:
- Script field in job creation / storage / update
- Script execution and output injection into prompts
- Error handling (missing script, timeout, non-zero exit)
- Path resolution (absolute, relative to HERMES_HOME/scripts/)
"""

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

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


class TestRunJobScriptSubprocessEnv:
    """Test that subprocess.run receives a HOME-aware env.

    Mirrors agent.copilot_acp_client._build_subprocess_env coverage:
    when the gateway is launched from a launchd plist that nulls HOME,
    cron-job script subprocesses must still see a usable HOME so any
    ~/path.expanduser() in the script does not crash.
    """

    def test_subprocess_run_receives_env_with_home_when_parent_home_empty(
        self, cron_env, monkeypatch, tmp_path,
    ):
        """env= must be passed to subprocess.run with a non-empty HOME, even
        when the parent's HOME is empty (the production-incident scenario)."""
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        # Force the bug condition: parent gateway has no HOME (e.g. launchd
        # plist that did not set Environment HOME=...).  Without the fix,
        # subprocess.run would inherit HOME='' and child scripts would crash
        # on Path('~/foo').expanduser().
        monkeypatch.delenv("HOME", raising=False)
        # HERMES_HOME points at the cron_env fixture; create the home/
        # subdir so get_subprocess_home() returns a real path.
        hermes_home = cron_env  # cron_env IS the temp HERMES_HOME
        (hermes_home / "home").mkdir(exist_ok=True)

        captured = {}
        real_run = sched_mod.subprocess.run

        def _capture_run(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return real_run(*args, **kwargs)

        monkeypatch.setattr(sched_mod.subprocess, "run", _capture_run)

        script = cron_env / "scripts" / "noop.py"
        script.write_text("print('ok')\n")

        success, output = _run_job_script(str(script))
        assert success is True

        assert "env" in captured["kwargs"], (
            "subprocess.run for cron-job scripts must pass env= so "
            "child scripts get a stable HOME (matches the fix in "
            "agent.copilot_acp_client._run_prompt)."
        )
        env = captured["kwargs"]["env"]
        assert env.get("HOME"), (
            "_build_subprocess_env() must populate HOME from the resolver "
            "chain even when the parent process has HOME=''. Without this, "
            "child scripts inherit an empty HOME and crash on expanduser()."
        )

    def test_resolve_home_dir_prefers_profile_home_subdir_when_present(
        self, monkeypatch, tmp_path,
    ):
        """When HERMES_HOME is set AND {HERMES_HOME}/home/ exists,
        _resolve_home_dir() returns that subdirectory.

        get_subprocess_home() in hermes_constants only returns a path when
        the subdirectory exists on disk; otherwise it returns None and
        _resolve_home_dir falls through to the HOME env var."""
        from cron.scheduler import _resolve_home_dir

        hermes_home = tmp_path / "hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)

        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = _resolve_home_dir()
        assert result == str(profile_home), (
            f"Expected profile-scoped HOME at {profile_home}, got {result!r}"
        )

    def test_resolve_home_dir_falls_back_to_home_env_when_no_profile_subdir(
        self, monkeypatch, tmp_path,
    ):
        """When HERMES_HOME is set but {HERMES_HOME}/home/ does NOT exist,
        _resolve_home_dir() must fall through to $HOME (the common case for
        most installs that don't use the optional profile-home subdir)."""
        from cron.scheduler import _resolve_home_dir

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # Deliberately do NOT create hermes_home / "home" / — exercise the
        # fall-through path.

        fake_home = tmp_path / "user-home"
        fake_home.mkdir()

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = _resolve_home_dir()
        assert result == str(fake_home), (
            f"Expected fall-through to $HOME ({fake_home}) when "
            f"{hermes_home}/home/ is absent, got {result!r}"
        )

    def test_resolve_home_dir_falls_back_when_env_clean(self, monkeypatch):
        """No HOME, no HERMES_HOME → _resolve_home_dir still returns a usable dir."""
        from cron.scheduler import _resolve_home_dir

        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)

        result = _resolve_home_dir()
        assert result, "Fallback chain must return a non-empty path"
        assert result != "~", "Unexpanded ~ leak from os.path.expanduser fallback"

    def test_build_subprocess_env_does_not_mutate_os_environ(self, monkeypatch, tmp_path):
        """env dict is a copy — must not mutate process environment."""
        from cron.scheduler import _build_subprocess_env

        before = dict(os.environ)
        env = _build_subprocess_env()
        env["HOME"] = "/tmp/should-not-leak"
        after = dict(os.environ)
        assert before == after, (
            "_build_subprocess_env() must return a copy; mutations should not "
            "propagate back to os.environ"
        )


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
