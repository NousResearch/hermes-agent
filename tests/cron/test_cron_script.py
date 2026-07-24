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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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

    # --- interpreter field persistence ------------------------------------

    def test_create_job_with_interpreter(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="Analyze the data",
            schedule="every 30m",
            script="monitor.py",
            interpreter="~/workspace/.venv/bin/python3",
        )
        assert job["interpreter"] == "~/workspace/.venv/bin/python3"

        loaded = get_job(job["id"])
        assert loaded["interpreter"] == "~/workspace/.venv/bin/python3"

    def test_create_job_without_interpreter_has_no_field(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", script="monitor.py")
        # Absent (not a falsy sentinel) so existing records stay byte-identical.
        assert "interpreter" not in job

    def test_no_agent_positional_arg_not_shifted_by_interpreter(self, cron_env):
        """The new ``interpreter`` param must not break positional callers.

        ``interpreter`` is appended AFTER ``no_agent``/``attach_to_session`` so
        an existing positional call like ``create_job(p, sched, script=..., True)``
        keeps binding ``True`` to ``no_agent`` — not silently to ``interpreter``,
        which would flip a no-agent watchdog into an LLM job.
        """
        import inspect
        from cron.jobs import create_job

        params = list(inspect.signature(create_job).parameters)
        # interpreter must come after both no_agent and attach_to_session.
        assert params.index("interpreter") > params.index("no_agent")
        assert params.index("interpreter") > params.index("attach_to_session")

        # And the real positional call must still set no_agent=True.
        job = create_job(None, "every 5m", script="w.sh", no_agent=True)
        assert job["no_agent"] is True
        assert "interpreter" not in job

    def test_create_job_interpreter_whitespace_trimmed(self, cron_env):
        from cron.jobs import create_job

        job = create_job(
            prompt="Hello", schedule="every 1h", script="monitor.py",
            interpreter="  /opt/venv/bin/python  ",
        )
        assert job["interpreter"] == "/opt/venv/bin/python"

    def test_create_job_empty_interpreter_normalized_to_absent(self, cron_env):
        from cron.jobs import create_job

        job = create_job(
            prompt="Hello", schedule="every 1h", script="monitor.py",
            interpreter="   ",
        )
        assert "interpreter" not in job

    def test_update_job_set_interpreter(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Hello", schedule="every 1h", script="monitor.py")
        assert "interpreter" not in job

        updated = update_job(job["id"], {"interpreter": "/opt/venv/bin/python"})
        assert updated["interpreter"] == "/opt/venv/bin/python"

    def test_update_job_clear_interpreter(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(
            prompt="Hello", schedule="every 1h", script="monitor.py",
            interpreter="/opt/venv/bin/python",
        )
        assert job["interpreter"] == "/opt/venv/bin/python"

        updated = update_job(job["id"], {"interpreter": ""})
        assert "interpreter" not in updated

    def test_update_job_preserves_interpreter_when_absent(self, cron_env):
        """A partial update that does not mention interpreter must keep it.

        Desktop/web editors send only the fields they render; the field is
        load-bearing for that preservation contract.
        """
        from cron.jobs import create_job, update_job

        job = create_job(
            prompt="Hello", schedule="every 1h", script="monitor.py",
            interpreter="/opt/venv/bin/python",
        )
        updated = update_job(job["id"], {"prompt": "New prompt"})
        assert updated["interpreter"] == "/opt/venv/bin/python"
        assert updated["prompt"] == "New prompt"


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

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(sched_mod.sys, "executable", str(venv_python))
        monkeypatch.setattr(sched_mod, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [str(base_python), str(script.resolve())]
        assert captured["kwargs"]["creationflags"] == 0x08000000
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

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(sched_mod.sys, "executable", str(pythonw))
        monkeypatch.setattr(sched_mod, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [str(python), str(script.resolve())]
        assert captured["kwargs"]["encoding"] == "utf-8"
        assert captured["kwargs"]["errors"] == "replace"

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

    # --- interpreter selection / validation --------------------------------

    def test_default_python_uses_sys_executable(self, cron_env, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")
        assert success is True
        assert captured["argv"][0] == sys.executable

    def test_script_uses_configured_interpreter(self, cron_env):
        """An external interpreter (absolute, executable) replaces sys.executable."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "uses_env.py"
        script.write_text(textwrap.dedent("""\
            import os
            print(os.environ.get("CRON_WRAPPER_USED", "0"))
        """))

        # A wrapper that re-execs the real interpreter with an env marker so we
        # can prove it was the one Hermes invoked.
        wrapper = cron_env / "scripts" / "python-wrapper"
        wrapper.write_text(textwrap.dedent(f"""\
            #!{sys.executable}
            import os
            import sys

            env = os.environ.copy()
            env["CRON_WRAPPER_USED"] = "1"
            os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)
        """))
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

        success, output = _run_job_script(str(script), interpreter=str(wrapper))
        assert success is True
        assert output == "1"

    def test_script_uses_external_interpreter_importing_local_module(self, cron_env, tmp_path):
        """A Python script can import a module only available to the external env.

        Builds a throwaway stdlib venv (no network) and plants a tiny test-only
        module into its site-packages, then asserts the cron script — run via
        that interpreter — can import it.
        """
        import subprocess
        import venv as _venv

        from cron.scheduler import _run_job_script

        ext_venv = tmp_path / "extvenv"
        _venv.EnvBuilder(with_pip=False, symlinks=True, clear=True).create(str(ext_venv))
        ext_python = ext_venv / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        assert ext_python.exists(), "external venv python was not created"

        # Plant a test-only module into the venv's site-packages so only that
        # interpreter can import it (Hermes' own interpreter cannot).
        site_dir = subprocess.run(
            [str(ext_python), "-c",
             "import site,sys; print(next(p for p in site.getsitepackages() if 'packages' in p))"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        Path(site_dir, "cron_ext_marker.py").write_text("VALUE = 'from-external-venv'\n")

        script = cron_env / "scripts" / "importer.py"
        script.write_text(textwrap.dedent("""\
            from cron_ext_marker import VALUE
            print(VALUE)
        """))

        success, output = _run_job_script(str(script), interpreter=str(ext_python))
        assert success is True
        assert output == "from-external-venv"

    def test_interpreter_tilde_expanded_at_run_time(self, cron_env, monkeypatch, tmp_path):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        # Put the interpreter under a temp HOME so "~" expands to tmp_path
        # (expanduser reads the HOME env var, not Path.home()).
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        real_python = fake_home / "python-bin"
        real_python.write_text("", encoding="utf-8")
        real_python.chmod(real_python.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("HOME", str(fake_home))

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py", interpreter="~/python-bin")
        assert success is True
        assert captured["argv"][0] == str(real_python)

    def test_interpreter_relative_path_rejected(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("probe.py", interpreter="python3")
        assert success is False
        assert "absolute" in output.lower()

    def test_interpreter_missing_rejected(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("probe.py", interpreter="/no/such/python")
        assert success is False
        assert "interpreter" in output.lower()

    def test_interpreter_directory_rejected(self, cron_env, tmp_path):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("probe.py", interpreter=str(tmp_path))
        assert success is False
        assert "not a file" in output.lower()

    def test_interpreter_non_executable_rejected_on_posix(self, cron_env):
        from cron.scheduler import _run_job_script

        if sys.platform == "win32":
            pytest.skip("executable bit is a POSIX-only check")

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        not_exec = cron_env / "scripts" / "python-noexec"
        not_exec.write_text("", encoding="utf-8")
        not_exec.chmod(0o644)  # explicitly NOT executable

        success, output = _run_job_script("probe.py", interpreter=str(not_exec))
        assert success is False
        assert "execut" in output.lower()

    def test_interpreter_unknown_user_does_not_raise(self, cron_env):
        """A ``~unknown-user/...`` interpreter must NOT escape as a RuntimeError.

        ``Path.expanduser()`` raises ``RuntimeError: Could not determine home
        directory`` for an unknown user. The resolver must convert that into its
        ``(False, message)`` contract so the cron tick surfaces a clear
        scheduled-run failure instead of crashing the scheduler.
        """
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        # An extremely unlikely-to-exist user name so expanduser() fails to
        # resolve a home directory regardless of the test host.
        success, output = _run_job_script(
            "probe.py", interpreter="~definitely-no-such-user-zzz/python"
        )
        assert success is False
        assert "interpreter" in output.lower() or "resolve" in output.lower()

    def test_interpreter_ignored_for_shell_scripts(self, cron_env):
        """``.sh``/``.bash`` always run under bash; the interpreter override
        is a Python-script-only contract."""
        from cron.scheduler import _run_job_script

        # A shell script that prints which interpreter ran it would be ideal,
        # but the simpler contract is: a present interpreter does not break the
        # bash path, and bash still executes the script.
        script = cron_env / "scripts" / "shelly.sh"
        script.write_text("#!/bin/bash\necho 'shell ran'\n")

        success, output = _run_job_script("shelly.sh", interpreter="/opt/venv/bin/python")
        assert success is True
        assert output == "shell ran"

    def test_configured_interpreter_passes_through_windows_helper(self, cron_env, tmp_path, monkeypatch):
        """The configured interpreter must still flow through the Windows
        invocation helper (output capture / no-window behavior)."""
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

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.sys, "platform", "win32")
        monkeypatch.setattr(sched_mod, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py", interpreter=str(venv_python))

        assert success is True
        assert output == "ok"
        # The helper bypassed the uv launcher and ran the base python directly.
        assert captured["argv"][0] == str(base_python)
        assert captured["kwargs"]["creationflags"] == 0x08000000
        assert captured["kwargs"]["env"]["VIRTUAL_ENV"] == str(venv)


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

    def test_build_job_prompt_passes_interpreter(self, cron_env):
        """The inline prompt-build path must forward the interpreter to
        ``_run_job_script`` (the path used when no precomputed result is
        handed in)."""
        from cron.scheduler import _build_job_prompt

        script = cron_env / "scripts" / "collector.py"
        script.write_text('print("ok")\n')

        job = {
            "prompt": "Report any notable changes.",
            "script": str(script),
            "interpreter": "~/workspace/.venv/bin/python3",
        }

        with patch("cron.scheduler._run_job_script", return_value=(True, "ok")) as mock_run:
            prompt = _build_job_prompt(job)

        mock_run.assert_called_once_with(
            str(script), interpreter="~/workspace/.venv/bin/python3"
        )
        assert "## Script Output" in prompt


class TestInterpreterPropagationThroughRunJob:
    """The job-level interpreter must reach the script runner on every path."""

    def test_run_job_wake_gate_passes_interpreter(self, cron_env):
        from cron.scheduler import SILENT_MARKER, run_job

        job = {
            "id": "abc123def456",
            "name": "wake-gated-job",
            "prompt": "Report status.",
            "schedule_display": "every 1h",
            "script": "gate.py",
            "interpreter": "~/workspace/.venv/bin/python3",
        }

        with patch(
            "cron.scheduler._run_job_script_with_claim_heartbeat",
            return_value=(True, '{"wakeAgent": false}'),
        ) as mock_run:
            success, output, final_response, error = run_job(job)

        mock_run.assert_called_once()
        # The wrapper receives the whole job; the interpreter is read from it.
        _args, kwargs = mock_run.call_args
        passed_job = kwargs.get("job") or (_args[0] if _args else None)
        assert passed_job is not None
        assert passed_job.get("interpreter") == "~/workspace/.venv/bin/python3"
        assert success is True
        assert "wakeAgent=false" in output
        assert final_response == SILENT_MARKER
        assert error is None

    def test_claim_heartbeat_forwards_interpreter_to_runner(self, cron_env):
        """The claim-heartbeat wrapper reads the interpreter off the job and
        passes it down to ``_run_job_script`` on every internal call path."""
        from cron.scheduler import _run_job_script_with_claim_heartbeat

        # Recurring job → no durable one-shot claim → plain passthrough path.
        job = {
            "id": "rec123abc456",
            "schedule": {"kind": "interval", "minutes": 5},
            "script": "monitor.py",
            "interpreter": "/opt/venv/bin/python",
        }

        with patch("cron.scheduler._run_job_script", return_value=(True, "ok")) as mock_run:
            success, output = _run_job_script_with_claim_heartbeat(job, "monitor.py")

        assert success is True
        assert output == "ok"
        mock_run.assert_called_once_with(
            "monitor.py", interpreter="/opt/venv/bin/python"
        )

    def test_one_shot_heartbeat_forwards_interpreter_to_runner(self, cron_env, monkeypatch):
        """The one-shot claim-heartbeat path (the threaded branch) must also
        forward the interpreter to ``_run_job_script``."""
        from cron.scheduler import _run_job_script_with_claim_heartbeat

        # A claimed one-shot job triggers the heartbeat-threaded branch.
        job = {
            "id": "once123abc456",
            "schedule": {"kind": "once"},
            "script": "oneshot.py",
            "interpreter": "/opt/venv/bin/python",
            "run_claim": {"by": "owner-xyz"},
        }
        # Keep the heartbeat snappy and no-op so the test doesn't hit storage.
        monkeypatch.setattr("cron.scheduler._RUN_CLAIM_HEARTBEAT_SECONDS", 3600)
        monkeypatch.setattr("cron.scheduler.heartbeat_run_claim", lambda *a, **k: None)

        with patch("cron.scheduler._run_job_script", return_value=(True, "ok")) as mock_run:
            success, output = _run_job_script_with_claim_heartbeat(job, "oneshot.py")

        assert success is True
        assert output == "ok"
        mock_run.assert_called_once_with(
            "oneshot.py", interpreter="/opt/venv/bin/python"
        )

    def test_heartbeat_start_failure_falls_back_with_interpreter(self, cron_env, monkeypatch):
        """When the heartbeat thread cannot start, the wrapper falls back to a
        direct ``_run_job_script`` call — the interpreter must survive that
        fallback path too."""
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script_with_claim_heartbeat

        job = {
            "id": "once456abc789",
            "schedule": {"kind": "once"},
            "script": "oneshot.py",
            "interpreter": "/opt/venv/bin/python",
            "run_claim": {"by": "owner-xyz"},
        }
        monkeypatch.setattr("cron.scheduler._RUN_CLAIM_HEARTBEAT_SECONDS", 3600)
        monkeypatch.setattr("cron.scheduler.heartbeat_run_claim", lambda *a, **k: None)

        # Force Thread.start() to raise so the wrapper takes the fallback branch.
        real_thread = sched_mod.threading.Thread

        class _BrokenThread(real_thread):
            def start(self):
                raise RuntimeError("cannot start thread")

        monkeypatch.setattr(sched_mod.threading, "Thread", _BrokenThread)

        with patch("cron.scheduler._run_job_script", return_value=(True, "ok")) as mock_run:
            success, output = _run_job_script_with_claim_heartbeat(job, "oneshot.py")

        assert success is True
        assert output == "ok"
        mock_run.assert_called_once_with(
            "oneshot.py", interpreter="/opt/venv/bin/python"
        )


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

    def test_create_with_interpreter(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
            interpreter="~/venvs/reporting/bin/python3",
        ))
        assert result["success"] is True
        assert result["job"]["script"] == "monitor.py"
        assert result["job"]["interpreter"] == "~/venvs/reporting/bin/python3"

    def test_cronjob_positional_task_id_not_shifted_by_interpreter(self, cron_env, monkeypatch):
        """``interpreter`` must not break positional callers of ``cronjob()``.

        The legacy signature ended in ``no_agent, attach_to_session, task_id``.
        A positional caller passing a trailing ``task_id`` would otherwise bind
        it onto ``interpreter`` (and leave ``task_id`` as None), which on a
        create/update could store the task id as an interpreter path and break
        the script run. ``interpreter`` is appended AFTER ``task_id`` so the
        old positional slots keep their meaning.
        """
        import inspect
        from tools.cronjob_tools import cronjob

        params = list(inspect.signature(cronjob).parameters)
        assert params.index("interpreter") > params.index("task_id")

        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        # Simulate a legacy positional call: fill slots up to and including
        # task_id (index 20). interpreter (now index 21) stays default.
        # If interpreter wrongly preceded task_id, "legacy-task-id" would be
        # stored as the interpreter and the job would carry a bogus field.
        positional = [None] * (params.index("task_id") + 1)
        positional[0] = "create"            # action
        positional[2] = "Monitor things"    # prompt
        positional[3] = "every 1h"          # schedule
        positional[14] = "monitor.py"       # script
        positional[params.index("task_id")] = "legacy-task-id"
        result = json.loads(cronjob(*positional))
        assert result["success"] is True
        # No interpreter was supplied; the task id must not leak into the job.
        assert "interpreter" not in result["job"]

    def test_update_interpreter(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
        ))
        job_id = create_result["job_id"]
        assert "interpreter" not in create_result["job"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            interpreter="~/venvs/reporting/bin/python3",
        ))
        assert update_result["success"] is True
        assert update_result["job"]["interpreter"] == "~/venvs/reporting/bin/python3"

    def test_clear_interpreter(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
            interpreter="~/venvs/reporting/bin/python3",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            interpreter="",
        ))
        assert update_result["success"] is True
        assert "interpreter" not in update_result["job"]

    def test_list_shows_interpreter(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="monitor.py",
            interpreter="~/venvs/reporting/bin/python3",
        )

        list_result = json.loads(cronjob(action="list"))
        assert list_result["success"] is True
        assert list_result["jobs"][0]["interpreter"] == "~/venvs/reporting/bin/python3"


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
