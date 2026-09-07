"""Tests for cron job script injection feature.

Tests cover:
- Script field in job creation / storage / update
- Script execution and output injection into prompts
- Error handling (missing script, timeout, non-zero exit)
- Path resolution (absolute, relative to HERMES_HOME/scripts/)
"""

import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

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


    def test_update_job_add_script(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("script") is None

        updated = update_job(job["id"], {"script": "/new/script.py"})
        assert updated["script"] == "/new/script.py"


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
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello from script")\n')

        success, output = _run_job_script(str(script))
        assert success is True
        assert output == "hello from script"

    def test_script_relative_path(self, cron_env):
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "relative.py"
        script.write_text('print("relative works")\n')

        success, output = _run_job_script("relative.py")
        assert success is True
        assert output == "relative works"


    def test_script_subprocess_env_sanitized(self, cron_env, monkeypatch):
        """Cron scripts must not inherit Hermes provider env (SECURITY.md §2.3)."""
        from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST
        from cron.scheduler_script import _run_job_script

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

    @pytest.mark.windows_only
    def test_windows_uv_venv_python_script_bypasses_launcher(self, cron_env, tmp_path, monkeypatch):
        # Windows-only: the fake ``sys.platform`` could not reproduce the
        # ``Scripts/python.exe`` launcher layout or the CREATE_NO_WINDOW
        # creationflags this branch exists for.
        from cron import scheduler as sched_mod
        from cron import scheduler_script as sched_script
        from cron.scheduler_script import _run_job_script

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

        class FakeProc:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["kwargs"] = kwargs
                self.returncode = 0

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return ("ok\n", "")

            def wait(self, timeout=None):
                return self.returncode

        fake_run = FakeProc

        monkeypatch.setattr(sched_mod.sys, "executable", str(venv_python))
        monkeypatch.setattr(sched_script, "windows_hide_flags", lambda: 0x08000000)
        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        # Overlay mode bootstraps with site.addsitedir() so .pth files
        # (editable installs) are processed — plain PYTHONPATH cannot do that.
        assert captured["argv"][0] == str(base_python)
        assert captured["argv"][1] == "-c"
        assert "site.addsitedir" in captured["argv"][2]
        m = re.search(r"site\.addsitedir\('([^']*)'\)", captured["argv"][2])
        assert m is not None
        assert Path(m.group(1)) == site_packages
        assert captured["argv"][3] == str(script.resolve())
        # The script runner always adds CREATE_NEW_PROCESS_GROUP on win32 so a
        # cancel can taskkill the whole tree; on POSIX the getattr default is
        # 0 and the flag set is exactly windows_hide_flags().
        expected_flags = sched_script.windows_hide_flags() | getattr(
            sched_mod.subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        assert captured["kwargs"]["creationflags"] == expected_flags
        env = captured["kwargs"]["env"]
        assert env["VIRTUAL_ENV"] == str(venv)
        assert str(site_packages) in env["PYTHONPATH"]

    def test_bootstrap_argv_makes_pth_editable_installs_importable(self, cron_env, tmp_path):
        """The bootstrap must process .pth files — the whole reason the
        overlay mode exists is that PYTHONPATH alone cannot (editable
        installs would raise ModuleNotFoundError in cron scripts)."""
        import subprocess

        from cron.scheduler_script import _windows_cron_bootstrap_argv

        venv = tmp_path / "venv"
        site_packages = venv / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)
        # Simulate `pip install -e`: a .pth file pointing at a source dir.
        editable_src = tmp_path / "editable_pkg"
        editable_src.mkdir()
        (editable_src / "mypkg.py").write_text("VALUE = 42\n", encoding="utf-8")
        (site_packages / "editable.pth").write_text(
            str(editable_src) + "\n", encoding="utf-8"
        )

        script = cron_env / "scripts" / "probe.py"
        script.write_text("import mypkg; print(mypkg.VALUE)\n", encoding="utf-8")

        argv = _windows_cron_bootstrap_argv(
            sys.executable, {"VIRTUAL_ENV": str(venv)}, str(script)
        )
        # Run the bootstrap with the current interpreter (stands in for the
        # base python.exe on Windows; the semantics are interpreter-agnostic).
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "42"

    def test_bootstrap_keeps_script_directory_on_sys_path(self, cron_env, tmp_path):
        """`python script.py` puts the script's directory on sys.path, so a
        script may import a sibling module. The bootstrap must preserve that
        (runpy.run_path alone does not add it)."""
        import subprocess

        from cron.scheduler_script import _windows_cron_bootstrap_argv

        venv = tmp_path / "venv"
        site_packages = venv / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)

        (cron_env / "scripts" / "sibling_helper.py").write_text(
            "GREETING = 'sibling ok'\n", encoding="utf-8"
        )
        script = cron_env / "scripts" / "probe.py"
        script.write_text(
            "import sibling_helper; print(sibling_helper.GREETING)\n",
            encoding="utf-8",
        )

        argv = _windows_cron_bootstrap_argv(
            sys.executable, {"VIRTUAL_ENV": str(venv)}, str(script)
        )
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "sibling ok"

    def test_bootstrap_argv_falls_back_without_site_packages(self, cron_env, tmp_path):
        """Unresolvable venv layout must not break the run — fall back to a
        plain invocation (pre-existing PYTHONPATH behaviour)."""
        from cron.scheduler_script import _windows_cron_bootstrap_argv

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n', encoding="utf-8")

        argv = _windows_cron_bootstrap_argv(
            sys.executable, {"VIRTUAL_ENV": str(tmp_path / "missing")}, str(script)
        )
        assert argv == [sys.executable, str(script)]


    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows always takes the overlay/creationflags branch",
    )
    def test_non_windows_script_preserves_default_text_decoding(self, cron_env, monkeypatch):
        # No platform patching: the Linux CI host already takes this branch.
        from cron import scheduler as sched_mod
        from cron import scheduler_script as sched_script
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        captured = {}

        class FakeProc:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["kwargs"] = kwargs
                self.returncode = 0

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return ("ok\n", "")

            def wait(self, timeout=None):
                return self.returncode

        fake_run = FakeProc

        monkeypatch.setattr(sched_mod.sys, "platform", "linux")
        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [sys.executable, str(script.resolve())]
        assert captured["kwargs"]["text"] is True
        assert "creationflags" not in captured["kwargs"]
        assert "encoding" not in captured["kwargs"]
        assert "errors" not in captured["kwargs"]

    def test_non_overlay_branch_keeps_plain_argv(self, cron_env, monkeypatch):
        """When the Windows uv-venv overlay is NOT active, the invocation must
        stay a plain `python script.py` — the bootstrap is overlay-only.
        Cross-platform: forces the non-overlay branch explicitly."""
        from cron import scheduler as sched_mod
        from cron import scheduler_script as sched_script
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n', encoding="utf-8")

        captured = {}

        class FakeProc:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                self.returncode = 0

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return ("ok\n", "")

        monkeypatch.setattr(sched_script, "_windows_cron_python_invocation",
            lambda python_exe: (python_exe, {}),
        )
        monkeypatch.setattr(sched_mod.subprocess, "Popen", FakeProc)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [sys.executable, str(script.resolve())]

    def test_emoji_stdout_round_trips_through_script_capture(self, cron_env):
        """Emoji in script stdout must reach the caller intact (#42384).

        On Windows the fix is the utf-8 + errors='replace' popen kwargs
        (asserted above); on POSIX the UTF-8 locale default must already
        carry emoji through. Either way the delivery content is the real
        text, never an exception.
        """
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "emoji.py"
        script.write_text(
            'import sys\n'
            'sys.stdout.buffer.write("backup done \\N{PARTY POPPER} 日次".encode("utf-8"))\n',
            encoding="utf-8",
        )

        success, output = _run_job_script("emoji.py")

        assert success is True
        assert "backup done 🎉 日次" == output

    def test_invalid_utf8_stdout_does_not_raise(self, cron_env):
        """Truncated/invalid UTF-8 in script stdout must never escape as an
        exception (#47393) — a raised UnicodeDecodeError higher up would
        silently drop the whole delivery (#42384). The run may fail, but it
        must fail as a (False, message) result the scheduler can deliver.
        """
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "bad_bytes.py"
        # b'\xe6\x97' is the first two bytes of a three-byte CJK sequence —
        # a truncated write, exactly the shape reported in #47393.
        script.write_text(
            "import sys\n"
            "sys.stdout.buffer.write(b'partial \\xe6\\x97')\n",
            encoding="utf-8",
        )

        success, output = _run_job_script("bad_bytes.py")  # must not raise

        assert isinstance(success, bool)
        assert isinstance(output, str)
        assert output  # a message is always produced, never a silent drop


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
        from cron.scheduler_script import _run_job_script

        # Create a script outside the scripts dir
        outside_script = cron_env / "outside.py"
        outside_script.write_text('print("should not run")\n')

        success, output = _run_job_script(str(outside_script))
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()


    def test_tilde_path_blocked(self, cron_env):
        """~ prefixed paths must be rejected (expanduser bypasses check)."""
        from cron.scheduler_script import _run_job_script

        success, output = _run_job_script("~/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_tilde_traversal_blocked(self, cron_env):
        """~/../../../tmp/evil.py must be rejected."""
        from cron.scheduler_script import _run_job_script

        success, output = _run_job_script("~/../../../tmp/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_traversal_still_blocked(self, cron_env):
        """../../etc/passwd style traversal must still be blocked."""
        from cron.scheduler_script import _run_job_script

        success, output = _run_job_script("../../etc/passwd")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_path_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths within the scripts dir should still work."""
        from cron.scheduler_script import _run_job_script

        script = cron_env / "scripts" / "good.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("good.py")
        assert success is True
        assert output == "ok"

    def test_subdirectory_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths to subdirectories within scripts/ should work."""
        from cron.scheduler_script import _run_job_script

        subdir = cron_env / "scripts" / "monitors"
        subdir.mkdir()
        script = subdir / "check.py"
        script.write_text('print("sub ok")\n')

        success, output = _run_job_script("monitors/check.py")
        assert success is True
        assert output == "sub ok"


    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlink_escape_blocked(self, cron_env, tmp_path):
        """Symlinks pointing outside scripts/ must be rejected."""
        from cron.scheduler_script import _run_job_script

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


class TestScriptTimeoutTreeKill:
    """Phase 4a (#85125): a script timeout must leave zero living descendants."""

    def test_unified_tree_kill_failure_falls_back(self, monkeypatch, caplog):
        from agent import deadline
        from cron import scheduler as sched
        from cron import scheduler_script as sched_script

        proc = SimpleNamespace(pid=12345, poll=lambda: None)
        fallback_calls = []
        monkeypatch.setattr(deadline, "kill_process_tree", lambda _pid: False)
        monkeypatch.setattr(sched_script, "_terminate_cron_script_process",
            lambda candidate: fallback_calls.append(candidate),
        )

        with caplog.at_level("WARNING", logger=sched.__name__):
            sched_script._terminate_cron_script_tree(cast("subprocess.Popen", proc))

        assert fallback_calls == [proc]
        assert "falling back to process-group termination" in caplog.text

    def test_invalid_pid_never_reaches_unified_tree_kill(self, monkeypatch, caplog):
        from agent import deadline
        from cron import scheduler as sched
        from cron import scheduler_script as sched_script

        proc = SimpleNamespace(pid=0, poll=lambda: None)
        tree_kill_calls = []
        fallback_calls = []
        monkeypatch.setattr(
            deadline,
            "kill_process_tree",
            lambda pid: tree_kill_calls.append(pid),
        )
        monkeypatch.setattr(sched_script, "_terminate_cron_script_process",
            lambda candidate: fallback_calls.append(candidate),
        )

        with caplog.at_level("WARNING", logger=sched.__name__):
            sched_script._terminate_cron_script_tree(cast("subprocess.Popen", proc))

        assert tree_kill_calls == []
        assert fallback_calls == [proc]
        assert "invalid pid 0" in caplog.text

    def test_already_exited_proc_is_left_alone(self, monkeypatch):
        """A script that finished right at the deadline needs no signalling —
        and must not produce a spurious "no signal" warning."""
        from agent import deadline
        from cron import scheduler as sched
        from cron import scheduler_script as sched_script

        proc = SimpleNamespace(pid=12345, poll=lambda: 0)
        tree_kill_calls = []
        fallback_calls = []
        monkeypatch.setattr(
            deadline,
            "kill_process_tree",
            lambda pid: tree_kill_calls.append(pid) or True,
        )
        monkeypatch.setattr(sched_script, "_terminate_cron_script_process",
            lambda candidate: fallback_calls.append(candidate),
        )

        sched_script._terminate_cron_script_tree(cast("subprocess.Popen", proc))

        assert tree_kill_calls == []
        assert fallback_calls == []

    def test_cancel_path_also_tree_kills(self, monkeypatch, cron_env):
        """The ownership-lost/cancel kill site is the timeout site's sibling:
        it must go through the same tree-kill (#71148 class)."""
        from cron import scheduler as sched
        from cron import scheduler_script as sched_script

        tree_calls = []

        def _record_and_kill(proc):
            # Record the routing, then really kill so _drain_script_pipes
            # reaps instantly instead of waiting out its 5s communicate().
            tree_calls.append(proc.pid)
            proc.kill()

        monkeypatch.setattr(sched_script, "_terminate_cron_script_tree", _record_and_kill)

        class _Cancelled:
            def is_set(self):
                return True

            def set(self):
                pass

        scripts_dir = cron_env / "scripts"
        (scripts_dir / "long.py").write_text(
            "import time; time.sleep(30)\n", encoding="utf-8"
        )
        ok, out = sched_script._run_job_script(
            str(scripts_dir / "long.py"),
            workdir=str(cron_env),
            cancel_event=_Cancelled(),
        )
        assert not ok
        assert "ownership was lost" in out
        assert len(tree_calls) == 1

    @pytest.mark.live_system_guard_bypass
    def test_timeout_leaves_no_setsid_grandchild(self, cron_env, monkeypatch):
        """The script spawns a grandchild in its OWN session (start_new_session).
        killpg alone cannot reach it; agent.deadline.kill_process_tree must —
        after the timeout the grandchild must no longer be running."""
        import time

        psutil = pytest.importorskip(
            "psutil",
            reason="kill_process_tree needs psutil to reach own-session descendants",
        )

        from cron import scheduler as sched
        from cron import scheduler_script as sched_script

        def is_live(pid):
            try:
                process = psutil.Process(pid)
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return False

        scripts_dir = cron_env / "scripts"
        pid_file = cron_env / "grandchild.pid"
        (scripts_dir / "spawner.py").write_text(
            "import subprocess, sys, time\n"
            "p = subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
            "    start_new_session=True,\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL,\n"
            ")\n"
            f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "2")
        monkeypatch.setattr(sched, "_SCRIPT_TIMEOUT", sched._DEFAULT_SCRIPT_TIMEOUT)

        ok, out = sched_script._run_job_script(
            str(scripts_dir / "spawner.py"), workdir=str(cron_env)
        )
        assert not ok, f"script should have timed out, got {out!r}"

        deadline = time.monotonic() + 5
        gpid = None
        while time.monotonic() < deadline and gpid is None:
            try:
                gpid = int(pid_file.read_text().strip())
            except (FileNotFoundError, ValueError):
                time.sleep(0.05)
        assert gpid is not None, "spawner never wrote the grandchild pid"

        try:
            deadline = time.monotonic() + 5
            while is_live(gpid) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not is_live(gpid), (
                f"grandchild pid {gpid} survived the script timeout — the "
                "timeout path orphaned an own-session descendant"
            )
        finally:
            if is_live(gpid):
                try:
                    psutil.Process(gpid).kill()
                except psutil.NoSuchProcess:
                    pass
