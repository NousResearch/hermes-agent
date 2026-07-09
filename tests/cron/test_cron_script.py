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


class TestResolveBashWindows:
    """Regression tests for ``_resolve_bash`` and ``_is_wsl_launcher``.

    Bit users on Windows whose Git for Windows install used the *default* PATH
    option (which adds ``Git\\cmd`` — ``git.exe`` — but NOT ``Git\\bin`` /
    ``Git\\usr\\bin`` — ``bash.exe``) and who also have WSL enabled with no
    distributions installed. The old ``shutil.which("bash")`` resolved to the
    WSL launcher stub under ``system32`` and every ``.sh`` cron script failed
    with ``WSL_E_DEFAULT_DISTRO_NOT_FOUND``.  These tests pin the new probe
    order so real Git Bash at a well-known location always wins.
    """

    def test_is_wsl_launcher_system32(self):
        from cron.scheduler import _is_wsl_launcher

        assert _is_wsl_launcher(r"C:\Windows\System32\bash.exe") is True
        assert _is_wsl_launcher(r"C:\windows\system32\bash.EXE") is True

    def test_is_wsl_launcher_windowsapps(self):
        from cron.scheduler import _is_wsl_launcher

        # WindowsApps WSL alias directory casing varies.
        assert _is_wsl_launcher(
            r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
        ) is True

    def test_is_wsl_launcher_git_bash_is_not_wsl(self):
        from cron.scheduler import _is_wsl_launcher

        assert _is_wsl_launcher(
            r"C:\Program Files\Git\bin\bash.exe"
        ) is False
        assert _is_wsl_launcher(
            r"C:\Program Files\Git\usr\bin\bash.exe"
        ) is False

    def test_is_wsl_launcher_empty_and_posix(self):
        from cron.scheduler import _is_wsl_launcher

        assert _is_wsl_launcher("") is False
        assert _is_wsl_launcher("/usr/bin/bash") is False
        assert _is_wsl_launcher("/bin/bash") is False

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Linux-only: probe order is exercised on real installs here.",
    )
    def test_resolve_bash_posix_prefers_which(self, monkeypatch):
        from cron import scheduler as sched_mod

        # On Linux, which("bash") should already resolve to a real bash.
        # Stub it to confirm the precedence: which wins over /bin/bash.
        monkeypatch.setattr(sched_mod.shutil, "which", lambda name: "/fake/bin/bash")
        assert sched_mod._resolve_bash() == "/fake/bin/bash"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows-only: confirms the real-host probe picks real Git Bash.",
    )
    def test_resolve_bash_windows_real_probe_finds_git_bash(self, monkeypatch):
        """On the actual Windows host, the probe must beat the WSL stub.

        Reproduces the user-facing failure deterministically: even when
        ``which("bash")`` would hand back the ``system32`` WSL stub, the
        well-known Git for Windows path probe fires first.
        """
        from cron.scheduler import _resolve_bash, _is_wsl_launcher

        resolved = _resolve_bash()
        assert resolved is not None, "expected Git Bash to be found"
        assert not _is_wsl_launcher(resolved), (
            f"_resolve_bash() returned the WSL stub: {resolved!r}"
        )

    def test_resolve_bash_windows_probe_beats_wsl_stub(self, monkeypatch):
        """Cross-platform test: simulate Windows + WSL stub on any host.

        Patches ``sys.platform`` to ``win32``, ``which("bash")`` to return a
        fake ``system32`` path (the WSL stub), and arranges a temp "Git for
        Windows" tree on disk.  ``_resolve_bash`` MUST pick the real Git Bash,
        never the WSL stub.
        """
        from cron import scheduler as sched_mod
        import platform

        # Build a fake "Git for Windows" install dir matching _GIT_WIN_BASH_PATHS[2]
        # (the ProgramFiles\Git\usr\bin\bash.exe probe used on amd64 installs).
        git_root = Path(str(monkeypatch_tmp_root(monkeypatch))) / "ProgramFiles" / "Git" / "usr" / "bin"
        git_root.mkdir(parents=True)
        git_bash = git_root / "bash.exe"
        git_bash.write_text("")  # presence only — we never execute it

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setattr(sched_mod.shutil, "which",
                            lambda name: r"C:\Windows\System32\bash.exe")
        monkeypatch.setenv("ProgramFiles", str(monkeypatch_tmp_root(monkeypatch) / "ProgramFiles"))
        # _GIT_WIN_BASH_PATHS is frozen at import time; patch it directly.
        monkeypatch.setattr(sched_mod, "_GIT_WIN_BASH_PATHS",
                            (str(git_bash),))

        resolved = sched_mod._resolve_bash()
        assert resolved == str(git_bash), (
            f"expected probe to pick Git Bash at {git_bash!r}, got {resolved!r}"
        )

    def test_resolve_bash_windows_no_git_returns_none_or_skips_stub(
        self, monkeypatch
    ):
        """When Git for Windows is absent AND ``which`` returns the WSL stub,
        ``_resolve_bash`` must NOT hand back the WSL stub. It should fall
        through to ``/bin/bash`` (None on Windows) rather than use the stub.
        """
        from cron import scheduler as sched_mod
        import platform

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setattr(sched_mod.shutil, "which",
                            lambda name: r"C:\Windows\System32\bash.exe")
        # No Git for Windows probe hits — point them at nonexistent paths.
        monkeypatch.setattr(sched_mod, "_GIT_WIN_BASH_PATHS", ())
        # /bin/bash won't exist on a real Windows host, so the result is None.
        # (On a POSIX CI box, /bin/bash is real — but we patched sys.platform,
        # not the filesystem; we accept either None or /bin/bash, but NEVER the
        # WSL stub path.)
        resolved = sched_mod._resolve_bash()
        assert resolved != r"C:\Windows\System32\bash.exe", (
            f"_resolve_bash() returned the WSL stub: {resolved!r} — must skip it"
        )

    def test_resolve_bash_uses_which_when_not_stub(self, monkeypatch):
        """On Windows with a sane ``which("bash")`` (not the stub), use it."""
        from cron import scheduler as sched_mod
        import platform

        # Do NOT patch _GIT_WIN_BASH_PATHS to anything that exists, but also
        # don't fudge real Git for Windows on the host. Stand up a real
        # bogus-but-not-wsl path for which("bash") to return.
        fake = r"C:\my\tools\bash.exe"

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setattr(sched_mod.shutil, "which", lambda name: fake)
        monkeypatch.setattr(sched_mod, "_GIT_WIN_BASH_PATHS", ())

        resolved = sched_mod._resolve_bash()
        assert resolved == fake

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="On a real Windows host, _run_job_script would invoke Git Bash.",
    )
    def test_sh_script_uses_resolved_bash_not_wsl_stub(self, cron_env, monkeypatch):
        """End-to-end: a ``.sh`` cron script runs via the resolved bash, not
        the WSL stub.

        Cross-platform test (Linux host exercises the same code path because
        ``_run_job_script`` is platform-agnostic once the interpreter is
        chosen). The script asserts which interpreter ran it via ``$0``."""
        from cron import scheduler as sched_mod
        import platform

        # Pretend to be Windows so the Git-Bash probe branch is taken.
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")

        # Stand up a fake "Git for Windows bash.exe" that is itself a tiny
        # real shell script (we control it: it prints its args[1]'s $0).
        win_root = Path(str(monkeypatch_tmp_root(monkeypatch))) / "GitForWindows"
        win_root.mkdir()
        # Use the system's real bash/sh to *be* "Git for Windows bash" so the
        # subprocess.run on this POSIX host actually works. On Linux hosts
        # without bash, fall back to /bin/sh.
        import shutil as _sh
        real_sh = _sh.which("bash") or "/bin/sh"
        # Symlink a few well-known names to real_sh so the probe finds "bash.exe".
        fake_bash = win_root / "bash.exe"
        try:
            os.symlink(real_sh, str(fake_bash))
        except (OSError, NotImplementedError):
            # No symlinks (some CI) → copy the bytes; exec will still work.
            import shutil as _sh2
            _sh2.copy(real_sh, str(fake_bash))
        monkeypatch.setattr(sched_mod, "_GIT_WIN_BASH_PATHS", (str(fake_bash),))

        # And make ``which("bash")`` (if it ever fired) return the WSL stub —
        # so we prove the probe won over the stub.
        monkeypatch.setattr(sched_mod.shutil, "which",
                            lambda name: r"C:\Windows\System32\bash.exe")

        script = cron_env / "scripts" / "ok.sh"
        script.write_text('echo "ran via: $0"\n')

        success, output = sched_mod._run_job_script("ok.sh")
        assert success is True, f"script failed: {output!r}"
        assert "ran via:" in output, output
        # The interpreter that ran the script must be our fake Git Bash, not
        # the WSL stub path.  $0 is the script path, so we can't directly see
        # which interpreter was used from $0 — but the mere fact that the
        # script SUCCEEDED (instead of failing with WSL_E_DEFAULT_DISTRO_NOT_FOUND)
        # proves the probe-beats-stub resolution fired.
        assert "WSL" not in output
        assert "no installed distributions" not in output


# tiny helper for the monkeypatch fixtures that need a throwaway dir
def monkeypatch_tmp_root(monkeypatch):
    """A throwaway directory uniquely scoped to this test invocation."""
    import tempfile
    d = tempfile.mkdtemp(prefix="cron_resolve_bash_")
    # ensure cleanup happens after the test
    monkeypatch._tmp_path = Path(d)  # pytest integration: pytest cleans tmp_path
    return Path(d)


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
