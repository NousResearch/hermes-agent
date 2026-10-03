"""Tests for hermes_cli.relaunch — unified self-relaunch utility."""

import os
import subprocess
import sys
import venv

import pytest

from hermes_cli import relaunch as relaunch_mod


class TestResolveHermesBin:
    def test_prefers_absolute_argv0_when_executable(self, monkeypatch):
        fake = "/nix/store/abc/bin/hermes"
        monkeypatch.setattr(sys, "argv", [fake])
        monkeypatch.setattr(relaunch_mod.os.path, "isfile", lambda p: p == fake)
        monkeypatch.setattr(relaunch_mod.os, "access", lambda p, mode: p == fake)
        assert relaunch_mod.resolve_hermes_bin() == fake

    def test_does_not_resolve_bare_argv0_from_workspace_cwd(self, monkeypatch, tmp_path):
        fake = tmp_path / "hermes"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setattr(sys, "argv", [fake.name])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert relaunch_mod.resolve_hermes_bin() is None

    def test_does_not_resolve_hermes_from_path(self, monkeypatch, tmp_path):
        fake = tmp_path / "hermes"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setattr(sys, "argv", ["-c"])
        monkeypatch.setenv("PATH", str(tmp_path))
        assert relaunch_mod.resolve_hermes_bin() is None


class TestExtractInheritedFlags:
    def test_extracts_tui_and_dev(self):
        argv = ["--tui", "--dev", "chat"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--tui", "--dev"]


    def test_preserves_multiple_skills(self):
        argv = ["-s", "foo", "-s", "bar", "--tui"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["-s", "foo", "-s", "bar", "--tui"]


class TestInheritedFlagTable:
    """Sanity-check the argparse-introspected table that drives extraction."""



    def test_excluded_flags_are_not_inherited(self):
        table = dict(relaunch_mod._INHERITED_FLAGS_TABLE)
        # --worktree creates a new worktree per process; inheriting would
        # orphan the parent's. Chat-only flags (--quiet/-Q, --verbose/-v,
        # --source) can't be in argv at the existing relaunch callsites.
        for flag in ["-w", "--worktree", "-Q", "--quiet", "-v", "--verbose", "--source"]:
            assert flag not in table, f"{flag} should not be inherited"


class TestBuildRelaunchArgv:


    def test_preserves_inherited_flags(self, monkeypatch):
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/bin/hermes")
        original = ["--tui", "--dev", "--profile", "work", "sessions", "browse"]
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"], original_argv=original)
        assert "--tui" in argv
        assert "--dev" in argv
        assert "--profile" in argv
        assert "work" in argv
        assert "--resume" in argv
        assert "abc" in argv
        # The original subcommand should not survive
        assert "sessions" not in argv
        assert "browse" not in argv

    def test_can_disable_preserve(self, monkeypatch):
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/bin/hermes")
        original = ["--tui", "chat"]
        argv = relaunch_mod.build_relaunch_argv(
            ["--resume", "abc"], preserve_inherited=False, original_argv=original
        )
        assert "--tui" not in argv
        assert argv == ["/usr/bin/hermes", "--resume", "abc"]

    def test_fallback_uses_current_installation_command(self, monkeypatch):
        from hermes_cli import _launchers

        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: None)
        monkeypatch.setattr(_launchers, "current_installation_command", lambda: ["/install/bin/hermes"])
        argv = relaunch_mod.build_relaunch_argv(["--version"], preserve_inherited=False)
        assert argv == ["/install/bin/hermes", "--version"]

    def test_isolated_import_failure_recovers_from_foreign_workspace_cwd(self, monkeypatch, tmp_path):
        """The old ``python -I -m hermes_cli.main`` fallback cannot see source installed outside
        site-packages. The replacement's install-bound bootstrap reaches the real CLI from a
        workspace CWD with PATH, PYTHONPATH, and HERMES_BIN deliberately unavailable.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        clean_env = {
            key: value for key, value in os.environ.items()
            if key not in {"HERMES_BIN", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }
        clean_env.update({"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"),
                          "HERMES_HOME": str(tmp_path / "hermes-home")})
        isolated = tmp_path / "isolated-python"
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(isolated)
        isolated_python = isolated / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        legacy = subprocess.run(
            [str(isolated_python), "-I", "-m", "hermes_cli.main", "--version"],
            cwd=workspace, env=clean_env, capture_output=True, text=True, timeout=30,
        )
        assert legacy.returncode != 0
        assert "No module named 'hermes_cli'" in legacy.stderr

        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: None)
        argv = relaunch_mod.build_relaunch_argv(["--version"], preserve_inherited=False)
        launched = subprocess.run(
            argv, cwd=workspace, env=clean_env, capture_output=True, text=True, timeout=90,
        )
        assert launched.returncode == 0, launched.stderr
        assert launched.stdout.strip(), launched.stderr


class TestRelaunch:
    def test_calls_execvp(self, monkeypatch):
        calls = []

        def fake_execvp(path, argv):
            calls.append((path, argv))
            raise SystemExit(0)

        monkeypatch.setattr(relaunch_mod.os, "execvp", fake_execvp)
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/bin/hermes")

        with pytest.raises(SystemExit):
            relaunch_mod.relaunch(["--resume", "abc"])

        assert calls == [("/usr/bin/hermes", ["/usr/bin/hermes", "--resume", "abc"])]

    @pytest.mark.platforms("windows")
    def test_windows_uses_subprocess_not_execvp(self, monkeypatch):
        """On Windows, os.execvp raises OSError "Exec format error" when the
        target is a .cmd shim or console-script wrapper (both common for
        hermes).  relaunch() must detect win32 and use subprocess.run +
        sys.exit instead.

        ``platforms("windows")``: the bug is that ``os.execvp`` cannot exec a Windows
        console-script shim. On Linux ``execvp`` works fine, so a patched
        platform only re-asserted the branch we wrote, never the constraint
        that motivated it.
        """
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: r"C:\Users\test\hermes.exe")
        # Pin sys.argv: relaunch() preserves inherited flags from the LIVE
        # argv, so under pytest it happily inherited the runner's own
        # "-m 'platforms and not integration'" and the assertion below saw
        # them in the child argv. Nothing to do with Windows — it only showed
        # up here because this is the first lane that actually executes the
        # test, and -m is how that lane selects it.
        monkeypatch.setattr(relaunch_mod.sys, "argv", [r"C:\Users\test\hermes.exe"])

        import subprocess as _subprocess

        captured_argv = []

        def fake_subprocess_run(argv, **kwargs):
            captured_argv.append(list(argv))
            class _Result:
                returncode = 0
            return _Result()

        monkeypatch.setattr(_subprocess, "run", fake_subprocess_run)

        # execvp MUST NOT be called on Windows — route must go through subprocess
        execvp_calls = []

        def fake_execvp(*args, **kwargs):
            execvp_calls.append(args)
            raise AssertionError("os.execvp must not be called on Windows")

        monkeypatch.setattr(relaunch_mod.os, "execvp", fake_execvp)

        with pytest.raises(SystemExit) as exc_info:
            relaunch_mod.relaunch(["chat"])

        assert exc_info.value.code == 0
        assert execvp_calls == []
        assert captured_argv == [[r"C:\Users\test\hermes.exe", "chat"]]

    @pytest.mark.platforms("windows")
    def test_windows_propagates_child_exit_code(self, monkeypatch):
        """A non-zero exit from the child should flow through to sys.exit."""
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: r"C:\hermes.exe")

        import subprocess as _subprocess

        def fake_run(argv, **kwargs):
            class _Result:
                returncode = 42
            return _Result()

        monkeypatch.setattr(_subprocess, "run", fake_run)
        monkeypatch.setattr(relaunch_mod.os, "execvp", lambda *a, **kw: None)

        with pytest.raises(SystemExit) as exc_info:
            relaunch_mod.relaunch(["chat"])
        assert exc_info.value.code == 42


class TestResolveHermesBinWindowsPyGuard:
    """On Windows, resolve_hermes_bin MUST NOT return a .py path.
    os.access(x, os.X_OK) returns True for .py files on Windows because
    PATHEXT includes .py when the Python launcher is installed — but
    subprocess.run can't actually exec a .py directly, so the relaunch
    would fail with the cryptic "%1 is not a valid Win32 application" error.

    The Windows cases are ``platforms("windows")``: the PATHEXT-driven ``os.access``
    result the guard defends against simply does not occur on POSIX, so a
    faked ``sys.platform`` could never reproduce the hazard.
    """

    @pytest.mark.platforms("windows")
    def test_windows_rejects_py_argv0_uses_installation_fallback(self, monkeypatch, tmp_path):
        """A .py argv0 is rejected; callers then use the install-bound command, never PATH."""
        script = tmp_path / "main.py"
        script.write_text("# stub")
        wrapper = tmp_path / "bin" / "hermes.exe"
        wrapper.parent.mkdir()
        wrapper.write_text("executable stub")
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setenv("PATH", str(wrapper.parent))
        assert relaunch_mod.resolve_hermes_bin() is None

    def test_posix_python_launcher_falls_through_to_installation_bootstrap(
        self, monkeypatch, tmp_path
    ):
        """A source launcher with a foreign shebang is not exec'd or looked up on PATH."""
        if sys.platform == "win32":
            pytest.skip("POSIX semantics")
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        wrapper = tmp_path / "bin" / "hermes"
        wrapper.parent.mkdir()
        wrapper.write_text("#!/usr/bin/env bash\n")
        wrapper.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setenv("PATH", str(wrapper.parent))
        assert relaunch_mod.resolve_hermes_bin() is None

        # A console script pinned to the running interpreter remains a safe absolute executable.
        pinned = tmp_path / "pinned" / "hermes"
        pinned.parent.mkdir()
        pinned.write_text(f"#!{sys.executable}\n")
        pinned.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(pinned), "chat"])
        assert relaunch_mod.resolve_hermes_bin() == str(pinned)

    @pytest.mark.platforms("windows")
    def test_windows_py_argv0_with_no_hermes_on_path_returns_none(self, monkeypatch, tmp_path):
        """If argv0 is .py, return None so relaunch uses the installation bootstrap."""
        script = tmp_path / "main.py"
        script.write_text("# stub")
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        assert relaunch_mod.resolve_hermes_bin() is None
