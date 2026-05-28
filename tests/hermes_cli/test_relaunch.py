"""Tests for hermes_cli.relaunch — unified self-relaunch utility."""

import sys

import pytest

from hermes_cli import relaunch as relaunch_mod


class TestResolveHermesBin:
    def test_prefers_absolute_argv0_when_executable(self, monkeypatch):
        fake = "/nix/store/abc/bin/hermes"
        monkeypatch.setattr(sys, "argv", [fake])
        monkeypatch.setattr(relaunch_mod.os.path, "isfile", lambda p: p == fake)
        monkeypatch.setattr(relaunch_mod.os, "access", lambda p, mode: p == fake)
        assert relaunch_mod.resolve_hermes_bin() == fake

    def test_resolves_relative_argv0(self, monkeypatch, tmp_path):
        fake = tmp_path / "hermes"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setattr(sys, "argv", [str(fake.name)])
        monkeypatch.chdir(tmp_path)
        # Ensure we don't accidentally match a real 'hermes' on PATH
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda _name: None)
        assert relaunch_mod.resolve_hermes_bin() == str(fake)

    def test_falls_back_to_path_which(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["-c"])  # not a real path
        monkeypatch.setattr(
            relaunch_mod.shutil, "which", lambda name: "/usr/bin/hermes" if name == "hermes" else None
        )
        assert relaunch_mod.resolve_hermes_bin() == "/usr/bin/hermes"

    def test_returns_none_when_unresolvable(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["-c"])
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda _name: None)
        assert relaunch_mod.resolve_hermes_bin() is None


class TestExtractInheritedFlags:
    def test_extracts_tui_and_dev(self):
        argv = ["--tui", "--dev", "chat"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--tui", "--dev"]

    def test_extracts_profile_with_value(self):
        argv = ["--profile", "work", "chat"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--profile", "work"]

    def test_extracts_short_p_with_value(self):
        argv = ["-p", "work"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["-p", "work"]

    def test_extracts_equals_form(self):
        argv = ["--profile=work", "--model=anthropic/claude-sonnet-4"]
        assert relaunch_mod._extract_inherited_flags(argv) == [
            "--profile=work",
            "--model=anthropic/claude-sonnet-4",
        ]

    def test_skips_unknown_flags(self):
        argv = ["--foo", "bar", "--tui"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--tui"]

    def test_does_not_consume_flag_like_value(self):
        argv = ["--tui", "--resume", "abc123"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--tui"]

    def test_preserves_multiple_skills(self):
        argv = ["-s", "foo", "-s", "bar", "--tui"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["-s", "foo", "-s", "bar", "--tui"]


class TestInheritedFlagTable:
    """Sanity-check the argparse-introspected table that drives extraction."""

    def test_short_and_long_aliases_are_paired(self):
        table = dict(relaunch_mod._INHERITED_FLAGS_TABLE)
        # Each pair declared together in the parser shares takes_value.
        for short, long_ in [
            ("-p", "--profile"),
            ("-m", "--model"),
            ("-s", "--skills"),
        ]:
            assert table[short] == table[long_], f"{short}/{long_} disagree"

    def test_store_true_flags_do_not_take_value(self):
        table = dict(relaunch_mod._INHERITED_FLAGS_TABLE)
        for flag in ["--tui", "--dev", "--yolo", "--ignore-user-config", "--ignore-rules"]:
            assert table[flag] is False, f"{flag} should not take a value"

    def test_value_flags_take_value(self):
        table = dict(relaunch_mod._INHERITED_FLAGS_TABLE)
        for flag in ["--profile", "--model", "--provider", "--skills"]:
            assert table[flag] is True, f"{flag} should take a value"

    def test_excluded_flags_are_not_inherited(self):
        table = dict(relaunch_mod._INHERITED_FLAGS_TABLE)
        # --worktree creates a new worktree per process; inheriting would
        # orphan the parent's. Chat-only flags (--quiet/-Q, --verbose/-v,
        # --source) can't be in argv at the existing relaunch callsites.
        for flag in ["-w", "--worktree", "-Q", "--quiet", "-v", "--verbose", "--source"]:
            assert flag not in table, f"{flag} should not be inherited"


class TestBuildRelaunchArgv:
    def test_uses_bin_when_available(self, monkeypatch):
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: "/usr/bin/hermes")
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"])
        assert argv[0] == "/usr/bin/hermes"

    def test_falls_back_to_python_module(self, monkeypatch):
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: None)
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"])
        assert argv == [sys.executable, "-m", "hermes_cli.main", "--resume", "abc"]

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

    def test_windows_uses_subprocess_not_execvp(self, monkeypatch):
        """On Windows, os.execvp raises OSError "Exec format error" when the
        target is a .cmd shim or console-script wrapper (both common for
        hermes).  relaunch() must detect win32 and use subprocess.run +
        sys.exit instead."""
        monkeypatch.setattr(relaunch_mod.sys, "platform", "win32")
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: r"C:\Users\test\hermes.exe")

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

    def test_windows_propagates_child_exit_code(self, monkeypatch):
        """A non-zero exit from the child should flow through to sys.exit."""
        monkeypatch.setattr(relaunch_mod.sys, "platform", "win32")
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

    def test_windows_surfaces_oserror_with_help(self, monkeypatch, capsys):
        """When subprocess itself raises OSError (file-not-found / bad format),
        we must NOT let it bubble up as a cryptic traceback — print a
        user-readable hint and sys.exit(1)."""
        monkeypatch.setattr(relaunch_mod.sys, "platform", "win32")
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: r"C:\missing.exe")

        import subprocess as _subprocess

        def fake_run(argv, **kwargs):
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(_subprocess, "run", fake_run)
        monkeypatch.setattr(relaunch_mod.os, "execvp", lambda *a, **kw: None)

        with pytest.raises(SystemExit) as exc_info:
            relaunch_mod.relaunch(["chat"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "relaunch failed" in err
        assert "open a new terminal" in err.lower() or "path" in err.lower()


class TestResolveHermesBinWindowsPyGuard:
    """On Windows, resolve_hermes_bin MUST NOT return a .py path.
    os.access(x, os.X_OK) returns True for .py files on Windows because
    PATHEXT includes .py when the Python launcher is installed — but
    subprocess.run can't actually exec a .py directly, so the relaunch
    would fail with the cryptic "%1 is not a valid Win32 application" error.
    """

    def test_windows_rejects_py_argv0_falls_through_to_path(self, monkeypatch, tmp_path):
        """On Windows, if sys.argv[0] is a .py file, we must skip the
        argv[0] fast-path and fall through to PATH / python -m."""
        # Build a fake .py script that "passes" the isfile + X_OK checks.
        script = tmp_path / "main.py"
        script.write_text("# stub")

        monkeypatch.setattr(relaunch_mod.sys, "platform", "win32")
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        # Force PATH lookup to return a hermes.exe so the test doesn't
        # exercise the None-fallback path (that's a separate test).
        monkeypatch.setattr(
            relaunch_mod.shutil, "which",
            lambda name: r"C:\venv\Scripts\hermes.exe" if name == "hermes" else None,
        )

        bin_path = relaunch_mod.resolve_hermes_bin()
        # Must NOT be the .py — must be the hermes.exe PATH entry.
        assert bin_path == r"C:\venv\Scripts\hermes.exe"

    def test_posix_still_accepts_py_argv0(self, monkeypatch, tmp_path):
        """POSIX behaviour unchanged: argv[0] pointing at an executable
        script (including .py with a shebang + chmod +x) is fine to return
        because POSIX exec can route through the shebang line."""
        if sys.platform == "win32":
            pytest.skip("POSIX semantics")
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        assert relaunch_mod.resolve_hermes_bin() == str(script)

    def test_windows_py_argv0_with_no_hermes_on_path_returns_none(self, monkeypatch, tmp_path):
        """Bulletproof fallback: if argv0 is .py on Windows AND hermes.exe
        isn't on PATH, return None so the caller falls back to
        python -m hermes_cli.main."""
        script = tmp_path / "main.py"
        script.write_text("# stub")

        monkeypatch.setattr(relaunch_mod.sys, "platform", "win32")
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda name: None)

        assert relaunch_mod.resolve_hermes_bin() is None
