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


class TestExtractInheritedFlags:
    def test_extracts_tui_and_dev(self):
        argv = ["--tui", "--dev", "chat"]
        assert relaunch_mod._extract_inherited_flags(argv) == ["--tui", "--dev"]


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


class TestBuildRelaunchArgvEnvShebangInterpreter:
    """Regression: an env-shebang script (``#!/usr/bin/env python3``, e.g. the
    source-tree ``hermes`` script on a git install) must be exec'd under the
    CURRENT interpreter, never via its shebang.

    The kernel re-resolves ``env python3`` from PATH at exec time. On a git
    install the resolved bin is the repo ``hermes`` script whose shebang
    lands on a system python that lacks hermes' deps -> ModuleNotFoundError
    (e.g. ``prompt_toolkit``) on any relaunch path: ``sessions browse``
    resume, /update, /refine.
    """

    def _env_shebang_script(self, tmp_path):
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        return script

    def test_env_shebang_script_runs_under_current_interpreter(self, monkeypatch, tmp_path):
        script = self._env_shebang_script(tmp_path)
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: str(script))
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"], preserve_inherited=False)
        # Must NOT be [script, ...] — that execs via the env shebang.
        assert argv[0] == sys.executable
        assert argv[1] == str(script)
        assert argv[2:] == ["--resume", "abc"]

    def test_pinned_shebang_script_still_execs_directly(self, monkeypatch, tmp_path):
        script = tmp_path / "hermes"
        script.write_text("#!/opt/venv/bin/python\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: str(script))
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"], preserve_inherited=False)
        # A pinned interpreter is trusted — direct exec preserved.
        assert argv == [str(script), "--resume", "abc"]

    def test_real_binary_still_execs_directly(self, monkeypatch, tmp_path):
        binary = tmp_path / "hermes"
        binary.write_bytes(b"\x7fELF not really, but no shebang")
        binary.chmod(0o755)
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: str(binary))
        argv = relaunch_mod.build_relaunch_argv(["--resume", "abc"], preserve_inherited=False)
        assert argv == [str(binary), "--resume", "abc"]

    @pytest.mark.linux_only
    def test_e2e_env_shebang_relaunch_runs_under_current_interpreter(self, monkeypatch, tmp_path):
        """E2E: actually exec the built argv and verify the child interpreter
        is sys.executable, not whatever ``env python3`` resolves to."""
        script = tmp_path / "hermes"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "print(sys.executable)\n"
        )
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: str(script))
        argv = relaunch_mod.build_relaunch_argv([])

        import subprocess
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == sys.executable


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

    @pytest.mark.windows_only
    def test_windows_uses_subprocess_not_execvp(self, monkeypatch):
        """On Windows, os.execvp raises OSError "Exec format error" when the
        target is a .cmd shim or console-script wrapper (both common for
        hermes).  relaunch() must detect win32 and use subprocess.run +
        sys.exit instead.

        ``windows_only``: the bug is that ``os.execvp`` cannot exec a Windows
        console-script shim. On Linux ``execvp`` works fine, so a patched
        platform only re-asserted the branch we wrote, never the constraint
        that motivated it.
        """
        monkeypatch.setattr(relaunch_mod, "resolve_hermes_bin", lambda: r"C:\Users\test\hermes.exe")
        # Pin sys.argv: relaunch() preserves inherited flags from the LIVE
        # argv, so under pytest it happily inherited the runner's own
        # "-m 'windows_only and not integration'" and the assertion below saw
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

    @pytest.mark.windows_only
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

    The Windows cases are ``windows_only``: the PATHEXT-driven ``os.access``
    result the guard defends against simply does not occur on POSIX, so a
    faked ``sys.platform`` could never reproduce the hazard.
    """

    @pytest.mark.windows_only
    def test_windows_rejects_py_argv0_falls_through_to_path(self, monkeypatch, tmp_path):
        """On Windows, if sys.argv[0] is a .py file, we must skip the
        argv[0] fast-path and fall through to PATH / python -m."""
        # Build a fake .py script that "passes" the isfile + X_OK checks.
        script = tmp_path / "main.py"
        script.write_text("# stub")

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

    @pytest.mark.linux_only
    def test_posix_env_shebang_argv0_prefers_path_launcher(self, monkeypatch, tmp_path):
        """POSIX: an argv[0] with an env shebang (#!/usr/bin/env python3) is
        demoted — its interpreter is re-resolved from PATH at exec time and
        may lack hermes' deps. Prefer the PATH launcher when one exists."""
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setattr(
            relaunch_mod.shutil, "which",
            lambda name: "/usr/bin/hermes" if name == "hermes" else None,
        )
        assert relaunch_mod.resolve_hermes_bin() == "/usr/bin/hermes"

    @pytest.mark.linux_only
    def test_posix_env_shebang_argv0_no_path_returns_none(self, monkeypatch, tmp_path):
        """With no PATH launcher, an env-shebang argv0 yields None so the
        caller falls back to python -m hermes_cli.main (current interpreter)."""
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda name: None)
        assert relaunch_mod.resolve_hermes_bin() is None

    @pytest.mark.linux_only
    def test_posix_accepts_pinned_shebang_argv0(self, monkeypatch, tmp_path):
        """A script with a pinned interpreter (#!/opt/venv/bin/python) is
        safe to return: the kernel execs that exact interpreter, no PATH
        re-resolution."""
        script = tmp_path / "hermes"
        script.write_text("#!/opt/venv/bin/python\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda name: None)
        assert relaunch_mod.resolve_hermes_bin() == str(script)

    @pytest.mark.linux_only
    def test_posix_rejects_env_shebang_path_lookup(self, monkeypatch, tmp_path):
        """Defense-in-depth: a PATH-resolved ``hermes`` whose shebang is
        env-resolved python must be rejected just like an argv0 candidate —
        not returned, and not demoted to nothing (falls through to the
        python -m fallback)."""
        script = tmp_path / "hermes"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", ["-c"])  # not a real path
        monkeypatch.setattr(
            relaunch_mod.shutil, "which",
            lambda name: str(script) if name == "hermes" else None,
        )
        assert relaunch_mod.resolve_hermes_bin() is None

    @pytest.mark.linux_only
    def test_posix_accepts_pinned_shebang_path_lookup(self, monkeypatch, tmp_path):
        """A PATH-resolved ``hermes`` with a pinned interpreter is trusted —
        the guard must not overblock non-env shebangs."""
        script = tmp_path / "hermes"
        script.write_text("#!/opt/venv/bin/python\n")
        script.chmod(0o755)
        monkeypatch.setattr(relaunch_mod.sys, "argv", ["-c"])  # not a real path
        monkeypatch.setattr(
            relaunch_mod.shutil, "which",
            lambda name: str(script) if name == "hermes" else None,
        )
        assert relaunch_mod.resolve_hermes_bin() == str(script)

    @pytest.mark.windows_only
    def test_windows_py_argv0_with_no_hermes_on_path_returns_none(self, monkeypatch, tmp_path):
        """Bulletproof fallback: if argv0 is .py on Windows AND hermes.exe
        isn't on PATH, return None so the caller falls back to
        python -m hermes_cli.main."""
        script = tmp_path / "main.py"
        script.write_text("# stub")

        monkeypatch.setattr(relaunch_mod.sys, "argv", [str(script), "chat"])
        monkeypatch.setattr(relaunch_mod.shutil, "which", lambda name: None)

        assert relaunch_mod.resolve_hermes_bin() is None
