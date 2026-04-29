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