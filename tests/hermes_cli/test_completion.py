"""Tests for hermes_cli/completion.py — shell completion script generation."""

import argparse
import os
import re
import shutil
import subprocess
import tempfile

import pytest

from hermes_cli.completion import _walk, generate_bash, generate_zsh, generate_fish


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    """Build a minimal parser that mirrors the real hermes structure."""
    p = argparse.ArgumentParser(prog="hermes")
    p.add_argument("--version", "-V", action="store_true")
    p.add_argument("-p", "--profile", help="Profile name")
    sub = p.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Interactive chat with the agent")
    chat.add_argument("-q", "--query")
    chat.add_argument("-m", "--model")

    gw = sub.add_parser("gateway", help="Messaging gateway management")
    gw_sub = gw.add_subparsers(dest="gateway_command")
    gw_sub.add_parser("start", help="Start service")
    gw_sub.add_parser("stop", help="Stop service")
    gw_sub.add_parser("status", help="Show status")
    # alias — should NOT appear as a duplicate in completions
    gw_sub.add_parser("run", aliases=["foreground"], help="Run in foreground")

    sess = sub.add_parser("sessions", help="Manage session history")
    sess_sub = sess.add_subparsers(dest="sessions_action")
    sess_sub.add_parser("list", help="List sessions")
    sess_sub.add_parser("delete", help="Delete a session")

    prof = sub.add_parser("profile", help="Manage profiles")
    prof_sub = prof.add_subparsers(dest="profile_command")
    prof_sub.add_parser("list", help="List profiles")
    prof_sub.add_parser("use", help="Switch to a profile")
    prof_sub.add_parser("create", help="Create a new profile")
    prof_sub.add_parser("delete", help="Delete a profile")
    prof_sub.add_parser("show", help="Show profile details")
    prof_sub.add_parser("alias", help="Set profile alias")
    prof_sub.add_parser("rename", help="Rename a profile")
    prof_sub.add_parser("export", help="Export a profile")

    sub.add_parser("version", help="Show version")

    return p


# ---------------------------------------------------------------------------
# 1. Parser extraction
# ---------------------------------------------------------------------------

class TestWalk:
    def test_top_level_subcommands_extracted(self):
        tree = _walk(_make_parser())
        assert set(tree["subcommands"].keys()) == {"chat", "gateway", "sessions", "profile", "version"}


    def test_aliases_not_duplicated(self):
        """'foreground' is an alias of 'run' — must not appear as separate entry."""
        tree = _walk(_make_parser())
        gw_subs = tree["subcommands"]["gateway"]["subcommands"]
        assert "foreground" not in gw_subs


# ---------------------------------------------------------------------------
# 2. Bash output
# ---------------------------------------------------------------------------

class TestGenerateBash:
    def test_contains_completion_function_and_register(self):
        out = generate_bash(_make_parser())
        assert "_hermes_completion()" in out
        assert "complete -F _hermes_completion hermes" in out


    def test_valid_bash_syntax(self):
        """Script must pass `bash -n` syntax check."""
        out = generate_bash(_make_parser())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bash", delete=False) as f:
            f.write(out)
            path = f.name
        try:
            result = subprocess.run(["bash", "-n", path], capture_output=True)
            assert result.returncode == 0, result.stderr.decode()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Zsh output
# ---------------------------------------------------------------------------

class TestGenerateZsh:


    def test_preserves_valid_zsh_arguments_alias_syntax(self):
        out = generate_zsh(_make_parser())
        assert "'(-)'{-h,--help}'[Show help and exit]'" in out
        assert "'(-)'{-V,--version}'[Show version and exit]'" in out
        assert "'(-)'{-p,--profile}'[Profile name]:profile:_hermes_profiles'" in out
        assert "'(-h --help){-h,--help}[Show help and exit]'" not in out
        assert '"(-h --help)"{-h,--help}"[Show help and exit]"' not in out

    def test_valid_zsh_syntax(self):
        if not shutil.which("zsh"):
            pytest.skip("zsh not installed")
        out = generate_zsh(_make_parser())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh", delete=False) as f:
            f.write(out)
            path = f.name
        try:
            result = subprocess.run(["zsh", "-n", path], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Fish output
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Subcommand drift prevention
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Profile completion (regression prevention)
# ---------------------------------------------------------------------------

class TestProfileCompletion:
    """Ensure profile name completion is present in all shell outputs."""


    def test_bash_completes_profiles_after_p_flag(self):
        out = generate_bash(_make_parser())
        assert '"-p"' in out or "== \"-p\"" in out
        assert '"--profile"' in out or '== "--profile"' in out
        assert "_hermes_profiles" in out

    def test_bash_profile_subcommand_has_action_completion(self):
        out = generate_bash(_make_parser())
        assert "use|delete|show|alias|rename|export)" in out

    def test_bash_profile_actions_complete_profile_names(self):
        """After 'hermes profile use', complete with profile names."""
        out = generate_bash(_make_parser())
        # The profile case should have _hermes_profiles for name-taking actions
        lines = out.split("\n")
        in_profile_case = False
        has_profiles_in_action = False
        for line in lines:
            if "profile)" in line:
                in_profile_case = True
            if in_profile_case and "_hermes_profiles" in line:
                has_profiles_in_action = True
                break
        assert has_profiles_in_action, "profile actions should complete with _hermes_profiles"


    def test_fish_profile_actions_complete_names(self):
        out = generate_fish(_make_parser())
        # Should have profile name completion for actions like use, delete, etc.
        assert "__hermes_profiles" in out
        count = out.count("(__hermes_profiles)")
        # At least the -p flag + the profile action completions
        assert count >= 2, f"Expected >=2 profile completion entries, got {count}"
