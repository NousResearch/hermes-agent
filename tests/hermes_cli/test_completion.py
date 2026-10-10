"""Tests for hermes_cli/completion.py — shell completion script generation."""

import argparse
import os
import subprocess
import tempfile

from hermes_cli.completion import _walk, generate_bash, generate_fish


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
# 3. Profile completion (regression prevention)
# ---------------------------------------------------------------------------

class TestProfileCompletion:
    """Ensure profile name completion is present in all shell outputs."""


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


# ---------------------------------------------------------------------------
# 4. Config key completion
# ---------------------------------------------------------------------------

def test_bash_completes_config_keys_and_boolean_values(monkeypatch):
    """`hermes config set <TAB>` offers the `config keys` inventory, then true/false for a bool key."""
    import hermes_cli.config_inventory as inventory

    monkeypatch.setattr(inventory, "registered_config_keys",
                        lambda: ["compression.enabled", "model.default", "ra.service\\.name"])
    monkeypatch.setattr(inventory, "boolean_config_keys", lambda: ["compression.enabled"])
    parser = _make_parser()
    cfg = parser._subparsers._group_actions[0].add_parser("config", help="Config")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    for name in ("get", "set", "unset", "show"):
        cfg_sub.add_parser(name, help=name)
    script = generate_bash(parser)

    def complete(*words):
        driver = (f'{script}\nCOMP_WORDS=(hermes {" ".join(repr(w) for w in words)}); '
                  'COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 )); _hermes_completion; echo "${COMPREPLY[*]}"')
        return subprocess.run(["bash", "-c", driver], capture_output=True, text=True,
                              encoding="utf-8", check=True).stdout.split()

    assert complete("config", "set", "comp") == ["compression.enabled"]
    assert complete("config", "get", "") == ["compression.enabled", "model.default"]
    assert complete("config", "set", "compression.enabled", "") == ["true", "false"]
    assert complete("config", "set", "model.default", "") == []
    assert complete("config", "show", "") == []
