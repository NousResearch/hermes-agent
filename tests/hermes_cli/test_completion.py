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

    def test_nested_subcommands_extracted(self):
        tree = _walk(_make_parser())
        gw_subs = set(tree["subcommands"]["gateway"]["subcommands"].keys())
        assert {"start", "stop", "status", "run"}.issubset(gw_subs)

    def test_aliases_not_duplicated(self):
        """'foreground' is an alias of 'run' — must not appear as separate entry."""
        tree = _walk(_make_parser())
        gw_subs = tree["subcommands"]["gateway"]["subcommands"]
        assert "foreground" not in gw_subs

    def test_flags_extracted(self):
        tree = _walk(_make_parser())
        chat_flags = tree["subcommands"]["chat"]["flags"]
        assert "-q" in chat_flags or "--query" in chat_flags

    def test_help_text_captured(self):
        tree = _walk(_make_parser())
        assert tree["subcommands"]["chat"]["help"] != ""
        assert tree["subcommands"]["gateway"]["help"] != ""


# ---------------------------------------------------------------------------
# 2. Bash output
# ---------------------------------------------------------------------------

class TestGenerateBash:
    def test_contains_completion_function_and_register(self):
        out = generate_bash(_make_parser())
        assert "_hermes_completion()" in out
        assert "complete -F _hermes_completion hermes" in out

    def test_top_level_commands_present(self):
        out = generate_bash(_make_parser())
        for cmd in ("chat", "gateway", "sessions", "version"):
            assert cmd in out

    def test_nested_subcommands_in_case(self):
        out = generate_bash(_make_parser())
        assert "start" in out
        assert "stop" in out

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
    def test_contains_compdef_header(self):
        out = generate_zsh(_make_parser())
        assert "#compdef hermes" in out

    def test_top_level_commands_present(self):
        out = generate_zsh(_make_parser())
        for cmd in ("chat", "gateway", "sessions", "version"):
            assert cmd in out

    def test_nested_describe_blocks(self):
        out = generate_zsh(_make_parser())
        assert "_describe" in out
        # gateway has subcommands so a _cmds array must be generated
        assert "gateway_cmds" in out

    def test_registers_compdef_instead_of_invoking_completion_function(self):
        out = generate_zsh(_make_parser())
        assert 'compdef _hermes hermes' in out
        assert '_hermes "$@"' not in out

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

    def test_zsh_eval_style_source_registers_after_compinit(self):
        if not shutil.which("zsh"):
            pytest.skip("zsh not installed")
        out = generate_zsh(_make_parser())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh", delete=False) as f:
            f.write(out)
            path = f.name
        try:
            result = subprocess.run(
                [
                    "zsh",
                    "-fc",
                    f"autoload -Uz compinit && compinit -D; source {path}; [[ ${{_comps[hermes]}} == _hermes ]]",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert result.stderr == ""
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Fish output
# ---------------------------------------------------------------------------

class TestGenerateFish:
    def test_disables_file_completion(self):
        out = generate_fish(_make_parser())
        assert "complete -c hermes -f" in out

    def test_top_level_commands_present(self):
        out = generate_fish(_make_parser())
        for cmd in ("chat", "gateway", "sessions", "version"):
            assert cmd in out

    def test_subcommand_guard_present(self):
        out = generate_fish(_make_parser())
        assert "__fish_seen_subcommand_from" in out

    def test_valid_fish_syntax(self):
        """Script must be accepted by fish without errors."""
        if not shutil.which("fish"):
            pytest.skip("fish not installed")
        out = generate_fish(_make_parser())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fish", delete=False) as f:
            f.write(out)
            path = f.name
        try:
            result = subprocess.run(["fish", path], capture_output=True)
            assert result.returncode == 0, result.stderr.decode()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Subcommand drift prevention
# ---------------------------------------------------------------------------

class TestSubcommandDrift:
    def test_SUBCOMMANDS_covers_required_commands(self):
        """_SUBCOMMANDS must include all known top-level commands so that
        multi-word session names after -c/-r are never accidentally split.
        """
        import inspect
        from hermes_cli.main import _coalesce_session_name_args

        source = inspect.getsource(_coalesce_session_name_args)
        match = re.search(r'_SUBCOMMANDS\s*=\s*\{([^}]+)\}', source, re.DOTALL)
        assert match, "_SUBCOMMANDS block not found in _coalesce_session_name_args()"
        defined = set(re.findall(r'"(\w+)"', match.group(1)))

        required = {
            "chat", "model", "gateway", "setup", "login", "logout", "auth",
            "status", "cron", "config", "sessions", "version", "update",
            "uninstall", "profile", "skills", "tools", "mcp", "plugins",
            "acp", "claw", "honcho", "completion", "logs",
        }
        missing = required - defined
        assert not missing, f"Missing from _SUBCOMMANDS: {missing}"


# ---------------------------------------------------------------------------
# 6. Profile completion (regression prevention)
# ---------------------------------------------------------------------------

class TestProfileCompletion:
    """Ensure profile name completion is present in all shell outputs."""

    def test_bash_has_profiles_helper(self):
        out = generate_bash(_make_parser())
        assert "_hermes_profiles()" in out
        assert 'profiles_dir="$HOME/.hermes/profiles"' in out

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

    def test_zsh_has_profiles_helper(self):
        out = generate_zsh(_make_parser())
        assert "_hermes_profiles()" in out
        assert "$HOME/.hermes/profiles" in out

    def test_zsh_has_profile_flag_completion(self):
        out = generate_zsh(_make_parser())
        assert "--profile" in out
        assert "_hermes_profiles" in out

    def test_zsh_profile_actions_complete_names(self):
        out = generate_zsh(_make_parser())
        assert "use|delete|show|alias|rename|export)" in out

    def test_fish_has_profiles_helper(self):
        out = generate_fish(_make_parser())
        assert "__hermes_profiles" in out
        assert "$HOME/.hermes/profiles" in out

    def test_fish_has_profile_flag_completion(self):
        out = generate_fish(_make_parser())
        assert "-s p -l profile" in out
        assert "(__hermes_profiles)" in out

    def test_fish_profile_actions_complete_names(self):
        out = generate_fish(_make_parser())
        # Should have profile name completion for actions like use, delete, etc.
        assert "__hermes_profiles" in out
        count = out.count("(__hermes_profiles)")
        # At least the -p flag + the profile action completions
        assert count >= 2, f"Expected >=2 profile completion entries, got {count}"
