"""Tests for hermes_cli.claude_operator — pure logic + safety contract.

Deliberately avoids tmux integration. The side-effect layer (spawn / stop /
list_sessions) shells out to ``tmux`` and ``claude``; covering it in unit
tests would require either a real tmux server or a brittle mock harness.
The pure pieces below are where every bug actually lands:

* slug normalization (used for tmux session naming + filesystem paths)
* the safety refusal for bypass / dontAsk permission modes
* the argv the worker is launched with (auto is the default; bypass and
  dontAsk must never appear)
* the tmux launch shell-string (worker output must be tee'd to the log)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import claude_operator as op


# ---------------------------------------------------------------------------
# Slug + naming
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercases_and_kebabs(self):
        assert op.slugify("Fix Stale AGM") == "fix-stale-agm"

    def test_strips_punctuation(self):
        assert op.slugify("PR #769: rename roster!") == "pr-769-rename-roster"

    def test_collapses_repeats(self):
        assert op.slugify("a   b___c") == "a-b-c"

    def test_truncates_to_max(self):
        result = op.slugify("a" * 100, max_len=10)
        assert len(result) <= 10
        assert result == "a" * 10

    def test_empty_raises(self):
        with pytest.raises(op.OperatorError):
            op.slugify("!!!")

    def test_none_raises(self):
        with pytest.raises(op.OperatorError):
            op.slugify(None)  # type: ignore[arg-type]


class TestSessionName:
    def test_shape_matches_plan(self):
        # operator-plan.md section 4 fixes the shape: hermes/<project>/<worker>/<intent>
        assert (
            op.session_name("lyra", "claude", "fix-stale-agm")
            == "hermes/lyra/claude/fix-stale-agm"
        )

    def test_normalizes_all_segments(self):
        assert (
            op.session_name("Lyra A&R", "Claude Code", "Fix Stale AGM")
            == "hermes/lyra-a-r/claude-code/fix-stale-agm"
        )

    def test_supervisor_namespace_is_reserved_only_by_convention(self):
        # We don't block hermes/_supervisor at the slug layer — supervisor names
        # would be slugified to "supervisor" anyway. This test pins the behavior
        # so a future maintainer doesn't add a hidden hard-block here without
        # also updating the convention in operator-plan.md section 4.
        assert op.session_name("supervisor", "x", "y") == "hermes/supervisor/x/y"


# ---------------------------------------------------------------------------
# Paths — must scope under HERMES_HOME so profiles isolate workers
# ---------------------------------------------------------------------------


class TestPaths:
    def test_paths_scope_to_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        name = "hermes/lyra/claude/fix-stale-agm"
        log = op.log_path(name)
        prompt = op.prompt_path(name)
        assert log.is_relative_to(tmp_path)
        assert prompt.is_relative_to(tmp_path)

    def test_paths_replace_slashes(self, tmp_path, monkeypatch):
        # Tmux session names contain slashes; filesystem paths must not nest
        # them as directories or status/list logic would walk into surprises.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        name = "hermes/lyra/claude/fix-stale-agm"
        assert "/" not in op.log_path(name).name
        assert "/" not in op.prompt_path(name).name


# ---------------------------------------------------------------------------
# Safety contract — bypass is refused, period
# ---------------------------------------------------------------------------


class TestPermissionModeSafety:
    def test_auto_is_allowed(self):
        # Bryan's required default — auto-accepts without bypassing safety hooks.
        assert op.validate_permission_mode("auto") == "auto"

    def test_accept_edits_is_allowed(self):
        assert op.validate_permission_mode("acceptEdits") == "acceptEdits"

    def test_default_is_allowed(self):
        # Used for risky DDL where Hermes forwards prompts to Bryan.
        assert op.validate_permission_mode("default") == "default"

    def test_plan_is_allowed(self):
        assert op.validate_permission_mode("plan") == "plan"

    @pytest.mark.parametrize(
        "bypass_mode",
        [
            "bypassPermissions",
            "bypass",
            "dangerouslySkipPermissions",
            "dontAsk",
        ],
    )
    def test_bypass_and_dontask_modes_are_refused(self, bypass_mode):
        with pytest.raises(op.OperatorError, match="refused"):
            op.validate_permission_mode(bypass_mode)

    def test_unknown_mode_is_refused(self):
        with pytest.raises(op.OperatorError, match="unknown"):
            op.validate_permission_mode("yolo")

    def test_build_command_refuses_bypass(self, tmp_path):
        with pytest.raises(op.OperatorError):
            op.build_claude_command(
                workdir=tmp_path,
                permission_mode="bypassPermissions",
                prompt="hi",
            )

    def test_build_command_refuses_dontask(self, tmp_path):
        with pytest.raises(op.OperatorError):
            op.build_claude_command(
                workdir=tmp_path,
                permission_mode="dontAsk",
                prompt="hi",
            )


# ---------------------------------------------------------------------------
# Command construction — argv shape, acceptEdits presence, prompt placement
# ---------------------------------------------------------------------------


class TestBuildClaudeCommand:
    def test_auto_is_default_in_argv(self, tmp_path):
        argv = op.build_claude_command(workdir=tmp_path, prompt="do thing")
        assert "--permission-mode" in argv
        idx = argv.index("--permission-mode")
        assert argv[idx + 1] == "auto"

    def test_accept_edits_passes_through_when_requested(self, tmp_path):
        argv = op.build_claude_command(
            workdir=tmp_path, prompt="do thing", permission_mode="acceptEdits"
        )
        idx = argv.index("--permission-mode")
        assert argv[idx + 1] == "acceptEdits"

    def test_no_bypass_token_ever_in_argv(self, tmp_path):
        argv = op.build_claude_command(workdir=tmp_path, prompt="do thing")
        forbidden = {
            "--dangerously-skip-permissions",
            "--bypass-permissions",
            "bypassPermissions",
            "dontAsk",
        }
        assert not (forbidden & set(argv))

    def test_workdir_added_with_add_dir(self, tmp_path):
        argv = op.build_claude_command(workdir=tmp_path, prompt="do thing")
        assert "--add-dir" in argv
        idx = argv.index("--add-dir")
        assert argv[idx + 1] == str(tmp_path)

    def test_agent_preset_propagates(self, tmp_path):
        argv = op.build_claude_command(
            workdir=tmp_path, prompt="x", agent="backend-builder"
        )
        assert "--agent" in argv
        idx = argv.index("--agent")
        assert argv[idx + 1] == "backend-builder"

    def test_prompt_appended_as_positional(self, tmp_path):
        argv = op.build_claude_command(workdir=tmp_path, prompt="ship it")
        # Claude Code consumes the first positional as the initial user message.
        assert argv[-1] == "ship it"

    def test_extra_args_pass_through(self, tmp_path):
        argv = op.build_claude_command(
            workdir=tmp_path, prompt="x", extra_args=["--model", "opus"]
        )
        assert "--model" in argv
        assert "opus" in argv

    def test_binary_overrideable(self, tmp_path):
        argv = op.build_claude_command(
            workdir=tmp_path, prompt="x", binary="/usr/local/bin/claude"
        )
        assert argv[0] == "/usr/local/bin/claude"


# ---------------------------------------------------------------------------
# Tmux launch composition — log redirection must be present
# ---------------------------------------------------------------------------


class TestTmuxLaunchCommand:
    def test_tmux_new_session_detached(self, tmp_path):
        argv = op.tmux_launch_command(
            session="hermes/lyra/claude/x",
            workdir=tmp_path,
            claude_argv=["claude", "--permission-mode", "auto", "go"],
            log_file=tmp_path / "x.log",
        )
        assert argv[0] == "tmux"
        assert "new-session" in argv
        assert "-d" in argv

    def test_log_file_teed(self, tmp_path):
        log = tmp_path / "x.log"
        argv = op.tmux_launch_command(
            session="hermes/lyra/claude/x",
            workdir=tmp_path,
            claude_argv=["claude", "--permission-mode", "auto", "go"],
            log_file=log,
        )
        shell_cmd = argv[-1]
        assert "tee -a" in shell_cmd
        assert str(log) in shell_cmd

    def test_workdir_in_cd(self, tmp_path):
        argv = op.tmux_launch_command(
            session="hermes/lyra/claude/x",
            workdir=tmp_path,
            claude_argv=["claude", "go"],
            log_file=tmp_path / "x.log",
        )
        shell_cmd = argv[-1]
        assert f"cd {tmp_path}" in shell_cmd or f"cd '{tmp_path}'" in shell_cmd

    def test_prompts_with_quotes_dont_break_shell(self, tmp_path):
        # Prompts come from users; they will contain quotes. The launcher must
        # use shlex.quote so the shell parses them as a single token.
        prompt = """She said "ship it" and didn't blink"""
        argv = op.tmux_launch_command(
            session="hermes/lyra/claude/x",
            workdir=tmp_path,
            claude_argv=["claude", "--permission-mode", "auto", prompt],
            log_file=tmp_path / "x.log",
        )
        shell_cmd = argv[-1]
        # Whichever quoting scheme shlex picked, the prompt is preserved exactly
        # somewhere in the command.
        import shlex

        # Re-parse and assert the prompt round-trips as a single argv token.
        parsed = shlex.split(shell_cmd)
        assert prompt in parsed


# ---------------------------------------------------------------------------
# Attach command — surface for human-visible watchers
# ---------------------------------------------------------------------------


class TestAttachCommand:
    def test_attach_quotes_session_name(self):
        cmd = op.attach_command("hermes/lyra/claude/fix-stale-agm")
        assert cmd.startswith("tmux attach -t")
        assert "hermes/lyra/claude/fix-stale-agm" in cmd
