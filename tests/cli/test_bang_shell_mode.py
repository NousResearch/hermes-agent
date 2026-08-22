"""Tests for `!<command>` shell mode in the interactive CLI.

Covers bang detection/parsing, that the terminal tool's approval gate is
invoked for a dangerous command, that non-zero exit codes surface, and the
load-bearing invariant: a bang command leaves conversation_history
byte-identical because it never becomes a turn.
"""
import copy
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.bang_shell import (
    USAGE_HINT,
    bang_shell_enabled,
    is_bang_command,
    parse_bang_command,
    run_bang_command,
)


# ── detection / parsing ────────────────────────────────────────────────────

class TestBangDetection:
    @pytest.mark.parametrize("text", [
        "!ls",
        "!git status",
        "  !ls -la",          # leading whitespace still counts
        "!  spaced out",      # `!` followed by spaces
        "!!",                 # double bang
        "!",                  # bare bang
    ])
    def test_leading_bang_is_bang(self, text):
        assert is_bang_command(text) is True

    @pytest.mark.parametrize("text", [
        "fix the bug!",                 # trailing `!` in prose
        "echo hi! please",              # mid-text `!`
        "run this: !ls",                # `!` not at the start
        "/help",
        "hello world",
        "",
        "   ",
        None,
        123,
    ])
    def test_non_leading_bang_is_not_bang(self, text):
        assert is_bang_command(text) is False

    @pytest.mark.parametrize("text,expected", [
        ("!ls", "ls"),
        ("!git status", "git status"),
        ("  !ls -la  ", "ls -la"),
        ("!   echo hi", "echo hi"),
        ("!!", "!"),                     # second bang belongs to the command
        ("!!ls", "!ls"),
        ("!", ""),                       # bare bang → no command
        ("!   ", ""),
        ("not a bang", ""),
    ])
    def test_parse_strips_exactly_one_bang(self, text, expected):
        assert parse_bang_command(text) == expected


class TestBangContextGating:
    """Bang mode is CLI-only — gateway/cron users have their own shells."""

    def test_enabled_in_plain_cli(self, monkeypatch):
        for var in ("HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                    "HERMES_SESSION_PLATFORM"):
            monkeypatch.delenv(var, raising=False)
        assert bang_shell_enabled() is True

    @pytest.mark.parametrize("var,value", [
        ("HERMES_GATEWAY_SESSION", "1"),
        ("HERMES_CRON_SESSION", "true"),
        ("HERMES_SESSION_PLATFORM", "discord"),
    ])
    def test_disabled_in_non_cli_contexts(self, monkeypatch, var, value):
        for v in ("HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                  "HERMES_SESSION_PLATFORM"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(var, value)
        assert bang_shell_enabled() is False


# ── execution ──────────────────────────────────────────────────────────────

class TestBangExecution:
    def test_output_is_streamed_to_writer(self):
        lines = []
        code = run_bang_command("echo bang-one; echo bang-two", writer=lines.append)
        assert code == 0
        assert "bang-one" in lines
        assert "bang-two" in lines

    def test_stderr_is_merged_into_output(self):
        lines = []
        run_bang_command("echo to-stderr >&2", writer=lines.append)
        assert "to-stderr" in lines

    def test_nonzero_exit_code_is_returned(self):
        lines = []
        code = run_bang_command("exit 42", writer=lines.append)
        assert code == 42

    def test_runs_in_requested_cwd(self, tmp_path):
        lines = []
        code = run_bang_command("pwd", cwd=str(tmp_path), writer=lines.append)
        assert code == 0
        # macOS resolves /tmp through /private, so compare realpaths.
        assert os.path.realpath(lines[-1].strip()) == os.path.realpath(str(tmp_path))

    def test_missing_cwd_falls_back_without_crashing(self, tmp_path):
        lines = []
        code = run_bang_command(
            "echo ok", cwd=str(tmp_path / "does-not-exist"), writer=lines.append
        )
        assert code == 0
        assert "ok" in lines


# ── timeout / backgrounded-child containment ───────────────────────────────

class TestBangTimeout:
    """``timeout`` must be a real wall-clock ceiling on the composer.

    The module comment above ``DEFAULT_TIMEOUT`` states the invariant these
    pin: an accidental ``!sleep 999`` "should not wedge the composer".
    Draining the output pipe on the calling thread defeats that — the pipe
    only reaches EOF once the shell *and every descendant that inherited its
    write end* have closed stdout, so the deadline could only ever be applied
    after the work was already over.
    """

    def test_timeout_is_enforced_on_a_silent_command(self):
        lines = []
        started = time.monotonic()
        code = run_bang_command("sleep 5", timeout=1, writer=lines.append)
        elapsed = time.monotonic() - started

        assert code == 124  # GNU `timeout` convention
        assert any("timed out after 1s" in line for line in lines)
        assert elapsed < 3, f"deadline not enforced — returned after {elapsed:.1f}s"

    def test_returns_when_the_shell_exits_with_a_background_child(self):
        """A backgrounded grandchild must not hold the composer.

        ``(sleep 5 &) ; echo started`` exits immediately, but the grandchild
        inherited the write end of our stdout pipe via ``fork()`` — so reading
        to EOF holds the caller for the grandchild's whole lifetime. This is
        the hang ``tools/environments/base.py`` documents for the terminal
        tool. The generous ``timeout`` is deliberate: this pins the "stop
        draining once the shell itself is gone" contract, not the deadline.
        """
        lines = []
        started = time.monotonic()
        code = run_bang_command(
            "(sleep 5 &) ; echo started", timeout=60, writer=lines.append
        )
        elapsed = time.monotonic() - started

        assert code == 0
        assert "started" in lines
        assert elapsed < 3, f"blocked on the grandchild's pipe for {elapsed:.1f}s"

    def test_a_tail_printed_after_a_silence_is_streamed_in_full(self):
        """No-regression guard on bounding the drain.

        Not a red-before case — unpatched code streams this correctly too. It
        pins the risk the fix itself introduces: the post-exit flush window is
        measured from the last line seen *or* the shell's exit, whichever is
        later, so a command that goes quiet for longer than the idle grace and
        then prints on its way out still has its whole tail delivered, in
        order, rather than truncated by the new deadline.
        """
        lines = []
        code = run_bang_command(
            "sleep 1; for i in 1 2 3; do echo tail-$i; done",
            timeout=30,
            writer=lines.append,
        )
        assert code == 0
        assert [ln for ln in lines if ln.startswith("tail-")] == [
            "tail-1",
            "tail-2",
            "tail-3",
        ]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGPIPE semantics")
    def test_a_backgrounded_descendant_survives_the_composer_returning(self, tmp_path):
        """Handing the composer back must not kill what `!cmd &` launched.

        Once the deadline or the shell's exit has returned control, later
        output must not be printed into the composer — but the read end has to
        stay open regardless. Closing it hands EPIPE to the descendant on its
        next write, which would kill the very `!npm run dev &` server the user
        backgrounded. Output is discarded, not cut off.
        """
        marker = tmp_path / "descendant-finished"
        started = time.monotonic()
        lines = []
        # The two writes are spaced so the second one lands *after* any close
        # the drain thread might do on seeing the first — that second write is
        # what would take EPIPE.
        code = run_bang_command(
            f"( sleep 1; echo late-one; sleep 1; echo late-two; : > '{marker}' ) "
            f"& echo started",
            timeout=60,
            writer=lines.append,
        )
        elapsed = time.monotonic() - started

        assert code == 0
        assert "started" in lines
        assert elapsed < 3, f"blocked on the descendant for {elapsed:.1f}s"

        time.sleep(3.5)
        # Writes after the composer returned are dropped, never printed...
        assert not any(ln.startswith("late-") for ln in lines)
        # ...but the descendant ran to completion instead of dying on EPIPE.
        assert marker.exists(), "descendant was killed by the read end closing"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    def test_timeout_kills_descendants_not_just_the_shell(self, tmp_path):
        """The deadline must stop the whole tree, not just the shell wrapper.

        Mirrors ``tools/environments/local.py``, which runs terminal commands
        with ``start_new_session=True`` and kills the process *group*: killing
        only the direct child leaves grandchildren doing the work the user
        just asked to abort.
        """
        marker = tmp_path / "grandchild-survived"
        started = time.monotonic()
        code = run_bang_command(
            f"( sleep 2 && : > '{marker}' ) & sleep 8",
            timeout=1,
            writer=lambda line: None,
        )
        elapsed = time.monotonic() - started

        assert code == 124
        assert elapsed < 3, f"deadline not enforced — returned after {elapsed:.1f}s"
        # The grandchild writes the marker ~2s in if it outlived the kill.
        time.sleep(3)
        assert not marker.exists(), "grandchild survived the timeout kill"


# ── CLI handler: approval gate, usage hint, exit codes ─────────────────────

def _make_cli(history=None):
    """Build a HermesCLI shell with only what handle_bang_shell touches."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.session_id = "test-session"
    cli.conversation_history = [] if history is None else history
    cli._app = None
    return cli


def _printed(cli):
    """All text the CLI printed, flattened to plain strings."""
    out = []
    for call in cli.console.print.call_args_list:
        if not call.args:
            continue
        arg = call.args[0]
        out.append(getattr(arg, "plain", None) or str(arg))
    return out


class TestBangHandlerDispatch:
    def test_non_bang_text_is_not_handled(self):
        cli = _make_cli()
        assert cli.handle_bang_shell("please fix this bug!") is False
        assert cli.handle_bang_shell("/help") is False

    def test_bare_bang_prints_usage_and_runs_nothing(self):
        cli = _make_cli()
        with patch("hermes_cli.bang_shell.run_bang_command") as runner:
            assert cli.handle_bang_shell("!") is True
        runner.assert_not_called()
        assert any(USAGE_HINT in line for line in _printed(cli))

    def test_command_output_is_printed(self):
        cli = _make_cli()
        assert cli.handle_bang_shell("!echo hello-bang") is True
        assert any("hello-bang" in line for line in _printed(cli))

    def test_nonzero_exit_is_surfaced_to_the_user(self):
        cli = _make_cli()
        assert cli.handle_bang_shell("!exit 3") is True
        assert any("exited 3" in line for line in _printed(cli))

    def test_zero_exit_prints_no_exit_line(self):
        cli = _make_cli()
        cli.handle_bang_shell("!true")
        assert not any("exited" in line for line in _printed(cli))

    def test_disabled_context_falls_through(self, monkeypatch):
        """Gateway sessions must not execute bang commands."""
        cli = _make_cli()
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        with patch("hermes_cli.bang_shell.run_bang_command") as runner:
            assert cli.handle_bang_shell("!echo nope") is False
        runner.assert_not_called()


class TestBangApprovalGate:
    """A user-typed command still goes through the terminal tool's gate."""

    def test_approval_gate_is_invoked_for_a_dangerous_command(self):
        cli = _make_cli()
        gate = MagicMock(return_value={"approved": True, "message": None})
        with patch("tools.terminal_tool._check_all_guards", gate), \
             patch("hermes_cli.bang_shell.run_bang_command", return_value=0):
            cli.handle_bang_shell("!rm -rf ./build")

        gate.assert_called_once()
        assert gate.call_args.args[0] == "rm -rf ./build"

    def test_gate_is_invoked_for_every_command_not_just_dangerous_ones(self):
        cli = _make_cli()
        gate = MagicMock(return_value={"approved": True, "message": None})
        with patch("tools.terminal_tool._check_all_guards", gate), \
             patch("hermes_cli.bang_shell.run_bang_command", return_value=0):
            cli.handle_bang_shell("!ls")
        gate.assert_called_once()

    def test_denied_command_is_not_executed(self):
        cli = _make_cli()
        gate = MagicMock(return_value={
            "approved": False,
            "message": "Command denied: recursive delete",
        })
        with patch("tools.terminal_tool._check_all_guards", gate), \
             patch("hermes_cli.bang_shell.run_bang_command") as runner:
            assert cli.handle_bang_shell("!rm -rf /important") is True

        runner.assert_not_called()
        assert any("denied" in line.lower() for line in _printed(cli))

    def test_real_gate_blocks_a_hardline_command(self):
        """End-to-end through the real approval module — no execution."""
        cli = _make_cli()
        with patch("hermes_cli.bang_shell.run_bang_command") as runner:
            assert cli.handle_bang_shell("!rm -rf /") is True
        runner.assert_not_called()


# ── THE load-bearing invariant ─────────────────────────────────────────────

_SEED_HISTORY = [
    {"role": "system", "content": "You are Hermes."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hi there."},
    {"role": "user", "content": "list the files"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "a.py b.py"},
]


class TestBangLeavesHistoryByteIdentical:
    """Nothing about a bang command may enter conversation history.

    This is what makes `!` free (zero tokens, prompt cache untouched) and
    unable to break role alternation. Compared as serialized JSON so an added,
    removed, or mutated message anywhere in the list fails the assertion.
    """

    @pytest.mark.parametrize("submission", [
        "!echo history-check",     # succeeds
        "!exit 7",                 # non-zero exit
        "!",                       # bare bang / usage hint
        "!definitely-not-a-real-binary-xyz",  # command not found
    ])
    def test_history_is_byte_identical_before_and_after(self, submission):
        cli = _make_cli(history=copy.deepcopy(_SEED_HISTORY))
        before = json.dumps(cli.conversation_history, sort_keys=True)

        cli.handle_bang_shell(submission)

        after = json.dumps(cli.conversation_history, sort_keys=True)
        assert after == before, (
            f"bang command {submission!r} mutated conversation history"
        )
        assert len(cli.conversation_history) == len(_SEED_HISTORY)

    def test_history_unchanged_when_command_is_denied(self):
        cli = _make_cli(history=copy.deepcopy(_SEED_HISTORY))
        before = json.dumps(cli.conversation_history, sort_keys=True)

        gate = MagicMock(return_value={"approved": False, "message": "nope"})
        with patch("tools.terminal_tool._check_all_guards", gate):
            cli.handle_bang_shell("!rm -rf /important")

        assert json.dumps(cli.conversation_history, sort_keys=True) == before

    def test_agent_is_never_invoked(self):
        """No model turn: the agent object is not touched at all."""
        cli = _make_cli(history=copy.deepcopy(_SEED_HISTORY))
        agent = MagicMock()
        cli.agent = agent

        cli.handle_bang_shell("!echo no-model-turn")

        # No chat/steer/run/redirect call of any kind.
        assert agent.mock_calls == []
        assert json.dumps(cli.conversation_history, sort_keys=True) == json.dumps(
            _SEED_HISTORY, sort_keys=True
        )

    def test_role_alternation_is_preserved_across_many_bangs(self):
        cli = _make_cli(history=copy.deepcopy(_SEED_HISTORY))
        before = json.dumps(cli.conversation_history, sort_keys=True)

        for _ in range(5):
            cli.handle_bang_shell("!echo repeated")

        assert json.dumps(cli.conversation_history, sort_keys=True) == before
        roles = [m["role"] for m in cli.conversation_history]
        assert roles == ["system", "user", "assistant", "user", "assistant", "tool"]



