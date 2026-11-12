"""Tests for the ``hermes hooks`` CLI subcommand."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import shell_hooks
from hermes_cli import hooks as hooks_cli


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


def _hook_script(tmp_path: Path, body: str, name: str = "hook.sh") -> Path:
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def _run(sub_args: SimpleNamespace) -> str:
    """Capture stdout for a hooks_command invocation."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli.hooks_command(sub_args)
    return buf.getvalue()


# ── list ──────────────────────────────────────────────────────────────────


class TestHooksList:

    def test_shows_configured_and_consent_status(self, tmp_path):
        script = _hook_script(
            tmp_path, "#!/usr/bin/env bash\nprintf '{}\\n'\n",
        )
        cfg = {
            "hooks": {
                "pre_tool_call": [
                    {"matcher": "terminal", "command": str(script), "timeout": 30},
                ],
                "on_session_start": [
                    {"command": str(script)},
                ],
            }
        }

        # Approve one of the two so we can see both states in the output
        shell_hooks._record_approval("pre_tool_call", str(script))

        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(hooks_action="list"))

        assert "[pre_tool_call]" in out
        assert "[on_session_start]" in out
        assert "✓ allowed" in out
        assert "✗ not allowlisted" in out
        assert str(script) in out


# ── test ──────────────────────────────────────────────────────────────────


@pytest.mark.platforms("linux")
class TestHooksTest:
    def test_synthetic_payload_matches_production_shape(self, tmp_path):
        """`hermes hooks test` must feed the script stdin in the same
        shape invoke_hook() would at runtime.  Prior to this fix,
        run_once bypassed _serialize_payload and the two paths diverged —
        scripts tested with `hermes hooks test` saw different top-level
        keys than at runtime, silently breaking in production."""
        capture = tmp_path / "captured.json"
        script = _hook_script(
            tmp_path,
            f"#!/usr/bin/env bash\ncat - > {capture}\nprintf '{{}}\\n'\n",
        )
        cfg = {"hooks": {"subagent_stop": [{"command": str(script)}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            _run(SimpleNamespace(
                hooks_action="test", event="subagent_stop",
                for_tool=None, payload_file=None,
            ))

        seen = json.loads(capture.read_text())
        # Same top-level keys _serialize_payload produces at runtime
        assert set(seen.keys()) == {
            "hook_event_name", "tool_name", "tool_input",
            "session_id", "cwd", "extra", "profile",
        }
        # parent_session_id was routed to top-level session_id (matches runtime)
        assert seen["session_id"] == "parent-sess"
        assert "parent_session_id" not in seen["extra"]
        # subagent_stop has no tool, so tool_name / tool_input are null
        assert seen["tool_name"] is None
        assert seen["tool_input"] is None

    def test_fires_real_subprocess_and_parses_block(self, tmp_path):
        block_script = _hook_script(
            tmp_path,
            "#!/usr/bin/env bash\n"
            'printf \'{"decision": "block", "reason": "nope"}\\n\'\n',
            name="block.sh",
        )
        cfg = {
            "hooks": {
                "pre_tool_call": [
                    {"matcher": "terminal", "command": str(block_script)},
                ],
            },
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(
                hooks_action="test", event="pre_tool_call",
                for_tool="terminal", payload_file=None,
            ))

        # Parsed block appears in output
        assert '"action": "block"' in out
        assert '"message": "nope"' in out


# ── accept ────────────────────────────────────────────────────────────────


class TestHooksAccept:
    """`hermes hooks accept` records consent with no TTY (#138413).

    Non-TTY runs cannot answer the first-use prompt, and `--accept-hooks` /
    `HERMES_ACCEPT_HOOKS` are only resolved during a start-up registration pass.
    This subcommand writes the same allowlist entry the prompt writes, so an
    operator never has to hand-edit shell-hooks-allowlist.json.
    """

    def _approve(self, cfg, command=()):
        # stdin is NOT a tty: no prompt may be reachable on this path.
        with patch("sys.stdin") as mock_stdin, patch(
            "builtins.input", side_effect=AssertionError("must not prompt"),
        ), patch("hermes_cli.config.load_config", return_value=cfg):
            mock_stdin.isatty.return_value = False
            return _run(SimpleNamespace(
                hooks_action="accept", hook_command=list(command)))

    def test_non_tty_records_entry_matching_the_prompt_shape(self, tmp_path):
        script = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n")
        cfg = {"hooks": {"post_llm_call": [{"command": str(script)}]}}

        out = self._approve(cfg)

        assert "✓ approved" in out
        entry = shell_hooks.allowlist_entry_for("post_llm_call", str(script))
        assert entry is not None
        assert entry["event"] == "post_llm_call"
        assert entry["command"] == str(script)
        # Same keys the interactive prompt records via _record_approval.
        assert set(entry) == {
            "event", "command", "approved_at", "script_mtime_at_approval",
        }

    def test_accepts_every_not_yet_allowlisted_hook(self, tmp_path):
        a = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n", name="a.sh")
        b = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n", name="b.sh")
        cfg = {
            "hooks": {
                "post_llm_call": [{"command": str(a)}],
                "on_session_start": [{"command": str(b)}],
            }
        }

        out = self._approve(cfg)

        assert "2 new approval(s)" in out
        assert shell_hooks.allowlist_entry_for("post_llm_call", str(a)) is not None
        assert shell_hooks.allowlist_entry_for("on_session_start", str(b)) is not None

    def test_specific_command_approves_only_that_hook(self, tmp_path):
        a = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n", name="a.sh")
        b = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n", name="b.sh")
        cfg = {
            "hooks": {
                "post_llm_call": [{"command": str(a)}],
                "on_session_start": [{"command": str(b)}],
            }
        }

        self._approve(cfg, command=[str(a)])

        assert shell_hooks.allowlist_entry_for("post_llm_call", str(a)) is not None
        assert shell_hooks.allowlist_entry_for("on_session_start", str(b)) is None

    def test_reports_already_allowlisted_without_duplicating(self, tmp_path):
        script = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\\\n'\n")
        cfg = {"hooks": {"post_llm_call": [{"command": str(script)}]}}
        shell_hooks._record_approval("post_llm_call", str(script))

        out = self._approve(cfg)

        assert "already allowlisted" in out
        assert "0 new approval(s)" in out
        approvals = shell_hooks.load_allowlist()["approvals"]
        assert len([e for e in approvals if e.get("command") == str(script)]) == 1


def test_hooks_parser_exposes_accept_subcommand():
    """The `accept` action must be reachable from the real `hermes hooks` parser."""
    import argparse

    from hermes_cli.subcommands.hooks import build_hooks_parser

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_hooks_parser(sub, cmd_hooks=lambda _a: None)

    args = parser.parse_args(["hooks", "accept", "/tmp/hook.sh"])
    # Regression guard: the accept positional must not clobber the top-level
    # subcommand name (a clobbering list breaks `hermes` dispatch).
    assert args.command == "hooks"
    assert args.hooks_action == "accept"
    assert args.hook_command == ["/tmp/hook.sh"]

    # Bare `accept` (no command) approves every configured hook.
    assert parser.parse_args(["hooks", "accept"]).hook_command == []


# ── revoke ────────────────────────────────────────────────────────────────


class TestHooksRevoke:
    def test_revoke_removes_entry(self, tmp_path):
        script = _hook_script(tmp_path, "#!/usr/bin/env bash\n")
        shell_hooks._record_approval("on_session_start", str(script))

        out = _run(SimpleNamespace(hooks_action="revoke", command=str(script)))
        assert "Removed 1" in out
        assert shell_hooks.allowlist_entry_for(
            "on_session_start", str(script),
        ) is None


# ── doctor ────────────────────────────────────────────────────────────────


class TestHooksDoctor:


    def test_flags_mtime_drift(self, tmp_path, monkeypatch):
        """Allowlist with older mtime than current -> drift warning."""
        script = _hook_script(tmp_path, "#!/usr/bin/env bash\nprintf '{}\\n'\n")

        # Manually stash an allowlist entry with an old mtime
        from agent.shell_hooks import allowlist_path
        allowlist_path().parent.mkdir(parents=True, exist_ok=True)
        allowlist_path().write_text(json.dumps({
            "approvals": [
                {
                    "event": "on_session_start",
                    "command": str(script),
                    "approved_at": "2000-01-01T00:00:00Z",
                    "script_mtime_at_approval": "2000-01-01T00:00:00Z",
                }
            ]
        }))

        cfg = {"hooks": {"on_session_start": [{"command": str(script)}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(hooks_action="doctor"))
        assert "modified since approval" in out


    def test_unallowlisted_script_is_not_executed(self, tmp_path):
        """Regression for M4: `hermes hooks doctor` used to run every
        listed script against a synthetic payload as part of its JSON
        smoke test, which contradicted the documented workflow of
        "spot newly-added hooks *before they register*".  An un-allowlisted
        script must not be executed during `doctor`."""
        sentinel = tmp_path / "executed"
        # Script would touch the sentinel if executed; we assert it wasn't.
        script = _hook_script(
            tmp_path,
            f"#!/usr/bin/env bash\ntouch {sentinel}\nprintf '{{}}\\n'\n",
        )
        cfg = {"hooks": {"on_session_start": [{"command": str(script)}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run(SimpleNamespace(hooks_action="doctor"))

        assert not sentinel.exists(), (
            "doctor executed an un-allowlisted script — "
            "M4 gate regressed"
        )
        assert "not allowlisted" in out.lower()
        assert "skipped JSON smoke test" in out


def test_print_run_result_shows_decision_for_error_and_timeout():
    """A failing hook's decision must be printed, not hidden by early returns (#115968).

    run_once always sets ``parsed`` (agent/shell_hooks.py::_evaluate_result), so `hooks test` must show
    it even when the hook errored or timed out. Before this fix the ``return`` on error/timeout skipped
    the parsed tail, so a fail-closed blocker rendered identically to a fail-open pass-through.
    """
    parsed = {"action": "block", "message": "hook failed closed: command not found"}

    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli._print_run_result(
            {"error": "command not found", "parsed": parsed}
        )
    out = buf.getvalue()
    assert "✗ error: command not found" in out
    assert '"action": "block"' in out
    assert '"message": "hook failed closed: command not found"' in out

    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli._print_run_result(
            {"timed_out": True, "elapsed_seconds": 2.33, "parsed": parsed}
        )
    out = buf.getvalue()
    assert "✗ timed out after 2.33s" in out
    assert '"action": "block"' in out


def test_hooks_test_distinguishes_fail_closed_from_fail_open(tmp_path):
    """`hermes hooks test` on a missing command shows the dispatcher's decision (#115968).

    Drives the real CLI subcommand: a fail_closed hook whose command does not exist
    must print the block decision, while the fail-open twin prints the "contributed
    nothing" line — the two must not render identically.
    """
    cfg = {
        "hooks": {
            "pre_tool_call": [
                {"matcher": "terminal", "command": "/nonexistent/hook-closed.sh",
                 "fail_closed": True},
                {"matcher": "terminal", "command": "/nonexistent/hook-open.sh"},
            ],
        },
        "hooks_auto_accept": True,
    }
    with patch("hermes_cli.config.load_config", return_value=cfg):
        out = _run(SimpleNamespace(
            hooks_action="test", event="pre_tool_call",
            for_tool="terminal", payload_file=None,
        ))

    closed, open_ = out.split("/nonexistent/hook-open.sh", 1)
    assert "✗ error:" in closed and "✗ error:" in open_
    assert '"action": "block"' in closed
    assert "failed closed" in closed
    assert '"action": "block"' not in open_
    assert "contributed nothing" in open_
