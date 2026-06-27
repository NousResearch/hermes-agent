"""Tests for parent→subparser flag propagation.

When flags like --yolo, -w, -s exist on both the parent parser and the 'chat'
subparser, placing the flag BEFORE the subcommand (e.g. 'hermes --yolo chat')
must not silently drop the flag value.

Regression test for: argparse subparser default=False overwriting parent's
parsed True when the same argument is defined on both parsers.

Fix: chat subparser uses default=argparse.SUPPRESS for all duplicated flags,
so the subparser only sets the attribute when the user explicitly provides it.
"""

import argparse
import os
import sys

import pytest


def _build_parser():
    """Build the hermes argument parser from the real code.

    We import the real main() and extract the parser it builds.
    Since main() is a large function that does much more than parse args,
    we replicate just the parser structure here to avoid side effects.
    """
    parser = argparse.ArgumentParser(prog="hermes")
    parser.add_argument("--resume", "-r", metavar="SESSION", default=None)
    parser.add_argument(
        "--continue", "-c", dest="continue_last", nargs="?",
        const=True, default=None, metavar="SESSION_NAME",
    )
    parser.add_argument("--worktree", "-w", action="store_true", default=False)
    parser.add_argument("--skills", "-s", action="append", default=None)
    parser.add_argument("--yolo", action="store_true", default=False)
    parser.add_argument("--pass-session-id", action="store_true", default=False)

    subparsers = parser.add_subparsers(dest="command")
    chat = subparsers.add_parser("chat")
    # These MUST use argparse.SUPPRESS to avoid overwriting parent values
    chat.add_argument("--yolo", action="store_true",
                      default=argparse.SUPPRESS)
    chat.add_argument("--worktree", "-w", action="store_true",
                      default=argparse.SUPPRESS)
    chat.add_argument("--skills", "-s", action="append",
                      default=argparse.SUPPRESS)
    chat.add_argument("--pass-session-id", action="store_true",
                      default=argparse.SUPPRESS)
    chat.add_argument("--resume", "-r", metavar="SESSION_ID",
                      default=argparse.SUPPRESS)
    chat.add_argument(
        "--continue", "-c", dest="continue_last", nargs="?",
        const=True, default=argparse.SUPPRESS, metavar="SESSION_NAME",
    )
    return parser


class TestChatVerboseArg:
    """Verify chat --verbose preserves config fallback when absent."""

    def test_chat_without_verbose_leaves_attribute_unset(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["chat"])

        assert not hasattr(args, "verbose")

    def test_chat_verbose_sets_attribute_true(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["chat", "--verbose"])

        assert args.verbose is True

    def test_cmd_chat_forwards_none_when_verbose_is_absent(self, monkeypatch):
        import types
        import sys

        import hermes_cli.main as main_mod
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=main_mod.cmd_chat)
        args = parser.parse_args(["chat"])
        captured = {}
        fake_cli = types.ModuleType("cli")

        def fake_main(**kwargs):
            captured.update(kwargs)

        setattr(fake_cli, "main", fake_main)
        fake_banner = types.ModuleType("hermes_cli.banner")
        setattr(fake_banner, "prefetch_update_check", lambda: None)
        fake_skills_sync = types.ModuleType("tools.skills_sync")
        setattr(fake_skills_sync, "sync_skills", lambda quiet=True: None)

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)

        main_mod.cmd_chat(args)

        assert captured["quiet"] is False
        assert "verbose" not in captured


class TestYoloEnvVar:
    """Verify --yolo sets HERMES_YOLO_MODE regardless of flag position.

    This tests the actual cmd_chat logic pattern (getattr → os.environ).
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self):
        os.environ.pop("HERMES_YOLO_MODE", None)
        yield
        os.environ.pop("HERMES_YOLO_MODE", None)

    def _simulate_cmd_chat_yolo_check(self, args):
        """Replicate the exact check from cmd_chat in main.py."""
        if getattr(args, "yolo", False):
            os.environ["HERMES_YOLO_MODE"] = "1"

    def test_yolo_before_chat_sets_env(self):
        parser = _build_parser()
        args = parser.parse_args(["--yolo", "chat"])
        self._simulate_cmd_chat_yolo_check(args)
        assert os.environ.get("HERMES_YOLO_MODE") == "1"

    def test_yolo_after_chat_sets_env(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "--yolo"])
        self._simulate_cmd_chat_yolo_check(args)
        assert os.environ.get("HERMES_YOLO_MODE") == "1"

    def test_no_yolo_no_env(self):
        parser = _build_parser()
        args = parser.parse_args(["chat"])
        self._simulate_cmd_chat_yolo_check(args)
        assert os.environ.get("HERMES_YOLO_MODE") is None


class TestAcceptHooksOnAgentSubparsers:
    """Verify --accept-hooks is accepted at every agent-subcommand
    position (before the subcommand, between group/subcommand, and
    after the leaf subcommand) for gateway/cron/mcp/acp.  Regression
    against prior behaviour where the flag only worked on the root
    parser and `chat`, so `hermes gateway run --accept-hooks` failed
    with `unrecognized arguments`."""

    @pytest.mark.parametrize("argv", [
        ["--accept-hooks", "gateway", "run", "--help"],
        ["gateway", "--accept-hooks", "run", "--help"],
        ["gateway", "run", "--accept-hooks", "--help"],
        ["--accept-hooks", "cron", "tick", "--help"],
        ["cron", "--accept-hooks", "tick", "--help"],
        ["cron", "tick", "--accept-hooks", "--help"],
        ["cron", "run", "--accept-hooks", "dummy-id", "--help"],
        ["--accept-hooks", "mcp", "serve", "--help"],
        ["mcp", "--accept-hooks", "serve", "--help"],
        ["mcp", "serve", "--accept-hooks", "--help"],
        ["acp", "--accept-hooks", "--help"],
    ])
    def test_accepted_at_every_position(self, argv):
        """Invoking `hermes <argv>` must exit 0 (help) rather than
        failing with `unrecognized arguments`."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"argv={argv!r} returned {result.returncode}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        assert "unrecognized arguments" not in result.stderr


def test_sessions_metadata_updates_current_session(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "current-session"

        def update_session_custom_metadata(self, session_id, updates, *, remove=None):
            captured["updated"] = (session_id, updates, remove)
            return True

        def get_session_custom_metadata(self, session_id):
            captured["read"] = session_id
            return {"issue_id": "HERMES-123", "priority": 2}

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setenv("HERMES_SESSION_ID", "current-session")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "metadata", "--current", "--set", "issue_id=HERMES-123", "--set", "priority=2"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "resolved_from": "current-session",
        "updated": ("current-session", {"issue_id": "HERMES-123", "priority": 2}, []),
        "read": "current-session",
        "closed": True,
    }
    assert "Updated metadata for session 'current-session'." in output
    assert '"issue_id": "HERMES-123"' in output


def test_sessions_metadata_replaces_with_json(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def set_session_custom_metadata(self, session_id, metadata):
            captured["set"] = (session_id, metadata)
            return True

        def get_session_custom_metadata(self, session_id):
            return captured["set"][1]

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "metadata", "s1", "--json", '{"summary":"done"}'])

    main_mod.main()

    assert captured["set"] == ("s1", {"summary": "done"})
    assert '"summary": "done"' in capsys.readouterr().out
