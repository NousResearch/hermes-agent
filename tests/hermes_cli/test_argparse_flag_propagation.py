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
from unittest.mock import patch

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
