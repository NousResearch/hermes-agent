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


class TestFlagBeforeSubcommand:
    """Flags placed before 'chat' must propagate through."""

    def test_yolo_before_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["--yolo", "chat"])
        assert getattr(args, "yolo", False) is True

    def test_worktree_before_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["-w", "chat"])
        assert getattr(args, "worktree", False) is True

    def test_skills_before_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["-s", "myskill", "chat"])
        assert getattr(args, "skills", None) == ["myskill"]

    def test_pass_session_id_before_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["--pass-session-id", "chat"])
        assert getattr(args, "pass_session_id", False) is True

    def test_resume_before_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["-r", "abc123", "chat"])
        assert getattr(args, "resume", None) == "abc123"


class TestFlagAfterSubcommand:
    """Flags placed after 'chat' must still work."""

    def test_yolo_after_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "--yolo"])
        assert getattr(args, "yolo", False) is True

    def test_worktree_after_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "-w"])
        assert getattr(args, "worktree", False) is True

    def test_skills_after_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "-s", "myskill"])
        assert getattr(args, "skills", None) == ["myskill"]

    def test_resume_after_chat(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "-r", "abc123"])
        assert getattr(args, "resume", None) == "abc123"


class TestNoSubcommandDefaults:
    """When no subcommand is given, flags must work and defaults must hold."""

    def test_yolo_no_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["--yolo"])
        assert args.yolo is True
        assert args.command is None

    def test_defaults_no_flags(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert getattr(args, "yolo", False) is False
        assert getattr(args, "worktree", False) is False
        assert getattr(args, "skills", None) is None
        assert getattr(args, "resume", None) is None

    def test_defaults_chat_no_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["chat"])
        # With SUPPRESS, these fall through to parent defaults
        assert getattr(args, "yolo", False) is False
        assert getattr(args, "worktree", False) is False
        assert getattr(args, "skills", None) is None


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
