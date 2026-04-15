"""Tests for the defensive subparser routing workaround (bpo-9338).

The main() function in hermes_cli/main.py sets subparsers.required=True
when argv contains a known subcommand name.  This forces deterministic
routing on Python versions where argparse fails to match subcommand tokens
when the parent parser has nargs='?' optional arguments (--continue).

If the subcommand token is consumed as a flag value (e.g. `hermes -c model`
to resume a session named 'model'), the required=True parse raises
SystemExit and the code falls back to the default required=False behaviour.
"""
import argparse
import io
import sys

import pytest


def _build_parser():
    """Build a minimal replica of the hermes top-level parser."""
    parser = argparse.ArgumentParser(prog="hermes")
    parser.add_argument("--version", "-V", action="store_true")
    parser.add_argument("--resume", "-r", metavar="SESSION", default=None)
    parser.add_argument(
        "--continue", "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_NAME",
    )
    parser.add_argument("--worktree", "-w", action="store_true", default=False)
    parser.add_argument("--skills", "-s", action="append", default=None)
    parser.add_argument("--yolo", action="store_true", default=False)
    parser.add_argument("--pass-session-id", action="store_true", default=False)

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    chat_p = subparsers.add_parser("chat")
    chat_p.add_argument("-q", "--query", default=None)
    subparsers.add_parser("model")
    subparsers.add_parser("gateway")
    subparsers.add_parser("setup")
    return parser, subparsers


def _safe_parse(parser, subparsers, argv):
    """Replica of the defensive parsing logic from main()."""
    known_cmds = set(subparsers.choices.keys()) if hasattr(subparsers, "choices") else set()
    has_cmd_token = any(t in known_cmds for t in argv if not t.startswith("-"))

    if has_cmd_token:
        subparsers.required = True
        saved_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            args = parser.parse_args(argv)
            sys.stderr = saved_stderr
            return args
        except SystemExit:
            sys.stderr = saved_stderr
            subparsers.required = False
            return parser.parse_args(argv)
    else:
        subparsers.required = False
        return parser.parse_args(argv)


class TestSubparserRoutingFallback:
    """Verify the bpo-9338 defensive routing works for all key cases."""

    def test_direct_subcommand(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["model"])
        assert args.command == "model"

    def test_subcommand_with_flags(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["--yolo", "model"])
        assert args.command == "model"
        assert args.yolo is True

    def test_bare_hermes_defaults_to_none(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, [])
        assert args.command is None

    def test_flags_only_defaults_to_none(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["--yolo"])
        assert args.command is None
        assert args.yolo is True

    def test_continue_flag_alone(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-c"])
        assert args.command is None
        assert args.continue_last is True

    def test_continue_with_session_name(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-c", "myproject"])
        assert args.command is None
        assert args.continue_last == "myproject"

    def test_continue_with_subcommand_name_as_session(self):
        """Edge case: session named 'model' — should be treated as session name, not subcommand."""
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-c", "model"])
        assert args.command is None
        assert args.continue_last == "model"

    def test_continue_with_session_then_subcommand(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-c", "myproject", "model"])
        assert args.command == "model"
        assert args.continue_last == "myproject"

    def test_chat_with_query(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["chat", "-q", "hello"])
        assert args.command == "chat"
        assert args.query == "hello"

    def test_resume_flag(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-r", "abc123"])
        assert args.command is None
        assert args.resume == "abc123"

    def test_resume_with_subcommand(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-r", "abc123", "chat"])
        assert args.command == "chat"
        assert args.resume == "abc123"

    def test_skills_flag_with_subcommand(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["-s", "myskill", "chat"])
        assert args.command == "chat"
        assert args.skills == ["myskill"]

    def test_all_flags_with_subcommand(self):
        parser, sub = _build_parser()
        args = _safe_parse(parser, sub, ["--yolo", "-w", "-s", "myskill", "model"])
        assert args.command == "model"
        assert args.yolo is True
        assert args.worktree is True
        assert args.skills == ["myskill"]
