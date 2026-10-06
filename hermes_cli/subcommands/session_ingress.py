"""Execution ingress for an existing live session."""
from __future__ import annotations


def build_session_ingress_parsers(subparsers) -> None:
    for action, help_text in (
        ("queue", "Submit a next user turn after the current run"),
        ("steer", "Steer the active turn, or submit a next turn when idle"),
    ):
        parser = subparsers.add_parser(action, help=help_text)
        parser.add_argument("--session", "-s", required=True,
                            help="Exact durable ID, live/UI ID or unique exact name")
        parser.add_argument("--message", "-m", required=True, help="Prompt text")
        parser.set_defaults(func=cmd_session_ingress)


def cmd_session_ingress(args) -> int:
    from hermes_cli.session_ingress import send_live_message
    return send_live_message(args.session, args.message, args.command)
