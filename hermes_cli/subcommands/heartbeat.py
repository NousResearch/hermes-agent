"""``hermes heartbeat`` subcommand parser — inspect and edit per-session heartbeats from the shell."""

from __future__ import annotations

import argparse
from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_heartbeat_parser(subparsers, *, cmd_heartbeat: Callable) -> None:
    """Attach the ``heartbeat`` subcommand (and its sub-actions) to ``subparsers``."""
    parser = subparsers.add_parser(
        "heartbeat", help="Session heartbeat management (list, set, status, pause, resume, clear)",
        description="Inspect and edit the recurring re-entry prompt (heartbeat) attached to a session.\n"
            "State lives in the profile's state.db; the gateway poller picks up changes to a\n"
            "gateway-routed session without a restart. Use `hermes sessions list --source telegram`\n"
            "to find session ids.",
        epilog="Examples:\n"
            "  hermes heartbeat list\n"
            "  hermes heartbeat set 20260920_130453_bbcc00aa --every 30m --prompt 'Check the board'\n"
            "  hermes heartbeat status 20260920_130453_bbcc00aa\n"
            "  hermes heartbeat pause 20260920_130453_bbcc00aa\n"
            "  hermes -p work heartbeat clear 20260920_130453_bbcc00aa\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="heartbeat_command")

    p_list = sub.add_parser("list", help="List every session with a heartbeat row in this profile")
    p_list.add_argument("--all", action="store_true",
        help="Include cleared heartbeats (default: active and paused only)")
    add_json_flag(p_list, "Machine-readable output")

    p_set = sub.add_parser("set", help="Create or replace the heartbeat on an existing session")
    p_set.add_argument("session_id", help="Session id (see `hermes sessions list`)")
    p_set.add_argument("--every", required=True, metavar="INTERVAL",
        help="Interval like 30m, 2h, 'every 90 minutes' (minimum 60s)")
    src = p_set.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Recurring instruction text")
    src.add_argument("--prompt-file", metavar="PATH", help="Read the instruction text from a file")

    for name, help_text in (
        ("status", "Show one session's heartbeat"),
        ("pause", "Pause a heartbeat (state kept)"),
        ("resume", "Resume a paused heartbeat (re-anchored: no immediate fire)"),
        ("clear", "Clear a heartbeat"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("session_id", help="Session id")
        if name == "status":
            add_json_flag(p, "Machine-readable output")

    parser.set_defaults(func=cmd_heartbeat)
