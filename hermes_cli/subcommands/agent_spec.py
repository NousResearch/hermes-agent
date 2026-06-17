"""``hermes agent-spec`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_agent_spec_parser(subparsers, *, cmd_agent_spec: Callable) -> None:
    parser = subparsers.add_parser(
        "agent-spec",
        help="Validate and preview typed Hermes agent specs (read-only)",
    )
    sub = parser.add_subparsers(dest="agent_spec_command")

    validate = sub.add_parser("validate", help="Validate an agent spec file")
    validate.add_argument("path_or_id")
    validate.add_argument("--profile")
    validate.add_argument("--json", action="store_true", help="Emit JSON output")
    validate.add_argument("--strict", action="store_true", help="Treat warnings as failures")

    preview = sub.add_parser("preview", help="Preview effective policy for a profile/spec")
    preview.add_argument("--profile", required=True)
    preview.add_argument("--spec")
    preview.add_argument("--json", action="store_true", help="Emit JSON output")
    preview.add_argument("--strict", action="store_true", help="Treat warnings as failures")

    list_p = sub.add_parser("list", help="List agent specs or profile coverage")
    list_p.add_argument("--profiles", action="store_true", required=True)
    list_p.add_argument("--json", action="store_true", help="Emit JSON output")

    parser.set_defaults(func=cmd_agent_spec)
