"""``hermes context`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_context_parser(subparsers, *, cmd_context: Callable) -> None:
    """Attach the ``context`` subcommand tree to ``subparsers``."""
    context_parser = subparsers.add_parser(
        "context",
        help="Inspect context/prompt budget surfaces",
        description="Inspect fixed context and prompt budget sources without making an API call.",
    )
    sub = context_parser.add_subparsers(dest="context_command")
    audit = sub.add_parser(
        "audit",
        help="Measure project context, skills, memory, and tool-schema payloads",
        description="Read-only context budget audit. Runs offline and prints sizes only.",
    )
    audit.add_argument("--cwd", default=None, help="Directory whose project context files should be measured")
    audit.add_argument("--platform", default="cli", help="Platform to simulate for prompt sizing. Default: cli")
    audit.add_argument("--json", action="store_true", help="Emit JSON")
    audit.set_defaults(func=cmd_context)
    context_parser.set_defaults(func=cmd_context, context_command="audit")
