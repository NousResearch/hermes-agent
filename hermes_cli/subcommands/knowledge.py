"""CLI parser for bounded, local knowledge synchronization."""

from __future__ import annotations

from typing import Callable


def build_knowledge_parser(subparsers, *, cmd_knowledge: Callable) -> None:
    """Attach the ``knowledge`` command without importing its engine."""
    parser = subparsers.add_parser(
        "knowledge",
        help="Synchronize bounded, source-backed knowledge into built-in memory",
        description=(
            "Read a bounded JSON manifest and synchronize accepted facts into "
            "the active profile's MEMORY.md. No gateway, cron, provider, "
            "upload, or automatic merge is started."
        ),
    )
    actions = parser.add_subparsers(dest="knowledge_command")

    sync = actions.add_parser(
        "sync",
        help="Preview or apply a knowledge manifest",
        description="Use exactly one of --dry-run or --apply.",
    )
    sync.add_argument("--manifest", required=True, help="Path to a JSON manifest")
    modes = sync.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="Validate without writing")
    modes.add_argument("--apply", action="store_true", help="Write accepted facts locally")
    sync.add_argument("--json", action="store_true", help="Print machine-readable output")
    sync.set_defaults(func=cmd_knowledge)

    verify = actions.add_parser(
        "verify",
        help="Verify a previous synchronization run",
    )
    verify.add_argument("--run-id", required=True, help="Run identifier from sync output")
    verify.add_argument("--json", action="store_true", help="Print machine-readable output")
    verify.set_defaults(func=cmd_knowledge)

    parser.set_defaults(func=lambda args: parser.print_help())
