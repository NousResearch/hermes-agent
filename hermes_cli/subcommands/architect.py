"""``hermes architect`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_architect_parser(subparsers, *, cmd_architect: Callable) -> None:
    parser = subparsers.add_parser(
        "architect",
        help="Review architecture-first project readiness",
        description="Review a project before implementation and report whether it is ready to proceed.",
    )
    nested = parser.add_subparsers(dest="architect_command")
    review = nested.add_parser(
        "review",
        help="Review a project for architecture readiness",
    )
    review.add_argument("project")
    review.add_argument("--projects-root")
    review.add_argument("--scope", default="", help="Comma-separated review categories to focus on")
    review.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    review.add_argument(
        "--block-on-critical",
        action="store_true",
        help="Exit with blocked status when critical gaps are present",
    )
    review.add_argument(
        "--write-report",
        action="store_true",
        help="Write the review report as a project artifact",
    )
    review.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate missing architecture document stubs",
    )
    review.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated documents/reports to overwrite existing files",
    )
    review.add_argument(
        "--persist",
        action="store_true",
        help="Persist the review report to the Hermes OS SQLite store",
    )
    review.add_argument(
        "--generate-tasks",
        action="store_true",
        help="Generate TASKS.md and .hermes/tasks.json from the review",
    )
    review.add_argument(
        "--db",
        default="",
        help="Optional SQLite database path for --persist",
    )
    review.set_defaults(func=cmd_architect)
    parser.set_defaults(func=cmd_architect)
