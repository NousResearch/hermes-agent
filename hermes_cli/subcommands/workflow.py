"""``hermes workflow`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_workflow_parser(subparsers, *, cmd_workflow: Callable) -> None:
    """Attach the ``workflow`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "workflow", help="Run a saved, repeatable sub-agent workflow",
        description="Validate a workflow manifest and inspect its checkpoint.")
    subs = parser.add_subparsers(dest="workflow_action")

    validate = subs.add_parser("validate", help="Validate a workflow manifest")
    validate.add_argument("manifest", help="Path to the workflow JSON manifest")

    status = subs.add_parser("status", help="Show completed and ready steps")
    status.add_argument("manifest", help="Path to the workflow JSON manifest")

    parser.set_defaults(func=cmd_workflow)
