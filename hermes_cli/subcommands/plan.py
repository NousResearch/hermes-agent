"""``hermes plan`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_plan_parser(subparsers, *, cmd_plan: Callable) -> None:
    parser = subparsers.add_parser(
        "plan",
        help="Compile architecture artifacts into a Hermes OS work graph",
        description="Read architecture artifacts or templates and emit an executable work graph.",
    )
    parser.add_argument("project")
    parser.add_argument("--projects-root")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--write", action="store_true", help="Write workgraph.json into the project")
    parser.add_argument("--generate-tasks", action="store_true", help="Generate TASKS.md and .hermes/tasks.json from the work graph")
    parser.add_argument(
        "--template",
        default="",
        help="Compile an external JSON/YAML template file instead of project documents",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist the work graph to the Hermes OS SQLite store",
    )
    parser.add_argument(
        "--db",
        default="",
        help="Optional SQLite database path for --persist",
    )
    parser.set_defaults(func=cmd_plan)
