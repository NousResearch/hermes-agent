"""``hermes htr`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_htr_parser(subparsers, *, cmd_htr: Callable) -> None:
    """Attach the ``htr`` subcommand to ``subparsers``."""
    htr_parser = subparsers.add_parser(
        "htr",
        help="Hermes Trusted Task Runtime (HTR) tools",
        description="HTR observation, planning, and project registry tools",
    )
    htr_subparsers = htr_parser.add_subparsers(dest="htr_command", required=True)

    observe_parser = htr_subparsers.add_parser(
        "observe",
        help="Build a read-only observation snapshot for one run",
    )
    observe_parser.add_argument("run_id", help="Run identifier to observe")
    observe_parser.add_argument(
        "--project-id",
        default=None,
        help="Registered HTR project id (resolves the project's runs root)",
    )
    observe_parser.add_argument(
        "--runs-root",
        default=None,
        help="Override HTR runs root directory (default: HERMES_HOME/runs)",
    )
    observe_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise human summary to stderr (stdout remains JSON only)",
    )
    observe_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warning-level integrity findings as non-zero exit",
    )

    plan_parser = htr_subparsers.add_parser(
        "plan",
        help="Build a derived read-only action plan for one run",
    )
    plan_parser.add_argument("run_id", help="Run identifier to plan against")
    plan_parser.add_argument(
        "--action",
        default=None,
        help="Explicit Phase 1 lifecycle API name to plan (catalog allowlist only)",
    )
    plan_parser.add_argument(
        "--inputs-file",
        default=None,
        help="JSON file with record/actor/executor inputs for the selected action",
    )
    plan_parser.add_argument(
        "--project-checkpoint",
        default=None,
        help="Optional opaque project repository checkpoint string",
    )
    plan_parser.add_argument(
        "--remediation-intent",
        action="store_true",
        help="Explicit remediation-oriented planning intent (Policy C successor protocol)",
    )
    plan_parser.add_argument(
        "--project-id",
        default=None,
        help="Registered HTR project id (resolves the project's runs root)",
    )
    plan_parser.add_argument(
        "--runs-root",
        default=None,
        help=(
            "HTR runs-storage root for observation and for canonical project_dir "
            "on APIs that require it (same Phase 1 path contract as base_dir; "
            "proposal input only — does not mutate storage)"
        ),
    )
    plan_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise human summary to stderr (stdout remains JSON only)",
    )

    project_parser = htr_subparsers.add_parser(
        "project",
        help="Register and inspect isolated HTR projects",
    )
    project_subparsers = project_parser.add_subparsers(
        dest="htr_project_command",
        required=True,
    )

    register_parser = project_subparsers.add_parser(
        "register",
        help="Register a project identity bound to an existing runs-root directory",
    )
    register_parser.add_argument(
        "--runs-root",
        required=True,
        help="Absolute existing directory to bind as this project's runs root",
    )
    register_parser.add_argument(
        "--project-id",
        default=None,
        help="Optional project id (prj_YYYYMMDD_hex); generated when omitted",
    )
    register_parser.add_argument(
        "--display-name",
        default=None,
        help="Optional non-identity label (not unique, never used as identity)",
    )

    show_parser = project_subparsers.add_parser(
        "show",
        help="Show one registered project",
    )
    show_parser.add_argument("project_id", help="Project identifier")

    list_parser = project_subparsers.add_parser(
        "list",
        help="List registered projects",
    )
    list_parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived projects",
    )

    update_parser = project_subparsers.add_parser(
        "update",
        help="Update non-identity project metadata",
    )
    update_parser.add_argument("project_id", help="Project identifier")
    update_parser.add_argument(
        "--display-name",
        default=None,
        help="Set or clear the non-identity label (empty string clears)",
    )
    update_parser.add_argument(
        "--status",
        choices=("active", "archived"),
        default=None,
        help="Set project status (active or archived)",
    )
    update_parser.add_argument(
        "--clear-display-name",
        action="store_true",
        help="Clear the non-identity display name",
    )

    htr_parser.set_defaults(func=cmd_htr)
