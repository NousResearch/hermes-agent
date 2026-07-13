"""Hermes OS project runtime subcommand parsers."""

from __future__ import annotations

from typing import Callable


def build_project_runtime_parsers(subparsers, *, cmd_project_runtime: Callable) -> None:
    workspace = subparsers.add_parser("workspace", help="Summarize Hermes OS workspace control-plane status")
    workspace.add_argument("--projects-root", required=False, default=".")
    workspace.add_argument("--json", action="store_true")
    workspace.set_defaults(func=cmd_project_runtime, project_runtime_command="workspace")

    projects = subparsers.add_parser("projects", help="List Hermes OS registered projects")
    projects.add_argument("--workspace-root", default=".")
    projects.set_defaults(func=cmd_project_runtime, project_runtime_command="projects")

    switch = subparsers.add_parser("switch", help="Dry-run switch to a Hermes OS project runtime")
    switch.add_argument("project")
    switch.add_argument("--workspace-root", default=".")
    switch.add_argument("--live", action="store_true", help="Perform live restore/start actions where supported")
    switch.set_defaults(func=cmd_project_runtime, project_runtime_command="switch")

    start = subparsers.add_parser("start", help="Dry-run start project runtime services")
    start.add_argument("project")
    start.add_argument("--workspace-root", default=".")
    start.add_argument("--live", action="store_true", help="Start services instead of dry-run planning")
    start.set_defaults(func=cmd_project_runtime, project_runtime_command="start")

    snapshot = subparsers.add_parser("snapshot", help="Save or restore Hermes OS project workspace snapshots")
    snapshot.add_argument("snapshot_action", choices=["save", "restore"])
    snapshot.add_argument("project")
    snapshot.add_argument("--workspace-root", default=".")
    snapshot.add_argument("--live", action="store_true", help="Perform live restore actions where supported")
    snapshot.set_defaults(func=cmd_project_runtime, project_runtime_command="snapshot")
