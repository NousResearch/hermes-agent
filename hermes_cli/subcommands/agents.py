"""``hermes agents`` subcommand parser.

Builds the argparse tree for the agents dashboard command. The handler
functions live in ``hermes_cli.main`` (injected) so this module stays
import-light and mirrors the pattern of ``hermes_cli/subcommands/dashboard.py``.
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_agents_parser(
    subparsers,
    *,
    cmd_agents: Callable,
    cmd_agents_list: Callable,
    cmd_agents_init: Callable,
    cmd_agents_orchestrate: Callable,
    cmd_agents_dashboard: Callable,
) -> None:
    """Attach the ``agents`` subcommand tree."""
    # =========================================================================
    # agents command
    # =========================================================================
    agents_parser = subparsers.add_parser(
        "agents",
        help="Show the AI team dashboard (agents, profiles, models)",
        description=(
            "Display a manager-friendly dashboard of Hermes AI agents: "
            "profile, model, provider, role and status. Use subcommands "
            "to list as JSON, create a starter registry, or send a task "
            "to orch for planning."
        ),
    )
    agents_parser.set_defaults(func=cmd_agents)

    agents_subparsers = agents_parser.add_subparsers(dest="agents_action")

    # agents list [--json]
    list_parser = agents_subparsers.add_parser(
        "list",
        help="List agents as a table (default) or JSON",
        description="Print the agent team as a table, or as JSON with --json.",
    )
    list_parser.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Output normalized JSON instead of a table",
    )
    list_parser.set_defaults(func=cmd_agents_list)

    # agents init [--force]
    init_parser = agents_subparsers.add_parser(
        "init",
        help="Create a starter agents.yaml registry",
        description=(
            "Write a starter agent registry to $HERMES_HOME/agents.yaml. "
            "Refuses to overwrite an existing file unless --force is passed."
        ),
    )
    init_parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Overwrite an existing agents.yaml",
    )
    init_parser.set_defaults(func=cmd_agents_init)

    # agents orchestrate "<task>"
    orchestrate_parser = agents_subparsers.add_parser(
        "orchestrate",
        help="Send a task to orch for planning (creates a Kanban card)",
        description=(
            "Prepare a planning handoff for the orch profile. Creates a "
            "Kanban card assigned to orch when the backend is available, "
            "and prints the exact next command to start processing. "
            "Does NOT execute the task itself."
        ),
    )
    orchestrate_parser.add_argument(
        "task",
        help='The task text to hand to orch, e.g. "Melhorar upload do app".',
    )
    orchestrate_parser.add_argument(
        "--no-create",
        dest="no_create",
        action="store_true",
        help="Do not create a Kanban card; only print the handoff command.",
    )
    orchestrate_parser.set_defaults(func=cmd_agents_orchestrate)

    # agents dashboard
    dashboard_parser = agents_subparsers.add_parser(
        "dashboard",
        help="Open a navigable terminal dashboard of the AI team",
        description=(
            "Display a navigable terminal dashboard of Hermes AI agents. "
            "Shows profile, model, provider, role, status, edit permissions, "
            "escalation and skills. In a TTY, supports numeric selection to "
            "inspect individual agent details. In non-interactive mode, "
            "prints a table and exits. Uses rich when available, falls back "
            "to plain text otherwise."
        ),
    )
    dashboard_parser.add_argument(
        "--no-detail",
        dest="no_detail",
        action="store_true",
        help="Non-interactive mode: print table without prompting for selection.",
    )
    dashboard_parser.set_defaults(func=cmd_agents_dashboard)