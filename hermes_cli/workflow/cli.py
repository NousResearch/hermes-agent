from __future__ import annotations

import argparse
import json
import sys

from hermes_cli.workflow.orchestrator import WorkflowError, run_workflow


def add_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    workflow_parser = parent_subparsers.add_parser(
        "workflow",
        help="Run a local dynamic workflow orchestrator",
        description=(
            "Analyze a request, optionally split it into subtasks, route those "
            "subtasks to specialized workers, evaluate outputs, and synthesize "
            "a concise final response."
        ),
    )
    sub = workflow_parser.add_subparsers(dest="workflow_command")

    run_parser = sub.add_parser(
        "run",
        help="Run a workflow for a user task",
        description="Run the Dynamic Workflow MVP for a single user request.",
    )
    run_parser.add_argument(
        "task",
        nargs=argparse.REMAINDER,
        help='Task text. Example: hermes workflow run "plan this feature"',
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="Path to a workflow YAML config. Defaults to bundled MVP config.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic worker outputs without calling configured models.",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full workflow run as JSON.",
    )
    run_parser.add_argument(
        "--max-subtasks",
        type=int,
        default=None,
        help="Limit planner output to 1-8 subtasks.",
    )
    return workflow_parser


def workflow_command(args: argparse.Namespace) -> int:
    command = getattr(args, "workflow_command", None)
    if command != "run":
        print('Usage: hermes workflow run "user task here"', file=sys.stderr)
        return 2

    request = " ".join(getattr(args, "task", []) or []).strip()
    if not request:
        print("Error: workflow run requires a task string.", file=sys.stderr)
        return 2

    try:
        result = run_workflow(
            request,
            config_path=getattr(args, "config", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            max_subtasks=getattr(args, "max_subtasks", None),
        )
    except WorkflowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.final_response)
        print()
        print(f"Run id: {result.run_id}")
        print(f"Log: {result.log_path}")
    return 0
