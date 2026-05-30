"""CLI for Dev project goals — ``hermes dev goals …``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from gateway.dev_control.project_goals import (
    DevProjectGoalStore,
    abandon_project_goal,
    create_project_goal,
    get_project_goal_tree,
    list_project_goals,
)
from gateway.dev_control.project_scope import DEFAULT_PROJECT_ID, resolve_project_id
from hermes_constants import get_hermes_home


def _goal_store() -> DevProjectGoalStore:
    return DevProjectGoalStore(db_path=get_hermes_home() / "state.db")


def _emit(payload: Any, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _print_tree(nodes: list[dict[str, Any]], indent: int = 0) -> None:
    for node in nodes:
        prefix = "  " * indent
        progress = float(node.get("progress") or 0.0)
        print(
            f"{prefix}{node.get('kind', '?')}: {node.get('title', '')} "
            f"[{node.get('status', '?')}, {progress:.0%}] ({node.get('goal_id', '')})"
        )
        children = node.get("children") or []
        if children:
            _print_tree(children, indent + 1)


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    goals = parent_subparsers.add_parser(
        "goals",
        help="Project goal hierarchy (vision → goal → milestone → subgoal)",
        description="Manage durable Dev project goals and inspect rolled-up progress.",
    )
    goals_sub = goals.add_subparsers(dest="goals_command")

    create = goals_sub.add_parser("create", help="Create a project goal node")
    create.add_argument("kind", choices=("vision", "goal", "milestone", "subgoal"))
    create.add_argument("title")
    create.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    create.add_argument("--parent-goal-id", default=None)
    create.add_argument("--status", default="proposed")
    create.add_argument("--markdown", default="")
    create.add_argument("--plan-artifact-id", default=None)
    create.add_argument("--json", action="store_true")

    list_cmd = goals_sub.add_parser("list", help="List project goals")
    list_cmd.add_argument("--project-id", default=None)
    list_cmd.add_argument("--kind", default=None)
    list_cmd.add_argument("--status", default=None)
    list_cmd.add_argument("--parent-goal-id", default=None)
    list_cmd.add_argument("--include-abandoned", action="store_true")
    list_cmd.add_argument("--json", action="store_true")

    tree = goals_sub.add_parser("tree", help="Show the assembled goal tree")
    tree.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    tree.add_argument("--include-abandoned", action="store_true")
    tree.add_argument("--json", action="store_true")

    abandon = goals_sub.add_parser("abandon", help="Mark a goal abandoned")
    abandon.add_argument("goal_id")
    abandon.add_argument("--json", action="store_true")

    return goals


def dev_goals_command(args: argparse.Namespace) -> int:
    command = getattr(args, "goals_command", None)
    if not command:
        print("Usage: hermes dev goals {create,list,tree,abandon} …", file=sys.stderr)
        return 2

    store = _goal_store()
    try:
        if command == "create":
            result = create_project_goal(
                store=store,
                kind=args.kind,
                title=args.title,
                project_id=resolve_project_id(args.project_id),
                parent_goal_id=args.parent_goal_id,
                markdown=args.markdown,
                status=args.status,
                plan_artifact_id=args.plan_artifact_id,
            )
            return _emit(result, as_json=args.json)

        if command == "list":
            result = list_project_goals(
                store=store,
                project_id=args.project_id,
                kind=args.kind,
                status=args.status,
                parent_goal_id=args.parent_goal_id,
                include_abandoned=bool(args.include_abandoned),
            )
            if args.json:
                return _emit(result, as_json=True)
            for row in result.get("data", []):
                progress = float(row.get("progress") or 0.0)
                print(
                    f"{row.get('kind', '?'):9s}  {row.get('status', '?'):9s}  "
                    f"{progress:5.0%}  {row.get('title', '')}  ({row.get('goal_id', '')})"
                )
            return 0

        if command == "tree":
            result = get_project_goal_tree(
                store=store,
                project_id=resolve_project_id(args.project_id),
                include_abandoned=bool(args.include_abandoned),
            )
            if args.json:
                return _emit(result, as_json=True)
            _print_tree(result.get("roots") or [])
            return 0

        if command == "abandon":
            result = abandon_project_goal(store=store, goal_id=args.goal_id)
            return _emit(result, as_json=args.json)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(f"Unknown goals command: {command}", file=sys.stderr)
    return 2
