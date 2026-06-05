"""CLI surface for SitioUno Software Factory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hermes_cli import factory_db as db


def _print_json(payload: Any) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    conn = db.ensure_db(args.db)
    conn.close()
    print(f"✓ Factory DB initialized: {args.db or db.default_db_path()}")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    agents = db.list_agents(db.ensure_db(args.db))
    if args.json:
        return _print_json({"agents": agents})
    for agent in agents:
        print(f"{agent['agent_id']}: {agent['role']} [{agent.get('preferred_engine') or 'zeus'}]")
    return 0


def cmd_project_create(args: argparse.Namespace) -> int:
    result = db.create_project(
        args.name,
        project_id=args.project_id,
        repo_path=args.repo_path,
        repo_remote=args.repo_remote,
        base_branch=args.base_branch,
        human_owner=args.human_owner,
        summary=args.summary,
        risk_level=args.risk_level,
        autonomy_level=args.autonomy_level,
        create_default_lanes=not args.no_default_lanes,
        metadata={"source": "hermes factory project create"},
    )
    if args.json:
        return _print_json(result)
    print(f"✓ Project {result['project_id']} ready")
    for lane in result.get("lanes", []):
        print(f"  - lane {lane['lane_id']} ({lane['methodology']}) board={lane['kanban_board']} branch={lane['branch']}")
    return 0


def cmd_lane_create(args: argparse.Namespace) -> int:
    result = db.create_lane(
        args.project_id,
        args.name,
        args.methodology,
        lane_id=args.lane_id,
        kanban_board=args.kanban_board,
        branch=args.branch,
        worktree_path=args.worktree_path,
    )
    if args.json:
        return _print_json(result)
    print(f"✓ Lane {result['lane_id']} ready ({result['methodology']})")
    print(f"  board={result['kanban_board']} branch={result['branch']}")
    return 0


def cmd_task_create(args: argparse.Namespace) -> int:
    result = db.create_task(
        args.project_id,
        args.title,
        lane_id=args.lane_id,
        description=args.description,
        phase=args.phase,
        owner_agent_id=args.owner,
        reviewer_agent_id=args.reviewer,
        engine=args.engine,
        priority=args.priority,
        acceptance_criteria=args.acceptance or [],
    )
    if args.json:
        return _print_json(result)
    print(f"✓ Task {result['task_id']} created")
    return 0


def cmd_gate_record(args: argparse.Namespace) -> int:
    result = db.record_gate(
        args.project_id,
        args.gate_type,
        args.status,
        lane_id=args.lane_id,
        task_id=args.task_id,
        reviewer=args.reviewer,
        notes=args.notes,
        evidence={"source": "hermes factory gate record"},
    )
    if args.json:
        return _print_json(result)
    print(f"✓ Gate {result['gate_id']} recorded: {args.gate_type}={args.status}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    payload = db.status(args.project_id, db.ensure_db(args.db))
    if args.json:
        return _print_json(payload)
    print(f"Factory DB: {payload['db_path']}")
    print(f"Projects: {len(payload['projects'])} | Lanes: {len(payload['lanes'])} | Tasks: {len(payload['tasks'])} | Gates: {len(payload['gates'])}")
    for project in payload["projects"]:
        print(f"- {project['project_id']}: {project['name']} [{project['status']}] risk={project['risk_level']}")
        for lane in [item for item in payload["lanes"] if item["project_id"] == project["project_id"]]:
            print(f"    lane {lane['lane_id']} {lane['methodology']} board={lane['kanban_board']}")
    return 0


def factory_command(args: argparse.Namespace) -> int:
    command = getattr(args, "factory_command", None)
    if command == "init":
        return cmd_init(args)
    if command == "agents":
        return cmd_agents(args)
    if command == "project" and getattr(args, "factory_project_command", None) == "create":
        return cmd_project_create(args)
    if command == "lane" and getattr(args, "factory_lane_command", None) == "create":
        return cmd_lane_create(args)
    if command == "task" and getattr(args, "factory_task_command", None) == "create":
        return cmd_task_create(args)
    if command == "gate" and getattr(args, "factory_gate_command", None) == "record":
        return cmd_gate_record(args)
    if command == "status":
        return cmd_status(args)
    return cmd_status(args)


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "factory",
        help="SitioUno Software Factory orchestration state",
        description="Manage factory projects, lanes, tasks, gates, agents, and progress DB.",
    )
    parser.add_argument("--db", help="Override local SQLite factory DB path (default: ~/.hermes/factory/factory.db)")
    parser.set_defaults(func=factory_command)
    subs = parser.add_subparsers(dest="factory_command")

    init = subs.add_parser("init", help="Initialize local factory DB and seed agent roster")
    init.set_defaults(func=factory_command)

    agents = subs.add_parser("agents", help="List seeded factory agents")
    agents.add_argument("--json", action="store_true", help="Emit JSON")
    agents.set_defaults(func=factory_command)

    status = subs.add_parser("status", help="Show factory status")
    status.add_argument("project_id", nargs="?", help="Optional project id filter")
    status.add_argument("--json", action="store_true", help="Emit JSON")
    status.set_defaults(func=factory_command)

    project = subs.add_parser("project", help="Manage factory projects")
    project_sub = project.add_subparsers(dest="factory_project_command")
    project_create = project_sub.add_parser("create", help="Create/update a project and default Zeus/BMAD lanes")
    project_create.add_argument("name")
    project_create.add_argument("--project-id")
    project_create.add_argument("--repo-path")
    project_create.add_argument("--repo-remote")
    project_create.add_argument("--base-branch")
    project_create.add_argument("--human-owner", default="Jean García")
    project_create.add_argument("--summary")
    project_create.add_argument("--risk-level", default="medium", choices=["low", "medium", "high", "critical"])
    project_create.add_argument("--autonomy-level", type=int, default=3)
    project_create.add_argument("--no-default-lanes", action="store_true")
    project_create.add_argument("--json", action="store_true")
    project_create.set_defaults(func=factory_command)

    lane = subs.add_parser("lane", help="Manage method lanes")
    lane_sub = lane.add_subparsers(dest="factory_lane_command")
    lane_create = lane_sub.add_parser("create", help="Create/update a lane")
    lane_create.add_argument("project_id")
    lane_create.add_argument("name")
    lane_create.add_argument("--methodology", required=True, choices=sorted(db.VALID_METHODS))
    lane_create.add_argument("--lane-id")
    lane_create.add_argument("--kanban-board")
    lane_create.add_argument("--branch")
    lane_create.add_argument("--worktree-path")
    lane_create.add_argument("--json", action="store_true")
    lane_create.set_defaults(func=factory_command)

    task = subs.add_parser("task", help="Manage factory tasks")
    task_sub = task.add_subparsers(dest="factory_task_command")
    task_create = task_sub.add_parser("create", help="Create a task in the progress DB")
    task_create.add_argument("project_id")
    task_create.add_argument("title")
    task_create.add_argument("--lane-id")
    task_create.add_argument("--description")
    task_create.add_argument("--phase", default="planning")
    task_create.add_argument("--owner")
    task_create.add_argument("--reviewer")
    task_create.add_argument("--engine", default="zeus")
    task_create.add_argument("--priority", type=int, default=100)
    task_create.add_argument("--acceptance", action="append", help="Acceptance criterion; repeatable")
    task_create.add_argument("--json", action="store_true")
    task_create.set_defaults(func=factory_command)

    gate = subs.add_parser("gate", help="Record factory gates")
    gate_sub = gate.add_subparsers(dest="factory_gate_command")
    gate_record = gate_sub.add_parser("record", help="Record a gate result")
    gate_record.add_argument("project_id")
    gate_record.add_argument("gate_type", choices=["intake", "functional", "architecture", "planning", "implementation", "spec", "quality", "test", "security", "delivery"])
    gate_record.add_argument("status", choices=["pending", "passed", "failed", "waived"])
    gate_record.add_argument("--lane-id")
    gate_record.add_argument("--task-id")
    gate_record.add_argument("--reviewer")
    gate_record.add_argument("--notes")
    gate_record.add_argument("--json", action="store_true")
    gate_record.set_defaults(func=factory_command)

    return parser
