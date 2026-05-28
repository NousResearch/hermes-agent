from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli.project_autopilot import (
    bootstrap_project_home,
    sync_project_home,
    verify_project_home,
)


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "project",
        help="Filesystem-backed Project Autopilot projects",
        description="Create, sync, and verify deterministic Project Autopilot homes.",
    )
    sub = parser.add_subparsers(dest="project_action")

    init = sub.add_parser("init", help="Bootstrap a Project Autopilot home")
    init.add_argument("--slug", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--goal", required=True)
    init.add_argument("--board", required=True, dest="board_slug")
    init.add_argument("--root-task", required=True, dest="root_task_id")
    init.add_argument("--project-home", required=True)
    init.add_argument("--repo-org", required=True)
    init.add_argument("--repo-name", required=True)
    init.add_argument("--canonical-repo", required=True)
    init.add_argument("--final-branch", required=True)
    init.add_argument("--source-plan")

    verify = sub.add_parser("verify", help="Verify project-home invariants")
    verify.add_argument("project_home")
    sync = sub.add_parser("sync", help="Sync a Project Autopilot home from kanban")
    sync.add_argument("project_home")
    sync.add_argument("--kanban-db")
    parser.set_defaults(func=project_command)
    return parser


def project_command(args: argparse.Namespace) -> int:
    action = getattr(args, "project_action", None)
    if action == "init":
        doc = bootstrap_project_home(
            slug=args.slug,
            title=args.title,
            goal=args.goal,
            board_slug=args.board_slug,
            root_task_id=args.root_task_id,
            project_home=Path(args.project_home).expanduser(),
            repo_org=args.repo_org,
            repo_name=args.repo_name,
            canonical_checkout=Path(args.canonical_repo).expanduser(),
            final_branch=args.final_branch,
            source_plan=Path(args.source_plan).expanduser()
            if args.source_plan
            else None,
        )
        print(f"BOOTSTRAPPED {doc['slug']} at {doc['project_home']}")
        return 0
    if action == "verify":
        doc = verify_project_home(Path(args.project_home).expanduser())
        print(f"OK {doc['slug']} state={doc['state']}")
        return 0
    if action == "sync":
        doc = sync_project_home(
            Path(args.project_home).expanduser(),
            db_path=Path(args.kanban_db).expanduser() if args.kanban_db else None,
        )
        print(f"SYNCED {doc['slug']} from board {doc['board_slug']}")
        return 0
    print("usage: hermes project <init|sync|verify>")
    return 1
