"""CLI helpers for max-iteration closure artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_closure_latest(args: argparse.Namespace) -> int:
    from hermes_cli.closure_artifacts import (
        build_resume_prompt_from_artifact,
        latest_closure_artifact,
    )

    data = latest_closure_artifact(
        session_id=getattr(args, "session_id", None),
        task_id=getattr(args, "task_id", None),
    )
    if data is None:
        print("No closure artifact found.", file=sys.stderr)
        return 1
    if getattr(args, "resume_prompt", False):
        print(build_resume_prompt_from_artifact(data))
        return 0
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    print(f"Closure artifact: {data.get('artifact_path')}")
    print(f"  session_id: {data.get('session_id') or '(none)'}")
    print(f"  task_id: {data.get('task_id') or '(none)'}")
    print(f"  status: {data.get('status')}")
    print(f"  last_completed_step: {data.get('last_completed_step') or '(none)'}")
    print(f"  active_session_lease_released: {data.get('active_session_lease_released')}")
    prompt = data.get("exact_resume_prompt")
    if prompt:
        print("  exact_resume_prompt:")
        print(str(prompt))
    return 0


def cmd_closure(args: argparse.Namespace) -> int:
    action = getattr(args, "closure_action", None) or "latest"
    if action == "resume":
        setattr(args, "resume_prompt", True)
        return _cmd_closure_latest(args)
    if action in {"latest", "show"}:
        return _cmd_closure_latest(args)
    print(f"Unknown closure action: {action}", file=sys.stderr)
    return 2


def _add_closure_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-id", default=None, help="Filter by session id")
    parser.add_argument("--task-id", default=None, help="Filter by Kanban task/card id")


def build_closure_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "closure",
        help="Show max-iteration closure artifacts and compact resume packets",
        description="Inspect max-iteration closure artifacts written when a run stops unfinished.",
    )
    closure_sub = parser.add_subparsers(dest="closure_action")
    latest = closure_sub.add_parser("latest", aliases=["show"], help="Show the latest closure artifact")
    _add_closure_filters(latest)
    latest.add_argument("--json", action="store_true", help="Print raw artifact JSON")
    latest.add_argument(
        "--resume-prompt",
        action="store_true",
        help="Print compact resume prompt instead of artifact summary",
    )
    latest.set_defaults(func=cmd_closure)
    resume = closure_sub.add_parser("resume", help="Print the compact resume packet for the latest artifact")
    _add_closure_filters(resume)
    resume.set_defaults(func=cmd_closure, resume_prompt=True, json=False)
    parser.set_defaults(func=cmd_closure)
    return parser
