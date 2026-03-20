#!/usr/bin/env python3
"""CLI helper for the AutoResearch optional skill."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from autoresearch import runtime


def _run_local_command(command: str, workdir: Path, task_id: str | None) -> dict[str, Any]:
    """Run manifest-defined commands directly for skill-driven workflows."""
    del task_id

    completed = subprocess.run(
        command,
        cwd=str(workdir),
        shell=True,
        capture_output=True,
        text=True,
    )
    output_parts = [part for part in (completed.stdout, completed.stderr) if part]
    return {
        "exit_code": completed.returncode,
        "output": "\n".join(output_parts),
        "error": completed.stderr if completed.returncode != 0 else "",
    }


@contextmanager
def _with_local_command_runner():
    original = runtime._run_command
    runtime._run_command = _run_local_command
    try:
        yield
    finally:
        runtime._run_command = original


def run_action(action: str, **kwargs) -> dict[str, Any]:
    action = action.strip().lower().replace("-", "_")
    project_root = kwargs.get("project_root")

    try:
        if action == "list_projects":
            payload = runtime.list_projects(project_root)
        elif action == "inspect_project":
            payload = runtime.inspect_project(project_root)
        elif action == "validate_project":
            payload = runtime.validate_project(project_root)
        elif action == "research_cycle":
            family_id = kwargs.get("family_id")
            if not family_id:
                raise ValueError("family_id is required for action='research_cycle'")
            with _with_local_command_runner():
                payload = runtime.research_cycle(
                    project_root=project_root,
                    family_id=family_id,
                    population=kwargs.get("population"),
                    survivors=kwargs.get("survivors"),
                    seed=kwargs.get("seed", 7),
                    model=kwargs.get("model"),
                    task_id=None,
                )
        elif action == "status":
            run_id = kwargs.get("run_id")
            if not run_id:
                raise ValueError("run_id is required for action='status'")
            payload = runtime.status(run_id=run_id, project_root=project_root)
        elif action == "list_runs":
            payload = runtime.list_runs(project_root=project_root, limit=int(kwargs.get("limit") or 20))
        elif action == "inspect_run":
            run_id = kwargs.get("run_id")
            if not run_id:
                raise ValueError("run_id is required for action='inspect_run'")
            payload = runtime.inspect_run(run_id=run_id, project_root=project_root)
        elif action == "publish_summary":
            run_id = kwargs.get("run_id")
            if not run_id:
                raise ValueError("run_id is required for action='publish_summary'")
            payload = runtime.publish_summary(
                run_id=run_id,
                project_root=project_root,
                target=kwargs.get("target"),
                send=bool(kwargs.get("send", False)),
            )
        else:
            raise ValueError(f"Unknown autoresearch action '{action}'")
        return {"success": True, **payload}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoResearch helper for the Hermes optional skill.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    def add_project_root_arg(target: argparse.ArgumentParser) -> None:
        target.add_argument("--project-root", dest="project_root")

    add_project_root_arg(subparsers.add_parser("list-projects", help="List discoverable AutoResearch projects."))
    add_project_root_arg(subparsers.add_parser("inspect-project", help="Inspect the current AutoResearch project."))
    add_project_root_arg(subparsers.add_parser("validate-project", help="Validate the current AutoResearch manifests."))

    research_cycle = subparsers.add_parser("research-cycle", help="Run a bounded AutoResearch cycle.")
    add_project_root_arg(research_cycle)
    research_cycle.add_argument("--family-id", required=True)
    research_cycle.add_argument("--population", type=int)
    research_cycle.add_argument("--survivors", type=int)
    research_cycle.add_argument("--seed", type=int, default=7)
    research_cycle.add_argument("--model")

    status = subparsers.add_parser("status", help="Show compact run status.")
    add_project_root_arg(status)
    status.add_argument("--run-id", required=True)

    list_runs = subparsers.add_parser("list-runs", help="List recent AutoResearch runs.")
    add_project_root_arg(list_runs)
    list_runs.add_argument("--limit", type=int, default=20)

    inspect_run = subparsers.add_parser("inspect-run", help="Inspect one AutoResearch run.")
    add_project_root_arg(inspect_run)
    inspect_run.add_argument("--run-id", required=True)

    publish_summary = subparsers.add_parser("publish-summary", help="Build or send a run summary.")
    add_project_root_arg(publish_summary)
    publish_summary.add_argument("--run-id", required=True)
    publish_summary.add_argument("--target")
    publish_summary.add_argument("--send", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    action = args.pop("action").replace("-", "_")
    payload = run_action(action, **args)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
