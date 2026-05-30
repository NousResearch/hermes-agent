from __future__ import annotations

import argparse
import json
import sys

from hermes_cli.workflow.orchestrator import WorkflowConfig, WorkflowError, run_workflow
from hermes_cli.workflow.runtime import WorkflowRuntime
from hermes_cli.workflow.state import RunStore


def add_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    workflow_parser = parent_subparsers.add_parser(
        "workflow",
        help="Run a local dynamic workflow orchestrator",
        description=(
            "Analyze a request, split it into subtasks, route those subtasks "
            "to specialized workers, evaluate outputs, and synthesize a concise "
            "final response."
        ),
    )
    sub = workflow_parser.add_subparsers(dest="workflow_command")

    run_parser = sub.add_parser(
        "run",
        help="Run a workflow for a user task",
        description="Run a Dynamic Workflow request.",
    )
    _add_execution_flags(run_parser)
    run_parser.add_argument(
        "--resume-run-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--internal-background-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "task",
        nargs=argparse.REMAINDER,
        help='Task text. Example: hermes workflow run "plan this feature"',
    )

    script_parser = sub.add_parser(
        "script",
        help="Run a reusable Python workflow script",
        description="Run a Python-native Hermes workflow script.",
    )
    _add_execution_flags(script_parser)
    script_parser.add_argument("path", help="Path to a Python workflow script.")

    status_parser = sub.add_parser("status", help="Show workflow run status")
    _add_config_flag(status_parser)
    status_parser.add_argument("run_id")
    status_parser.add_argument("--json", action="store_true")

    logs_parser = sub.add_parser("logs", help="Show workflow run event logs")
    _add_config_flag(logs_parser)
    logs_parser.add_argument("run_id")
    logs_parser.add_argument("--json", action="store_true")
    logs_parser.add_argument("--tail", type=int, default=None)

    for command in ("pause", "cancel"):
        parser = sub.add_parser(command, help=f"{command.title()} a workflow run")
        _add_config_flag(parser)
        parser.add_argument("run_id")

    resume_parser = sub.add_parser("resume", help="Resume a paused workflow run")
    _add_config_flag(resume_parser)
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--json", action="store_true")
    return workflow_parser


def workflow_command(args: argparse.Namespace) -> int:
    command = getattr(args, "workflow_command", None)
    try:
        if command == "run":
            return _cmd_run(args)
        if command == "script":
            return _cmd_script(args)
        if command == "status":
            return _cmd_status(args)
        if command == "logs":
            return _cmd_logs(args)
        if command == "pause":
            return _cmd_mark(args, "paused")
        if command == "cancel":
            return _cmd_mark(args, "cancelled")
        if command == "resume":
            return _cmd_resume(args)
    except WorkflowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print('Usage: hermes workflow run "user task here"', file=sys.stderr)
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    resume_run_id = getattr(args, "resume_run_id", None)
    request = " ".join(getattr(args, "task", []) or []).strip()
    if not request and not resume_run_id:
        print("Error: workflow run requires a task string.", file=sys.stderr)
        return 2

    dry_run = bool(getattr(args, "dry_run", False))
    parallel = bool(getattr(args, "parallel", False))
    codex_plan = bool(getattr(args, "codex_plan", False))
    contest = bool(getattr(args, "contest", False))
    cross_review = bool(getattr(args, "cross_review", False))
    if resume_run_id:
        record = _store(args).get_run(resume_run_id)
        if record:
            dry_run = bool(record["dry_run"])
            parallel = "parallel" in str(record["mode"])
            codex_plan = bool(record["codex_plan"])
            contest = bool(record["contest"])
            cross_review = bool(record["cross_review"])

    result = run_workflow(
        request or "Resumed workflow run.",
        config_path=getattr(args, "config", None),
        dry_run=dry_run,
        max_subtasks=getattr(args, "max_subtasks", None),
        parallel=parallel,
        background=bool(getattr(args, "background", False)),
        codex_plan=codex_plan,
        contest=contest,
        cross_review=cross_review,
        run_id=resume_run_id,
        resume=bool(resume_run_id),
    )
    if getattr(args, "internal_background_worker", False):
        return 0
    _print_run(result, json_output=bool(getattr(args, "json", False)))
    return 0


def _cmd_script(args: argparse.Namespace) -> int:
    config = WorkflowConfig.load(getattr(args, "config", None))
    runtime = WorkflowRuntime(
        config=config,
        dry_run=bool(getattr(args, "dry_run", False)),
        max_subtasks=getattr(args, "max_subtasks", None),
        parallel=bool(getattr(args, "parallel", False)),
        codex_plan=bool(getattr(args, "codex_plan", False)),
        contest=bool(getattr(args, "contest", False)),
        cross_review=bool(getattr(args, "cross_review", False)),
    )
    result = runtime.run_script(
        getattr(args, "path"),
        background=bool(getattr(args, "background", False)),
    )
    _print_run(result, json_output=bool(getattr(args, "json", False)))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = _store(args)
    record = store.get_run(getattr(args, "run_id"))
    if not record:
        raise WorkflowError(f"Workflow run not found: {getattr(args, 'run_id')}")
    subtasks = store.load_subtasks(record["run_id"])
    outputs = store.load_outputs(record["run_id"])
    payload = {
        "run_id": record["run_id"],
        "status": record["status"],
        "mode": record["mode"],
        "dry_run": bool(record["dry_run"]),
        "codex_plan": bool(record["codex_plan"]),
        "contest": bool(record["contest"]),
        "cross_review": bool(record["cross_review"]),
        "pid": record["pid"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "subtasks": len(subtasks),
        "completed_outputs": len(outputs),
        "error": record["error"],
        "state_path": str(store.path),
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Run id: {payload['run_id']}")
        print(f"Status: {payload['status']}")
        print(f"Mode: {payload['mode']}")
        if payload["contest"]:
            print(f"Contest: enabled (cross-review={payload['cross_review']})")
        print(f"Progress: {payload['completed_outputs']}/{payload['subtasks']}")
        if payload["pid"]:
            print(f"PID: {payload['pid']}")
        if payload["error"]:
            print(f"Error: {payload['error']}")
        print(f"State: {payload['state_path']}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    store = _store(args)
    run_id = getattr(args, "run_id")
    if not store.get_run(run_id):
        raise WorkflowError(f"Workflow run not found: {run_id}")
    events = store.events(run_id, limit=getattr(args, "tail", None))
    if getattr(args, "json", False):
        print(json.dumps(events, indent=2, ensure_ascii=False))
    else:
        for event in events:
            print(f"{event['ts']} {event['event']}: {json.dumps(event['payload'], ensure_ascii=False)}")
    return 0


def _cmd_mark(args: argparse.Namespace, status: str) -> int:
    store = _store(args)
    run_id = getattr(args, "run_id")
    if not store.get_run(run_id):
        raise WorkflowError(f"Workflow run not found: {run_id}")
    store.update_run(run_id, status=status)
    store.append_event(run_id, "status", {"status": status})
    print(f"Workflow run {run_id} marked {status}.")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    store = _store(args)
    run_id = getattr(args, "run_id")
    record = store.get_run(run_id)
    if not record:
        raise WorkflowError(f"Workflow run not found: {run_id}")
    result = run_workflow(
        "Resumed workflow run.",
        config_path=getattr(args, "config", None),
        dry_run=bool(record["dry_run"]),
        parallel="parallel" in str(record["mode"]),
        codex_plan=bool(record["codex_plan"]),
        contest=bool(record["contest"]),
        cross_review=bool(record["cross_review"]),
        run_id=run_id,
        resume=True,
    )
    _print_run(result, json_output=bool(getattr(args, "json", False)))
    return 0


def _print_run(result, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.final_response)
        print()
        print(f"Run id: {result.run_id}")
        print(f"Status: {result.status}")
        print(f"Log: {result.log_path}")
        if result.state_path:
            print(f"State: {result.state_path}")


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a workflow YAML config. Defaults to bundled config.",
    )


def _add_execution_flags(parser: argparse.ArgumentParser) -> None:
    _add_config_flag(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic worker outputs without calling configured models.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--max-subtasks",
        type=int,
        default=None,
        help="Limit planner output to 1-8 subtasks.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run dependency-ready subtasks concurrently.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Queue the workflow in an explicit background worker process.",
    )
    parser.add_argument(
        "--codex-plan",
        action="store_true",
        help="Format the final response as a Codex Goal execution checklist.",
    )
    parser.add_argument(
        "--contest",
        action="store_true",
        help="Run multiple candidate workers for each subtask and select the best result.",
    )
    parser.add_argument(
        "--cross-review",
        action="store_true",
        help="Have reviewer/tester workers adversarially review contest candidates.",
    )


def _store(args: argparse.Namespace) -> RunStore:
    return RunStore.from_config(WorkflowConfig.load(getattr(args, "config", None)))
