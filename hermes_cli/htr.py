"""CLI for ``hermes htr …`` — read-only HTR observation and planning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from htr.action_plan import (
    EXIT_INVOCATION as PLAN_EXIT_INVOCATION,
    PlanningIntent,
    build_action_plan,
    compute_plan_exit_code,
    make_invocation_error,
)
from htr.observe import (
    EXIT_INVOCATION,
    ObserveInvocationError,
    build_run_snapshot,
    compute_exit_code,
)
from htr.project_registry import (
    ProjectRegistryError,
    project_registry_error_payload,
    register_project,
    get_project,
    list_projects,
    resolve_invocation_runs_root,
    update_project_metadata,
)


def _print_observe_summary(snapshot: dict[str, Any], *, stream: Any = None) -> None:
    out = stream if stream is not None else sys.stderr
    integrity = snapshot.get("integrity", {})
    phase1 = snapshot.get("phase1_chain", {})
    print(
        f"run {snapshot.get('run_id')}  "
        f"integrity={integrity.get('status')}  "
        f"errors={integrity.get('error_count', 0)}  "
        f"chain_complete={phase1.get('chain_complete')}  "
        f"terminal_reached={phase1.get('terminal_reached')}",
        file=out,
    )


def _print_plan_summary(plan: dict[str, Any], *, stream: Any = None) -> None:
    out = stream if stream is not None else sys.stderr
    print(
        f"run {plan.get('run_id')}  "
        f"state={plan.get('plan_state')}  "
        f"action={(plan.get('requested_intent') or {}).get('requested_action')}  "
        f"execution_eligible={(plan.get('automation_eligibility') or {}).get('execution_eligible')}",
        file=out,
    )


def _load_inputs_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read inputs file: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"inputs file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("inputs file must contain a JSON object")
    return data


def _print_registry_error(exc: BaseException) -> int:
    payload = project_registry_error_payload(exc)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return EXIT_INVOCATION


def _cli_runs_root(args) -> Path | None:
    return resolve_invocation_runs_root(
        project_id=getattr(args, "project_id", None),
        runs_root=getattr(args, "runs_root", None),
    )


def _handle_project(args) -> int:
    command = getattr(args, "htr_project_command", None)
    try:
        if command == "register":
            record = register_project(
                args.runs_root,
                project_id=getattr(args, "project_id", None),
                display_name=getattr(args, "display_name", None),
            )
            print(json.dumps({"ok": True, "project": record.to_dict()}, indent=2, ensure_ascii=False))
            return 0
        if command == "show":
            record = get_project(args.project_id)
            print(json.dumps({"ok": True, "project": record.to_dict()}, indent=2, ensure_ascii=False))
            return 0
        if command == "list":
            records = list_projects(include_archived=bool(getattr(args, "include_archived", False)))
            print(
                json.dumps(
                    {"ok": True, "projects": [item.to_dict() for item in records]},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if command == "update":
            display_name = getattr(args, "display_name", None)
            clear_name = bool(getattr(args, "clear_display_name", False))
            status = getattr(args, "status", None)
            if clear_name and display_name is not None:
                payload = {
                    "ok": False,
                    "error_class": "invalid_input",
                    "message": "--clear-display-name cannot be combined with --display-name",
                }
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                return EXIT_INVOCATION
            kwargs: dict[str, Any] = {}
            if clear_name:
                kwargs["display_name"] = None
            elif display_name is not None:
                kwargs["display_name"] = display_name
            if status is not None:
                kwargs["status"] = status
            record = update_project_metadata(args.project_id, **kwargs)
            print(json.dumps({"ok": True, "project": record.to_dict()}, indent=2, ensure_ascii=False))
            return 0
    except ProjectRegistryError as exc:
        return _print_registry_error(exc)

    print(f"unknown htr project subcommand: {command!r}", file=sys.stderr)
    return EXIT_INVOCATION


def _handle_plan(args) -> int:
    try:
        base_dir = _cli_runs_root(args)
    except ProjectRegistryError as exc:
        return _print_registry_error(exc)

    if getattr(args, "inputs_file", None) and not getattr(args, "action", None):
        error = make_invocation_error(
            "inputs_without_action",
            "--inputs-file requires --action",
        )
        print(json.dumps(error, indent=2, ensure_ascii=False))
        return PLAN_EXIT_INVOCATION

    action_inputs = None
    if getattr(args, "inputs_file", None):
        try:
            action_inputs = _load_inputs_file(Path(args.inputs_file))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            error = make_invocation_error("invalid_inputs_file", str(exc))
            print(json.dumps(error, indent=2, ensure_ascii=False))
            return PLAN_EXIT_INVOCATION

    try:
        snapshot = build_run_snapshot(args.run_id, base_dir=base_dir)
    except ObserveInvocationError as exc:
        print(str(exc), file=sys.stderr)
        error = make_invocation_error("observe_failed", str(exc))
        print(json.dumps(error, indent=2, ensure_ascii=False))
        return PLAN_EXIT_INVOCATION

    intent = PlanningIntent(
        requested_action=getattr(args, "action", None),
        action_inputs=action_inputs,
        project_repository_checkpoint=getattr(args, "project_checkpoint", None),
        htr_runs_root=str(base_dir) if base_dir is not None else None,
        remediation_oriented=bool(getattr(args, "remediation_intent", False)),
    )
    plan = build_action_plan(snapshot, intent)

    if args.summary:
        _print_plan_summary(plan)

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return compute_plan_exit_code(plan)


def htr_command(args) -> int:
    """Dispatch ``hermes htr`` subcommands."""
    if args.htr_command == "observe":
        try:
            base_dir = _cli_runs_root(args)
        except ProjectRegistryError as exc:
            return _print_registry_error(exc)
        try:
            snapshot = build_run_snapshot(args.run_id, base_dir=base_dir)
        except ObserveInvocationError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_INVOCATION

        if args.summary:
            _print_observe_summary(snapshot)

        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return compute_exit_code(snapshot, strict=bool(args.strict))

    if args.htr_command == "plan":
        return _handle_plan(args)

    if args.htr_command == "project":
        return _handle_project(args)

    print(f"unknown htr subcommand: {args.htr_command!r}", file=sys.stderr)
    return EXIT_INVOCATION
