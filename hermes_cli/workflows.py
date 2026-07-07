"""CLI for the Hermes workflow graph engine."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.workflows_capabilities import require_implemented_primitives
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "workflow",
        help="Manage workflow graph definitions and executions",
        description="Validate, deploy, run, and inspect Hermes workflow graphs.",
    )
    sub = parser.add_subparsers(dest="workflow_action")

    sub.add_parser("init", help="Create workflows.db if missing")

    p_validate = sub.add_parser("validate", help="Validate a workflow YAML file")
    p_validate.add_argument("file")

    p_deploy = sub.add_parser("deploy", help="Validate and deploy a workflow YAML file")
    p_deploy.add_argument("file")
    p_deploy.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", help="List deployed workflow definitions")
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show a deployed workflow definition")
    p_show.add_argument("workflow_id")
    p_show.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="Start a workflow execution")
    p_run.add_argument("workflow_id")
    p_run.add_argument("--input", default=None, help="JSON file path containing an object")
    p_run.add_argument("--json", action="store_true")

    p_exec = sub.add_parser("executions", help="List, show, or cancel workflow executions")
    exec_sub = p_exec.add_subparsers(dest="executions_action")
    p_exec_list = exec_sub.add_parser("list", help="List workflow executions")
    p_exec_list.add_argument("--workflow", default=None, help="Restrict to a workflow id")
    p_exec_list.add_argument("--json", action="store_true")
    p_exec_show = exec_sub.add_parser("show", help="Show one workflow execution")
    p_exec_show.add_argument("execution_id")
    p_exec_show.add_argument("--json", action="store_true")
    p_exec_cancel = exec_sub.add_parser("cancel", help="Cancel a non-terminal execution")
    p_exec_cancel.add_argument("execution_id")
    p_exec.set_defaults(_workflow_executions_parser=p_exec)

    p_tick = sub.add_parser("tick", help="Advance queued cheap workflow executions")
    p_tick.add_argument("--limit", type=int, default=10)
    p_tick.add_argument("--json", action="store_true")

    parser.set_defaults(_workflow_parser=parser)
    return parser


def workflow_command(args: argparse.Namespace) -> int:
    action = getattr(args, "workflow_action", None)
    if not action:
        parser = getattr(args, "_workflow_parser", None)
        if parser is not None:
            parser.print_help()
        return 0

    try:
        if action == "init":
            return _cmd_init(args)
        if action == "validate":
            return _cmd_validate(args)
        if action == "deploy":
            return _cmd_deploy(args)
        if action == "list":
            return _cmd_list(args)
        if action == "show":
            return _cmd_show(args)
        if action == "run":
            return _cmd_run(args)
        if action == "executions":
            return _cmd_executions(args)
        if action == "tick":
            return _cmd_tick(args)
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError, KeyError) as exc:
        print(f"Error: {_error_text(exc)}", file=sys.stderr)
        return 1

    print(f"Error: unknown workflow action {action!r}", file=sys.stderr)
    return 2


@contextlib.contextmanager
def _connect_initialized():
    wfdb.init_db()
    with contextlib.closing(wfdb.connect()) as conn:
        yield conn


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    return str(exc)


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True))


def _load_spec(path: str) -> WorkflowSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workflow file must contain a YAML object")
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)
    require_implemented_primitives(spec)
    return spec


def _definition_to_dict(record: wfdb.WorkflowDefinitionRecord, *, include_spec: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checksum": record.checksum,
        "created_at": record.created_at,
        "created_by": record.created_by,
        "enabled": record.enabled,
        "id": record.workflow_id,
        "name": record.name,
        "version": record.version,
        "workflow_id": record.workflow_id,
    }
    if include_spec:
        payload["spec"] = record.spec.model_dump(mode="json", by_alias=True)
    return payload


def _latest_definition_record(conn, workflow_id: str) -> wfdb.WorkflowDefinitionRecord:
    return wfdb.get_definition_record(conn, workflow_id)


def _execution_to_dict(execution: wfdb.WorkflowExecution) -> dict[str, Any]:
    return {
        "context": execution.context,
        "created_at": execution.created_at,
        "execution_id": execution.execution_id,
        "input": execution.input,
        "status": execution.status,
        "trigger_id": execution.trigger_id,
        "trigger_type": execution.trigger_type,
        "updated_at": execution.updated_at,
        "version": execution.version,
        "workflow_id": execution.workflow_id,
    }


def _run_result(execution: wfdb.WorkflowExecution) -> dict[str, Any]:
    return {
        "execution_id": execution.execution_id,
        "status": execution.status,
        "version": execution.version,
        "workflow_id": execution.workflow_id,
    }


def _read_input(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--input JSON must be an object")
    return data


def _list_executions(conn, workflow_id: str | None = None) -> list[wfdb.WorkflowExecution]:
    if workflow_id:
        rows = conn.execute(
            """
            SELECT execution_id FROM workflow_executions
             WHERE workflow_id = ?
             ORDER BY created_at, execution_id
            """,
            (workflow_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT execution_id FROM workflow_executions
             ORDER BY created_at, execution_id
            """
        ).fetchall()
    return [wfdb.get_execution(conn, row["execution_id"]) for row in rows]


def _cmd_init(_args: argparse.Namespace) -> int:
    wfdb.init_db()
    print("Initialized workflows DB.")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.file)
    print(f"OK: {spec.id} v{spec.version}")
    return 0


def _cmd_deploy(args: argparse.Namespace) -> int:
    spec = _load_spec(args.file)
    with _connect_initialized() as conn:
        wfdb.deploy_definition(conn, spec, created_by="cli")
        record = wfdb.get_definition_record(conn, spec.id, spec.version)
    if getattr(args, "json", False):
        _print_json(_definition_to_dict(record))
    else:
        print(f"Deployed workflow {spec.id} v{spec.version}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        records = wfdb.list_definitions(conn)
    if getattr(args, "json", False):
        _print_json([_definition_to_dict(record) for record in records])
        return 0
    if not records:
        print("(no workflows deployed)")
        return 0
    for record in records:
        state = "enabled" if record.enabled else "disabled"
        print(f"{record.workflow_id} v{record.version} {state}  {record.name}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        record = _latest_definition_record(conn, args.workflow_id)
    payload = _definition_to_dict(record, include_spec=True)
    if getattr(args, "json", False):
        _print_json(payload)
    else:
        state = "enabled" if record.enabled else "disabled"
        print(f"{record.workflow_id} v{record.version} {state}")
        print(f"Name: {record.name}")
        print(f"Checksum: {record.checksum}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    input_data = _read_input(getattr(args, "input", None))
    with _connect_initialized() as conn:
        execution_id = wfdb.start_execution(
            conn,
            args.workflow_id,
            input_data=input_data,
            trigger_type="manual",
        )
        execution = wfdb.get_execution(conn, execution_id)
    if getattr(args, "json", False):
        _print_json(_run_result(execution))
    else:
        print(f"Started execution {execution.execution_id} ({execution.status})")
    return 0


def _cmd_executions(args: argparse.Namespace) -> int:
    action = getattr(args, "executions_action", None)
    if not action:
        parser = getattr(args, "_workflow_executions_parser", None)
        if parser is not None:
            parser.print_help()
        return 0
    if action == "list":
        return _cmd_executions_list(args)
    if action == "show":
        return _cmd_executions_show(args)
    if action == "cancel":
        return _cmd_executions_cancel(args)
    print(f"Error: unknown workflow executions action {action!r}", file=sys.stderr)
    return 2


def _cmd_executions_list(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        executions = _list_executions(conn, getattr(args, "workflow", None))
    if getattr(args, "json", False):
        _print_json([_execution_to_dict(execution) for execution in executions])
        return 0
    if not executions:
        print("(no workflow executions)")
        return 0
    for execution in executions:
        print(
            f"{execution.execution_id} {execution.status}  "
            f"{execution.workflow_id} v{execution.version}"
        )
    return 0


def _cmd_executions_show(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        execution = wfdb.get_execution(conn, args.execution_id)
    if getattr(args, "json", False):
        _print_json(_execution_to_dict(execution))
    else:
        print(
            f"{execution.execution_id} {execution.status}  "
            f"{execution.workflow_id} v{execution.version}"
        )
    return 0


def _cmd_executions_cancel(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        execution, cancelled = wfdb.cancel_execution(conn, args.execution_id, source="cli")
        if not cancelled:
            print(f"Execution {execution.execution_id} already {execution.status}.")
            return 0
    print(f"Cancelled execution {args.execution_id}")
    return 0


def _cmd_tick(args: argparse.Namespace) -> int:
    wfdb.init_db()
    processed = workflows_dispatcher.tick(limit=args.limit)
    if getattr(args, "json", False):
        _print_json({"processed": processed})
    else:
        print(f"Processed {processed} workflow execution(s).")
    return 0
