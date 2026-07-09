"""CLI for the Hermes workflow graph engine."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.workflows_capabilities import require_implemented_primitives
from hermes_cli.workflows_spec import (
    WorkflowSpec,
    reject_unknown_spec_fields,
    validate_graph,
)


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
    p_deploy.add_argument(
        "--bump",
        action="store_true",
        help="On checksum conflict, redeploy as the next version instead of erroring",
    )
    p_deploy.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", help="List deployed workflow definitions")
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show a deployed workflow definition")
    p_show.add_argument("workflow_id")
    p_show.add_argument("--version", type=int, default=None, help="Version; latest when omitted")
    p_show.add_argument("--json", action="store_true")

    p_enable = sub.add_parser("enable", help="Enable a deployed workflow definition")
    p_enable.add_argument("workflow_id")
    p_enable.add_argument("--version", type=int, default=None, help="Version; latest when omitted")

    p_disable = sub.add_parser("disable", help="Disable a workflow (blocks new runs, removes schedules)")
    p_disable.add_argument("workflow_id")
    p_disable.add_argument("--version", type=int, default=None, help="Version; latest when omitted")

    p_run = sub.add_parser("run", help="Start a workflow execution")
    p_run.add_argument("workflow_id")
    p_run.add_argument("--input", default=None, help="JSON file path containing an object")
    p_run.add_argument("--json", action="store_true")

    p_exec = sub.add_parser("executions", help="List, show, or cancel workflow executions")
    exec_sub = p_exec.add_subparsers(dest="executions_action")
    p_exec_list = exec_sub.add_parser("list", help="List workflow executions (newest first)")
    p_exec_list.add_argument("--workflow", default=None, help="Restrict to a workflow id")
    p_exec_list.add_argument("--limit", type=int, default=20, help="Max executions to list (0 = all)")
    p_exec_list.add_argument("--json", action="store_true")
    p_exec_show = exec_sub.add_parser("show", help="Show one workflow execution")
    p_exec_show.add_argument("execution_id")
    p_exec_show.add_argument("--json", action="store_true")
    p_exec_runs = exec_sub.add_parser("node-runs", help="Show an execution's per-node runs")
    p_exec_runs.add_argument("execution_id")
    p_exec_runs.add_argument("--json", action="store_true")
    p_exec_events = exec_sub.add_parser("events", help="Show an execution's event timeline")
    p_exec_events.add_argument("execution_id")
    p_exec_events.add_argument("--json", action="store_true")
    p_exec_cancel = exec_sub.add_parser("cancel", help="Cancel a non-terminal execution")
    p_exec_cancel.add_argument("execution_id")
    p_exec.set_defaults(_workflow_executions_parser=p_exec)

    p_tick = sub.add_parser("tick", help="Advance queued cheap workflow executions")
    p_tick.add_argument("--limit", type=int, default=10)
    p_tick.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="Show dispatcher config and execution counts")
    p_status.add_argument("--json", action="store_true")

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
        if action == "enable":
            return _cmd_set_enabled(args, enabled=True)
        if action == "disable":
            return _cmd_set_enabled(args, enabled=False)
        if action == "run":
            return _cmd_run(args)
        if action == "executions":
            return _cmd_executions(args)
        if action == "tick":
            return _cmd_tick(args)
        if action == "status":
            return _cmd_status(args)
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
    reject_unknown_spec_fields(raw)
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)
    require_implemented_primitives(spec)
    return spec


def _dispatcher_settings() -> tuple[bool, float]:
    """Return (dispatch_in_gateway, tick_interval_seconds) from effective config."""
    from hermes_cli.config import load_config
    from utils import is_truthy_value

    try:
        cfg = load_config()
    except Exception:
        return True, 30.0
    workflow_cfg = cfg.get("workflow") if isinstance(cfg, dict) else {}
    if not isinstance(workflow_cfg, dict):
        workflow_cfg = {}
    enabled = is_truthy_value(workflow_cfg.get("dispatch_in_gateway"), default=True)
    try:
        interval = float(workflow_cfg.get("tick_interval_seconds", 30.0))
    except (TypeError, ValueError):
        interval = 30.0
    return enabled, interval


def _print_stall_hint(execution: wfdb.WorkflowExecution) -> None:
    if execution.status not in {"queued", "waiting"}:
        return
    dispatch_enabled, _interval = _dispatcher_settings()
    if dispatch_enabled:
        print(
            "Note: the gateway dispatcher advances this execution "
            "(workflow.dispatch_in_gateway is on). If no gateway is running, "
            "use `hermes workflow tick`."
        )
    else:
        print(
            "Warning: workflow.dispatch_in_gateway is off — nothing will advance "
            "this execution automatically. Run `hermes workflow tick` manually or "
            "set workflow.dispatch_in_gateway: true and restart the gateway."
        )


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
        deployed_version = wfdb.deploy_definition(
            conn, spec, created_by="cli", auto_bump=getattr(args, "bump", False)
        )
        record = wfdb.get_definition_record(conn, spec.id, deployed_version)
    if getattr(args, "json", False):
        _print_json(_definition_to_dict(record))
    else:
        suffix = f" (bumped from v{spec.version})" if deployed_version != spec.version else ""
        print(f"Deployed workflow {spec.id} v{deployed_version}{suffix}")
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
        record = wfdb.get_definition_record(conn, args.workflow_id, getattr(args, "version", None))
    payload = _definition_to_dict(record, include_spec=True)
    if getattr(args, "json", False):
        _print_json(payload)
    else:
        state = "enabled" if record.enabled else "disabled"
        print(f"{record.workflow_id} v{record.version} {state}")
        print(f"Name: {record.name}")
        print(f"Checksum: {record.checksum}")
    return 0


def _cmd_set_enabled(args: argparse.Namespace, *, enabled: bool) -> int:
    with _connect_initialized() as conn:
        record = wfdb.set_definition_enabled(
            conn, args.workflow_id, enabled, version=getattr(args, "version", None)
        )
    state = "enabled" if record.enabled else "disabled"
    print(f"Workflow {record.workflow_id} v{record.version} is now {state}.")
    if not enabled:
        print("New runs are blocked and schedule triggers are unregistered.")
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
    # Advance cheap nodes immediately so simple graphs finish inline and
    # agent_task nodes materialize their Kanban tasks without waiting for
    # the next dispatcher tick (same behavior as the dashboard Run button).
    try:
        workflows_dispatcher.tick(limit=1)
    except Exception as exc:
        print(f"Warning: initial tick failed: {exc}", file=sys.stderr)
    with _connect_initialized() as conn:
        execution = wfdb.get_execution(conn, execution_id)
    if getattr(args, "json", False):
        _print_json(_run_result(execution))
    else:
        print(f"Started execution {execution.execution_id} ({execution.status})")
        _print_stall_hint(execution)
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
    if action == "node-runs":
        return _cmd_executions_node_runs(args)
    if action == "events":
        return _cmd_executions_events(args)
    if action == "cancel":
        return _cmd_executions_cancel(args)
    print(f"Error: unknown workflow executions action {action!r}", file=sys.stderr)
    return 2


def _cmd_executions_list(args: argparse.Namespace) -> int:
    limit = getattr(args, "limit", 20)
    with _connect_initialized() as conn:
        executions = wfdb.list_executions(
            conn,
            getattr(args, "workflow", None),
            limit=limit if limit and limit > 0 else None,
        )
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
        if execution.status in {"queued", "waiting"}:
            _print_stall_hint(execution)
    return 0


def _format_epoch(value: Any) -> str:
    if not value:
        return "-"
    import datetime

    return datetime.datetime.fromtimestamp(int(value), datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ"
    )


def _cmd_executions_node_runs(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        node_runs = wfdb.list_node_runs(conn, args.execution_id)
    if getattr(args, "json", False):
        _print_json({"execution_id": args.execution_id, "node_runs": node_runs})
        return 0
    if not node_runs:
        print("(no node runs)")
        return 0
    for run in node_runs:
        line = f"{run['node_id']}  {run['status']}"
        if run.get("kanban_task_id"):
            line += f"  kanban={run['kanban_task_id']}"
        if run.get("wait_until"):
            line += f"  wait_until={_format_epoch(run['wait_until'])}"
        print(line)
        error = run.get("error")
        if error:
            print(f"  error: {json.dumps(error, sort_keys=True)}")
    return 0


def _cmd_executions_events(args: argparse.Namespace) -> int:
    with _connect_initialized() as conn:
        events = wfdb.list_events(conn, args.execution_id)
    if getattr(args, "json", False):
        _print_json({"execution_id": args.execution_id, "events": events})
        return 0
    if not events:
        print("(no events)")
        return 0
    for event in events:
        payload = event.get("payload") or {}
        line = f"{_format_epoch(event['created_at'])}  {event['kind']}"
        if payload:
            line += f"  {json.dumps(payload, sort_keys=True)}"
        print(line)
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


def _status_payload() -> dict[str, Any]:
    dispatch_enabled, interval = _dispatcher_settings()
    with _connect_initialized() as conn:
        counts = {
            row["status"]: row["n"]
            for row in conn.execute(
                "SELECT status, count(*) AS n FROM workflow_executions GROUP BY status"
            )
        }
        definitions = len(wfdb.list_definitions(conn))
    return {
        "dispatcher": {
            "dispatch_in_gateway": dispatch_enabled,
            "tick_interval_seconds": interval,
        },
        "definitions": definitions,
        "executions_by_status": counts,
    }


def _cmd_status(args: argparse.Namespace) -> int:
    payload = _status_payload()
    if getattr(args, "json", False):
        _print_json(payload)
        return 0
    dispatcher = payload["dispatcher"]
    state = "on" if dispatcher["dispatch_in_gateway"] else "OFF"
    print(f"Dispatcher: gateway dispatch {state} (interval {dispatcher['tick_interval_seconds']:.0f}s)")
    if not dispatcher["dispatch_in_gateway"]:
        print(
            "  Executions will not advance automatically. Run `hermes workflow tick` "
            "or set workflow.dispatch_in_gateway: true and restart the gateway."
        )
    print(f"Definitions: {payload['definitions']}")
    counts = payload["executions_by_status"]
    if counts:
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        print(f"Executions: {summary}")
    else:
        print("Executions: (none)")
    return 0


_SLASH_WORKFLOW_HELP = """/workflow — workflow graph engine
Usage:
  /workflow status                      dispatcher config + execution counts
  /workflow list                        deployed workflow definitions
  /workflow show <id> [--version N]     one definition
  /workflow enable|disable <id>         toggle a definition
  /workflow run <id>                    start a manual execution
  /workflow executions list [--workflow ID] [--limit N]
  /workflow executions show <execution-id>
  /workflow executions node-runs <execution-id>
  /workflow executions events <execution-id>
  /workflow executions cancel <execution-id>
  /workflow tick [--limit N]            advance queued executions
Authoring (validate/deploy) works with files: use `hermes workflow validate|deploy <file>`
or the dashboard /workflows tab."""


def run_slash(rest: str) -> str:
    """Execute a ``/workflow …`` string and return captured stdout/stderr.

    ``rest`` is everything after ``/workflow`` (may be empty). Shared by the
    interactive CLI and the gateway so formatting is identical.
    """
    tokens = shlex.split(rest) if rest and rest.strip() else []
    if not tokens or tokens[0] in {"help", "--help", "-h", "?"}:
        return _SLASH_WORKFLOW_HELP

    _wrap = argparse.ArgumentParser(prog="/workflow-wrap", add_help=False)
    _wrap.exit_on_error = False  # type: ignore[attr-defined]
    _top_sub = _wrap.add_subparsers(dest="_top")
    workflow_parser = build_parser(_top_sub)
    workflow_parser.prog = "/workflow"
    workflow_parser.exit_on_error = False  # type: ignore[attr-defined]
    for _action in workflow_parser._actions:
        if isinstance(_action, argparse._SubParsersAction):
            for _name, _choice in _action.choices.items():
                _choice.prog = f"/workflow {_name}"
                _choice.exit_on_error = False  # type: ignore[attr-defined]

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            args = workflow_parser.parse_args(tokens)
    except SystemExit as exc:
        out = buf_out.getvalue().rstrip()
        err = buf_err.getvalue().rstrip()
        if exc.code in {0, None} and out:
            return out
        body = err or out
        return f"⚠ /workflow usage error\n{body}" if body else "⚠ /workflow usage error"
    except argparse.ArgumentError as exc:
        return f"⚠ /workflow usage error\n{exc}"

    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            workflow_command(args)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
    out = buf_out.getvalue().rstrip()
    err = buf_err.getvalue().rstrip()
    combined = "\n".join(part for part in (out, err) if part)
    return combined or "(no output)"
