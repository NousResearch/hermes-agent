"""Workflows dashboard plugin — backend API routes."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph

router = APIRouter()
logger = logging.getLogger(__name__)
_TERMINAL_STATUSES = {"cancelled", "failed", "succeeded"}


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


def _http_400(exc: BaseException) -> HTTPException:
    return HTTPException(status_code=400, detail=_error_text(exc))


def _http_404(exc: BaseException) -> HTTPException:
    return HTTPException(status_code=404, detail=_error_text(exc))


async def _read_body(request: Request) -> Any:
    raw = await request.body()
    if not raw or not raw.strip():
        return {}
    text = raw.decode("utf-8")
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        return json.loads(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        loaded = yaml.safe_load(text)
        return {} if loaded is None else loaded


def _yaml_or_object(value: Any, *, what: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = yaml.safe_load(value)
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be an object")
    return value


def _load_spec_from_payload(payload: Any) -> WorkflowSpec:
    if isinstance(payload, dict) and "spec" in payload:
        payload = payload["spec"]
    raw = _yaml_or_object(payload, what="workflow spec")
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)
    return spec


def _spec_dict(spec: WorkflowSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json", by_alias=True)


def _definition_preview(spec: WorkflowSpec) -> dict[str, Any]:
    return {
        "enabled": spec.enabled,
        "id": spec.id,
        "name": spec.name,
        "spec": _spec_dict(spec),
        "version": spec.version,
        "workflow_id": spec.id,
    }


def _definition_record(conn, workflow_id: str, version: int | None = None):
    records = [r for r in wfdb.list_definitions(conn) if r.workflow_id == workflow_id]
    if version is not None:
        records = [r for r in records if r.version == version]
    if not records:
        if version is None:
            raise KeyError(f"workflow definition not found: {workflow_id}")
        raise KeyError(f"workflow definition not found: {workflow_id} v{version}")
    return max(records, key=lambda r: r.version)


def _definition_to_dict(record, *, include_spec: bool = False) -> dict[str, Any]:
    payload = {
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
        payload["spec"] = _spec_dict(record.spec)
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


def _event_to_dict(row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    return {
        "created_at": row["created_at"],
        "execution_id": row["execution_id"],
        "id": row["id"],
        "kind": row["kind"],
        "node_run_id": row["node_run_id"],
        "payload": payload,
    }


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


def _input_from_payload(payload: Any) -> dict[str, Any]:
    if payload in (None, ""):
        return {}
    payload = _yaml_or_object(payload, what="run request")
    if "input_json" in payload:
        raw_input = payload["input_json"]
        if raw_input is None or raw_input == "":
            raw_input = "{}"
        if not isinstance(raw_input, str):
            raise ValueError("input_json must be a JSON object string")
        input_data = json.loads(raw_input)
    else:
        input_data = payload.get("input", {})
    if not isinstance(input_data, dict):
        raise ValueError("workflow input must be a JSON object")
    return input_data


@router.get("/definitions")
async def list_definitions() -> dict[str, Any]:
    with _connect_initialized() as conn:
        definitions = [_definition_to_dict(r) for r in wfdb.list_definitions(conn)]
    return {"definitions": definitions}


@router.post("/definitions/validate")
async def validate_definition(request: Request) -> dict[str, Any]:
    try:
        spec = _load_spec_from_payload(await _read_body(request))
    except (json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise _http_400(exc) from exc
    return {"valid": True, "definition": _definition_preview(spec)}


@router.post("/definitions/deploy")
async def deploy_definition(request: Request) -> dict[str, Any]:
    try:
        spec = _load_spec_from_payload(await _read_body(request))
        with _connect_initialized() as conn:
            wfdb.deploy_definition(conn, spec, created_by="dashboard")
            record = _definition_record(conn, spec.id)
    except (json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise _http_400(exc) from exc
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"definition": _definition_to_dict(record, include_spec=True)}


@router.get("/definitions/{workflow_id}")
async def get_definition(
    workflow_id: str,
    version: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            record = _definition_record(conn, workflow_id, version)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"definition": _definition_to_dict(record, include_spec=True)}


@router.post("/definitions/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    request: Request,
    version: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        input_data = _input_from_payload(await _read_body(request))
        with _connect_initialized() as conn:
            execution_id = wfdb.start_execution(
                conn,
                workflow_id,
                input_data=input_data,
                trigger_type="manual",
                version=version,
            )
            try:
                workflows_dispatcher.tick(limit=1)
            except Exception:
                logger.debug("workflow dashboard tick failed", exc_info=True)
            execution = wfdb.get_execution(conn, execution_id)
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        raise _http_400(exc) from exc
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"execution": _execution_to_dict(execution)}


@router.get("/executions")
async def list_executions(workflow_id: str | None = Query(default=None)) -> dict[str, Any]:
    with _connect_initialized() as conn:
        executions = [_execution_to_dict(e) for e in _list_executions(conn, workflow_id)]
    return {"executions": executions}


@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            execution = wfdb.get_execution(conn, execution_id)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"execution": _execution_to_dict(execution)}


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            execution = wfdb.get_execution(conn, execution_id)
            cancelled = False
            if execution.status not in _TERMINAL_STATUSES:
                now = int(time.time())
                terminal_statuses = tuple(sorted(_TERMINAL_STATUSES))
                placeholders = ", ".join("?" for _ in terminal_statuses)
                with wfdb.write_txn(conn):
                    updated = conn.execute(
                        f"""
                        UPDATE workflow_executions
                           SET status = 'cancelled', claim_lock = NULL,
                               claim_expires = NULL, updated_at = ?
                         WHERE execution_id = ?
                           AND status NOT IN ({placeholders})
                        """,
                        (now, execution_id, *terminal_statuses),
                    ).rowcount
                    cancelled = updated > 0
                    if cancelled:
                        wfdb.append_event(
                            conn,
                            execution_id,
                            "execution_cancelled",
                            {"source": "dashboard"},
                        )
            execution = wfdb.get_execution(conn, execution_id)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"cancelled": cancelled, "execution": _execution_to_dict(execution)}


@router.get("/executions/{execution_id}/events")
async def list_events(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            wfdb.get_execution(conn, execution_id)
            rows = conn.execute(
                """
                SELECT id, execution_id, node_run_id, kind, payload_json, created_at
                  FROM workflow_events
                 WHERE execution_id = ?
                 ORDER BY id
                """,
                (execution_id,),
            ).fetchall()
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"events": [_event_to_dict(row) for row in rows]}
