"""Workflows dashboard plugin — backend API routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from hermes_cli import inventory as inventory_mod
from hermes_cli import profiles as profiles_mod
from hermes_cli import workflows_assistant
from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.config import load_config
from hermes_cli.workflows_capabilities import (
    require_implemented_primitives,
    workflow_capabilities,
)
from hermes_cli.workflows_redaction import redact_sensitive
from hermes_cli.workflows_spec import (
    WorkflowSpec,
    reject_unknown_spec_fields,
    validate_graph,
)
from utils import is_truthy_value

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_WORKFLOW_REQUEST_BYTES = 1_000_000


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


def _http_500(message: str = "workflow assistant failed") -> HTTPException:
    return HTTPException(status_code=500, detail=message)


def _http_413() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail={
            "code": "workflow_request_too_large",
            "message": f"Workflow request body must be <= {MAX_WORKFLOW_REQUEST_BYTES} bytes.",
        },
    )


def _assistant_validation_http() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": "workflow_assistant_validation_error",
            "message": "Workflow assistant returned a draft that failed validation.",
            "hint": "Revise the request, use a template, or switch to Advanced YAML.",
        },
    )


def _assistant_runtime_http(detail: str | None = None) -> HTTPException:
    message = "Workflow assistant failed before returning a valid draft."
    hint = "Check workflow assistant provider/model configuration, then retry or use Advanced YAML."
    if detail and "secret" not in detail.lower() and "token" not in detail.lower() and "api_key" not in detail.lower() and "password" not in detail.lower():
        message = message + " (" + detail + ")"
    return HTTPException(
        status_code=502,
        detail={
            "code": "workflow_assistant_runtime_error",
            "message": message,
            "hint": hint,
        },
    )


async def _read_body(request: Request) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_WORKFLOW_REQUEST_BYTES:
                raise _http_413()
        except HTTPException:
            raise
        except ValueError:
            pass
    raw = await request.body()
    if len(raw) > MAX_WORKFLOW_REQUEST_BYTES:
        raise _http_413()
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
    reject_unknown_spec_fields(raw)
    spec = WorkflowSpec.model_validate(raw)
    validate_graph(spec)
    require_implemented_primitives(spec)
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
    return wfdb.get_definition_record(conn, workflow_id, version)


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


def _redact_execution_for_display(execution: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(execution)
    for key in ("input", "context"):
        if key in redacted:
            redacted[key] = redact_sensitive(redacted[key])
    return redacted


def _redact_node_run_for_display(node_run: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(node_run)
    for key in ("input", "output", "payload", "error"):
        if key in redacted:
            redacted[key] = redact_sensitive(redacted[key])
    return redacted


def _execution_to_dict(execution: wfdb.WorkflowExecution) -> dict[str, Any]:
    return _redact_execution_for_display(
        {
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
    )


def _event_to_dict(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, (dict, list)):
        payload = {}
    return {
        "created_at": event["created_at"],
        "execution_id": event["execution_id"],
        "id": event["id"],
        "kind": event["kind"],
        "node_run_id": event["node_run_id"],
        "payload": redact_sensitive(payload),
    }


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


class PromptAssistantDraftRequest(BaseModel):
    workflow_goal: str = ""
    node_id: str = ""
    profile: str = ""
    provider: str = ""
    model: str = ""
    cell_objective: str = ""
    available_context: list[str] = Field(default_factory=list)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class WorkflowDraftRequest(BaseModel):
    goal: str


class WorkflowRefineRequest(BaseModel):
    instruction: str
    spec: dict[str, Any] | None = None
    workflow_id: str | None = None
    version: int | None = None


class PromptAssistantDraftResponse(BaseModel):
    prompt_text: str
    result_contract: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


def _profile_option(profile: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(profile, "name", "") or ""),
        "provider": str(getattr(profile, "provider", "") or ""),
        "model": str(getattr(profile, "model", "") or ""),
        "description": str(getattr(profile, "description", "") or ""),
        "is_default": bool(getattr(profile, "is_default", False)),
    }


def _draft_cell_prompt(req: PromptAssistantDraftRequest) -> PromptAssistantDraftResponse:
    context_lines = "\n".join(
        f"- {item}" for item in req.available_context if str(item).strip()
    )
    constraint_lines = "\n".join(
        f"- {item}" for item in req.constraints if str(item).strip()
    )
    contract = req.expected_output or {"summary": "string", "status": "string"}
    contract_json = json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True)
    default_constraints = "- Be concise.\n- Do not perform work outside this cell objective."
    routing_lines = []
    if req.provider.strip():
        routing_lines.append(f"Provider override: {req.provider.strip()}")
    if req.model.strip():
        routing_lines.append(f"Model override: {req.model.strip()}")
    routing_context = "\n".join(routing_lines) or "Provider/model: use assigned profile defaults."
    prompt = f"""You are the `{req.profile or 'assigned'}` profile executing workflow cell `{req.node_id or 'cell'}`.

Routing:
{routing_context}

Workflow goal:
{req.workflow_goal or 'Complete the workflow objective.'}

Cell objective:
{req.cell_objective or 'Complete this cell and report the result.'}

Available workflow context:
{context_lines or '- Use the workflow input and upstream node outputs referenced in this prompt.'}

Constraints:
{constraint_lines or default_constraints}

Return JSON only matching this contract:
```json
{contract_json}
```
""".strip()
    return PromptAssistantDraftResponse(
        prompt_text=prompt,
        result_contract=contract,
        notes=["Review placeholders before deploy; they are rendered at execution time."],
    )


@router.post("/prompt-assistant/draft")
def prompt_assistant_draft(req: PromptAssistantDraftRequest) -> dict[str, Any]:
    return _draft_cell_prompt(req).model_dump()


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return workflow_capabilities()


@router.get("/agent-routing-options")
def agent_routing_options() -> dict[str, Any]:
    profiles = [_profile_option(profile) for profile in profiles_mod.list_profiles()]
    models_payload = inventory_mod.build_models_payload(
        inventory_mod.load_picker_context(),
        include_unconfigured=True,
        picker_hints=True,
        canonical_order=True,
        probe_custom_providers=False,
        max_models=500,
    )
    providers = []
    for row in models_payload.get("providers", []):
        if not isinstance(row, dict):
            continue
        providers.append(
            {
                "slug": str(row.get("slug") or row.get("provider") or ""),
                "label": str(row.get("label") or row.get("name") or row.get("slug") or ""),
                "models": [str(model) for model in row.get("models") or []],
                "authenticated": bool(row.get("authenticated", False)),
            }
        )
    return {
        "profiles": profiles,
        "providers": providers,
        "default_provider": str(models_payload.get("provider") or ""),
        "default_model": str(models_payload.get("model") or ""),
    }


def _workflow_tick_interval_seconds(workflow_cfg: dict[str, Any]) -> float:
    raw_interval = workflow_cfg.get("tick_interval_seconds", 30.0)
    try:
        interval = float(raw_interval)
        if not math.isfinite(interval):
            raise ValueError("non-finite interval")
    except (TypeError, ValueError, OverflowError):
        return 30.0
    if interval < 1.0:
        return 1.0
    return interval


@router.get("/status")
def workflow_status() -> dict[str, Any]:
    cfg = load_config()
    workflow_cfg = cfg.get("workflow") if isinstance(cfg, dict) else {}
    if not isinstance(workflow_cfg, dict):
        workflow_cfg = {}
    dispatch_enabled = is_truthy_value(
        workflow_cfg.get("dispatch_in_gateway"), default=True
    )
    interval = _workflow_tick_interval_seconds(workflow_cfg)
    warning = None
    if not dispatch_enabled:
        warning = "Set workflow.dispatch_in_gateway: true and restart gateway, or run hermes workflow tick manually to advance waits, schedules, and completed agent tasks."
    return {
        "dispatcher": {
            "dispatch_in_gateway": dispatch_enabled,
            "tick_interval_seconds": interval,
            "warning": warning,
        }
    }


@router.post("/definitions/draft")
async def draft_definition(request: Request) -> dict[str, Any]:
    try:
        req = WorkflowDraftRequest.model_validate(await _read_body(request))
        result = workflows_assistant.draft_workflow_with_default_runner(req.goal)
    except HTTPException:
        raise
    except workflows_assistant.AssistantValidationError as exc:
        logger.info("workflow draft validation failed: %s", exc)
        raise _assistant_validation_http() from exc
    except (ValueError, ValidationError) as exc:
        raise _http_400(exc) from exc
    except Exception as exc:
        logger.exception("workflow draft runtime failed: %s", type(exc).__name__)
        raise _assistant_runtime_http(str(exc)[:200]) from exc
    return {"draft": result.to_dict()}


@router.post("/definitions/refine")
async def refine_definition(request: Request) -> dict[str, Any]:
    try:
        req = WorkflowRefineRequest.model_validate(await _read_body(request))
        if req.spec is not None:
            spec = WorkflowSpec.model_validate(req.spec)
            validate_graph(spec)
            require_implemented_primitives(spec)
        elif req.workflow_id:
            with _connect_initialized() as conn:
                record = _definition_record(conn, req.workflow_id, req.version)
                spec = record.spec
        else:
            raise ValueError("spec or workflow_id is required")
        result = workflows_assistant.refine_workflow_with_default_runner(
            spec, req.instruction
        )
    except KeyError as exc:
        raise _http_404(exc) from exc
    except HTTPException:
        raise
    except workflows_assistant.AssistantValidationError as exc:
        logger.info("workflow refine validation failed: %s", exc)
        raise _assistant_validation_http() from exc
    except (ValueError, ValidationError) as exc:
        raise _http_400(exc) from exc
    except Exception as exc:
        logger.exception("workflow refine runtime failed: %s", type(exc).__name__)
        raise _assistant_runtime_http(str(exc)[:200]) from exc
    return {"draft": result.to_dict()}


@router.get("/definitions")
def list_definitions() -> dict[str, Any]:
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

        def _deploy():
            with _connect_initialized() as conn:
                deployed_version = wfdb.deploy_definition(
                    conn, spec, created_by="dashboard", auto_bump=True
                )
                return _definition_record(conn, spec.id, deployed_version)

        record = await asyncio.to_thread(_deploy)
    except (json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise _http_400(exc) from exc
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"definition": _definition_to_dict(record, include_spec=True)}


@router.get("/definitions/{workflow_id}")
def get_definition(
    workflow_id: str,
    version: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            record = _definition_record(conn, workflow_id, version)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"definition": _definition_to_dict(record, include_spec=True)}


@router.delete("/definitions/{workflow_id}")
def delete_definition(workflow_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            wfdb.delete_definition(conn, workflow_id)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"deleted": True, "workflow_id": workflow_id}


@router.post("/definitions/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    request: Request,
    version: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        input_data = _input_from_payload(await _read_body(request))

        def _run():
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
            with _connect_initialized() as conn:
                return wfdb.get_execution(conn, execution_id)

        execution = await asyncio.to_thread(_run)
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        raise _http_400(exc) from exc
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"execution": _execution_to_dict(execution)}


@router.get("/executions")
def list_executions(
    workflow_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    with _connect_initialized() as conn:
        executions = [
            _execution_to_dict(e)
            for e in wfdb.list_executions(conn, workflow_id, limit=limit)
        ]
    return {"executions": executions}


@router.get("/executions/{execution_id}")
def get_execution(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            execution = wfdb.get_execution(conn, execution_id)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"execution": _execution_to_dict(execution)}


@router.get("/executions/{execution_id}/node-runs")
def execution_node_runs(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            wfdb.get_execution(conn, execution_id)
            node_runs = [
                _redact_node_run_for_display(row)
                for row in wfdb.list_node_runs(conn, execution_id)
            ]
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"execution_id": execution_id, "node_runs": node_runs}


@router.post("/executions/{execution_id}/cancel")
def cancel_execution(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            execution, cancelled = wfdb.cancel_execution(
                conn, execution_id, source="dashboard"
            )
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"cancelled": cancelled, "execution": _execution_to_dict(execution)}


@router.get("/executions/{execution_id}/events")
def list_events(execution_id: str) -> dict[str, Any]:
    try:
        with _connect_initialized() as conn:
            events = wfdb.list_events(conn, execution_id)
    except KeyError as exc:
        raise _http_404(exc) from exc
    return {"events": [_event_to_dict(event) for event in events]}
