"""Fail-closed local model admission decisions, separated from middleware plumbing."""

from __future__ import annotations

import logging
import os
from typing import Any

from hermes_cli.local_model_routes import is_local_llm_route as _is_local_llm_route

logger = logging.getLogger(__name__)
_LLM_ADMISSION_MIDDLEWARE = "llm_admission"


def _is_type_only_json_schema(schema: Any) -> bool:
    """Allow only the structural/type subset used by the local extraction capability."""
    if type(schema) is not dict:
        return False
    if set(schema) == {"anyOf"}:
        choices = schema.get("anyOf")
        return type(choices) is list and bool(choices) and all(
            _is_type_only_json_schema(choice) for choice in choices
        )
    if not isinstance(schema.get("type"), str):
        return False
    kind = schema["type"]
    if kind == "object":
        if set(schema) == {"type"}:
            return True
        properties = schema.get("properties")
        required = schema.get("required")
        return (
            set(schema) == {"type", "properties", "required", "additionalProperties"}
            and type(properties) is dict and type(required) is list
            and required == list(properties) and schema.get("additionalProperties") is False
            and all(isinstance(key, str) and _is_type_only_json_schema(value)
                    for key, value in properties.items())
        )
    if kind == "array":
        if set(schema) == {"type"}:
            return True
        return set(schema) == {"type", "items"} and _is_type_only_json_schema(schema["items"])
    if kind in {"string", "integer", "number", "boolean", "null"}:
        return set(schema) == {"type"}
    return False


def _valid_response_format(value: Any) -> bool:
    return (
        type(value) is dict and set(value) == {"type", "json_schema"}
        and value.get("type") == "json_schema"
        and type(value.get("json_schema")) is dict
        and set(value["json_schema"]) == {"schema"}
        and _is_type_only_json_schema(value["json_schema"].get("schema"))
    )


def _admission_scope_string(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def _controller_callbacks() -> tuple[list[Any], str | None]:
    """Load exactly one controller callback, returning a safe denial code if unavailable."""
    try:
        from hermes_cli.plugins import _delivery_manager, has_middleware

        if not has_middleware(_LLM_ADMISSION_MIDDLEWARE):
            return [], "controller_missing"
        callbacks = list(_delivery_manager()._middleware.get(_LLM_ADMISSION_MIDDLEWARE, []))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Local-model admission discovery failed closed: %s", exc)
        return [], "controller_unavailable"
    return callbacks, None


def _request_scope(context: dict[str, Any]) -> tuple[dict[str, str], Any | None]:
    """Normalize and require every immutable task/run/request route field."""
    keys = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db", "kanban_workspace",
        "kanban_claim_lock", "capability_id", "task_contract_sha256", "api_request_id",
        "context_length", "ollama_num_ctx", "session_id", "model", "provider", "base_url",
        "streaming", "has_tools", "api_call_count", "existing_response_format",
    )
    scope = {key: _admission_scope_string(context.get(key)) for key in keys}
    if not all(scope.values()):
        from hermes_cli.middleware import LLMAdmissionDecision
        return scope, LLMAdmissionDecision(
            False, "missing_admission_identity",
            "Local model admission requires task/run, capability, contract, request, session, model, and endpoint identity",
        )
    return scope, None


def _validate_schema_admission(result: dict[str, Any], context: dict[str, Any], lease_ids: tuple[str, ...],
                               policy_id: str, evidence_id: str):
    """Check a controller-provided response schema against the exact request phase."""
    from hermes_cli.middleware import LLMAdmissionDecision

    response_format = result.get("response_format")
    if response_format is None:
        return None
    ids = ((policy_id,), (evidence_id,), lease_ids)
    if not _valid_response_format(response_format):
        return LLMAdmissionDecision(False, "invalid_admission_response_format",
                                    "Admission returned a malformed or value-constraining JSON Schema", *ids)
    if context.get("streaming") or context.get("has_tools"):
        return LLMAdmissionDecision(False, "request_format_incompatible",
                                    "This local capability requires a non-streaming, tool-free request", *ids)
    phase = context.get("request_format_phase")
    if phase == "prepare":
        conflict = context.get("existing_response_format") or int(context.get("api_call_count") or 0) != 1
        if not conflict:
            return None
        return LLMAdmissionDecision(False, "request_format_conflict",
                                    "This local capability permits one first request with its registry-derived schema", *ids)
    if phase == "verify":
        if context.get("supplied_response_format") == response_format:
            return None
        return LLMAdmissionDecision(False, "request_format_not_applied",
                                    "Provider request does not carry the registry-pinned JSON Schema", *ids)
    return LLMAdmissionDecision(False, "request_format_phase_missing",
                                "Local JSON Schema admission was not applied at the request boundary", *ids)


def _validate_controller_result(result: Any, context: dict[str, Any], expected_scope: dict[str, str],
                                existing_lease_ids: tuple[str, ...]):
    """Validate a controller receipt and bind it to this task/run and request."""
    from hermes_cli.middleware import LLMAdmissionDecision

    if not isinstance(result, dict):
        return LLMAdmissionDecision(False, "invalid_controller_result",
                                    "Admission controller returned no structured decision")
    decision, policy_id, reason = result.get("decision"), result.get("policy_id"), result.get("reason")
    if decision not in {"allow", "block"} or not all(
        isinstance(value, str) and value.strip() for value in (policy_id, reason)
    ):
        return LLMAdmissionDecision(False, "invalid_controller_result",
                                    "Admission controller returned an incomplete decision")
    if decision == "block":
        return LLMAdmissionDecision(False, "controller_block", reason.strip(), (policy_id.strip(),))
    evidence_id, raw_leases = result.get("evidence_id"), result.get("lease_ids")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        return LLMAdmissionDecision(False, "missing_admission_evidence",
                                    "Admission allowed a request without an evidence ID", (policy_id.strip(),))
    if not isinstance(raw_leases, (list, tuple)) or not raw_leases or not all(
        isinstance(item, str) and item.strip() for item in raw_leases
    ):
        return LLMAdmissionDecision(False, "missing_gpu_lease",
                                    "Admission allowed a local request without GPU lease IDs",
                                    (policy_id.strip(),), (evidence_id.strip(),))
    lease_ids = tuple(item.strip() for item in raw_leases)
    ids = ((policy_id.strip(),), (evidence_id.strip(),), lease_ids)
    if existing_lease_ids and lease_ids != existing_lease_ids:
        return LLMAdmissionDecision(False, "admission_lease_changed",
                                    "Controller did not validate the current task/run lease", *ids)
    if result.get("scope") != expected_scope:
        return LLMAdmissionDecision(False, "admission_scope_mismatch",
                                    "Admission is not bound to the current task run, API request, and exact model route", *ids)
    schema_denial = _validate_schema_admission(result, context, lease_ids, *[x[0] for x in ids[:2]])
    if schema_denial is not None:
        return schema_denial
    return LLMAdmissionDecision(True, "admitted", "Admission passed", *ids, result.get("response_format"))


def run_llm_admission_middleware(**context: Any):
    """Resolve fail-closed admission before a local model provider request."""
    from hermes_cli.middleware import LLMAdmissionDecision, middleware_payload

    local_route = _is_local_llm_route(context.get("provider"), context.get("base_url"))
    if not local_route:
        changed_route = context.get("capability_id") and (context.get("kanban_task_id") or context.get("task_id"))
        if changed_route:
            return LLMAdmissionDecision(False, "local_capability_route_changed",
                                        "A local-GPU task cannot fall through to a remote model route or fallback")
        return LLMAdmissionDecision(True, "not_local_route", "Admission is only required for local routes")
    expected_scope, denial = _request_scope(context)
    if denial:
        return denial
    callbacks, error = _controller_callbacks()
    if error:
        reason = "No local-model admission controller is enabled" if error == "controller_missing" else \
                 "Local-model admission controller could not be loaded"
        return LLMAdmissionDecision(False, error, reason)
    if len(callbacks) != 1:
        return LLMAdmissionDecision(False, "controller_configuration_invalid",
                                    "Exactly one local-model admission controller must be enabled")
    lease_ids = tuple(context.pop("existing_lease_ids", ()) or ())
    action = "validate" if lease_ids else "acquire"
    try:
        result = callbacks[0](action=action, lease_ids=lease_ids, **middleware_payload(**context))
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.error("Local-model admission callback failed closed: %s", exc)
        return LLMAdmissionDecision(False, "controller_error",
                                    "Local-model admission check failed; no model request was sent")
    return _validate_controller_result(result, context, expected_scope, lease_ids)


def require_local_model_admission(agent: Any, api_kwargs: dict[str, Any], *, streaming: bool = False) -> None:
    """Revalidate the task-bound GPU lease at each sync or streaming provider seam."""
    import uuid

    from agent.delegation_context import owned_local_gpu_task_context
    from hermes_cli.middleware import LocalModelAdmissionBlockedError

    if not _is_local_llm_route(getattr(agent, "provider", ""), getattr(agent, "base_url", "")):
        return
    prior = getattr(agent, "_local_gpu_admission_scope", {})
    if not isinstance(prior, dict):
        prior = {}
    local_task_context = owned_local_gpu_task_context()
    context = {
        **prior,
        "task_id": prior.get("task_id") or os.environ.get("HERMES_KANBAN_TASK", ""),
        "kanban_task_id": os.environ.get("HERMES_KANBAN_TASK", ""),
        "task_run_id": os.environ.get("HERMES_KANBAN_RUN_ID", ""),
        "kanban_board": os.environ.get("HERMES_KANBAN_BOARD", ""),
        "kanban_db": os.environ.get("HERMES_KANBAN_DB", ""),
        "kanban_workspace": os.environ.get("HERMES_KANBAN_WORKSPACE", ""),
        "kanban_claim_lock": os.environ.get("HERMES_KANBAN_CLAIM_LOCK", ""),
        "capability_id": local_task_context.get("capability_id", ""),
        "task_contract_sha256": local_task_context.get("task_contract_sha256", ""),
        "api_request_id": uuid.uuid4().hex,
        "context_length": str(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or ""),
        "ollama_num_ctx": str(getattr(agent, "_ollama_num_ctx", 0) or ""),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "api_mode": str(getattr(agent, "api_mode", "") or ""),
        "api_call_count": int(getattr(agent, "_api_call_count", 0) or 0),
        "streaming": streaming,
        "has_tools": bool(api_kwargs.get("tools")),
        "existing_response_format": bool(api_kwargs.get("response_format") or
                                          ((api_kwargs.get("extra_body") or {}).get("response_format")
                                           if isinstance(api_kwargs.get("extra_body"), dict) else False)),
        "request_format_phase": "verify",
        "supplied_response_format": api_kwargs.get("response_format"),
        "existing_lease_ids": tuple(getattr(agent, "_local_gpu_admission_lease_ids", ()) or ()),
    }
    decision = run_llm_admission_middleware(**context)
    if not decision.allowed:
        raise LocalModelAdmissionBlockedError(
            f"Local model request stopped before provider I/O ({decision.code}): {decision.reason}"
        )
    agent._local_gpu_admission_lease_ids = decision.lease_ids
    agent._local_gpu_admission_scope = {
        key: str(context.get(key) or "") for key in (
            "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db",
            "kanban_workspace", "kanban_claim_lock", "capability_id", "task_contract_sha256",
            "api_request_id", "context_length", "session_id", "model", "provider", "base_url",
            "ollama_num_ctx",
        )
    }


def call_local_model_admission_backend(action: str, *, lease_ids: tuple[str, ...] = (),
                                       **context: Any) -> tuple[list[Any], int]:
    """Discover and call exactly one plugin backend while preserving the middleware facade API."""
    from hermes_cli.plugins import _delivery_manager, has_middleware
    from hermes_cli.middleware import middleware_payload

    if not has_middleware(_LLM_ADMISSION_MIDDLEWARE):
        return [], 0
    callbacks = list(_delivery_manager()._middleware.get(_LLM_ADMISSION_MIDDLEWARE, []))
    results = [callback(action=action, lease_ids=lease_ids, **middleware_payload(**context))
               for callback in callbacks]
    return results, len(callbacks)


def release_local_gpu_admission_leases(*, lease_ids: tuple[str, ...], scope: dict[str, Any]) -> bool:
    """Release turn-owned leases through the single configured controller."""
    from hermes_cli.middleware import middleware_payload

    if not lease_ids:
        return False
    try:
        results, callback_count = call_local_model_admission_backend(
            "release", lease_ids=tuple(lease_ids), **scope,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Local-model admission release failed closed: %s", exc)
        return False
    expected_scope = {key: str(value or "") for key, value in scope.items()}
    return callback_count == 1 and len(results) == 1 and bool(
        isinstance(results[0], dict)
        and results[0].get("decision") == "released"
        and tuple(results[0].get("lease_ids") or ()) == tuple(lease_ids)
        and results[0].get("scope") == expected_scope
    )
