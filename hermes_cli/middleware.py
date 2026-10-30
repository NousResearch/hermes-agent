"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"
LLM_ADMISSION_MIDDLEWARE = "llm_admission"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE, LLM_ADMISSION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LLMAdmissionDecision:
    """Trusted pre-request decision for a local model route.

    ``policy_ids`` and ``evidence_ids`` identify the controller records that allowed a
    request. A denial is terminal for the current turn and must never fall through to the
    provider or a configured model fallback.
    """

    allowed: bool
    code: str
    reason: str
    policy_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    lease_ids: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None


class LocalModelAdmissionBlockedError(RuntimeError):
    """Terminal refusal raised at a provider adapter before local inference begins."""


class LocalGPUAdmissionController:
    """Lifecycle contract for a trusted local-inference admission backend.

    Implementations must atomically validate capability and acquire a task/run-bound GPU
    reservation in ``acquire``. The provider adapter must validate that reservation immediately
    before each inference request. ``release`` is called from the outer turn finalizer and must
    be idempotent and holder-qualified. This interface intentionally has no default backend:
    without an installed controller, local inference remains denied.
    """

    def __call__(self, *, action: str, lease_ids: tuple[str, ...] = (), **context: Any) -> dict[str, Any]:
        """Plugin middleware entry point used by the pre-request gate and turn finalizer."""
        if action == "acquire":
            return self.acquire(**context)
        if action == "validate":
            return self.validate(lease_ids=lease_ids, **context)
        if action == "release":
            return self.release(lease_ids=lease_ids, **context)
        if action == "verify_completion":
            return self.verify_completion(**context)
        return {
            "decision": "block", "policy_id": "invalid-action",
            "reason": "Unknown local GPU admission lifecycle action",
        }

    def acquire(self, **scope: Any) -> dict[str, Any]:  # pragma: no cover - interface contract
        raise NotImplementedError

    def validate(self, *, lease_ids: tuple[str, ...], **scope: Any) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def release(self, *, lease_ids: tuple[str, ...], **scope: Any) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def verify_completion(self, **scope: Any) -> dict[str, Any]:  # pragma: no cover
        """Independently verify the exact task artifact before Kanban terminal mutation."""
        raise NotImplementedError


def observer_payload(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    An LLM request can carry clients/callbacks/file handles; a hard ``deepcopy`` failure would
    otherwise abort the whole request-middleware pass.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        return dict(payload) if isinstance(payload, dict) else payload


def _apply_request_chain(
    kind: str, payload_key: str, trace: list[dict[str, Any]], original: Any, **kwargs: Any
) -> RequestMiddlewareResult:
    """Feed ``kwargs[payload_key]`` through every ``kind`` middleware; each may return ``{payload_key: {...}}``."""
    from hermes_cli.plugins import invoke_middleware

    current = kwargs[payload_key]
    for result in invoke_middleware(kind, _payload_key=payload_key, **middleware_payload(**kwargs)):
        if not isinstance(result, dict):
            continue
        next_payload = result.get(payload_key)
        if not isinstance(next_payload, dict):
            continue
        current = _safe_copy(next_payload)
        entry = {
            key: value
            for key in ("source", "reason", "name")
            if isinstance(value := result.get(key), str) and value
        }
        trace.append(entry or {"source": "plugin"})
    return RequestMiddlewareResult(
        payload=current, original_payload=original, changed=bool(trace), trace=trace,
    )


def apply_llm_request_middleware(request: dict[str, Any], **context: Any) -> RequestMiddlewareResult:
    """Apply registered LLM request middleware; ``{"request": {...}}`` replaces the provider kwargs."""
    from hermes_cli.plugins import has_middleware

    if not has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(payload=request, original_payload=request)

    original_request = _safe_copy(request)
    return _apply_request_chain(
        LLM_REQUEST_MIDDLEWARE, "request", [], original_request,
        request=_safe_copy(original_request), original_request=original_request, **context,
    )


def apply_tool_request_middleware(
    tool_name: str, args: dict[str, Any], **context: Any
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware; ``{"args": {...}}`` replaces the effective tool
    arguments before hooks, guardrails, approvals, and execution see them."""
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: list[dict[str, Any]] = []

    session_id = str(context.get("session_id") or "")
    skip_relay = bool(context.pop("skip_relay", False))
    if session_id and not skip_relay:
        from agent import relay_runtime

        relay_args = relay_runtime.apply_tool_request_intercepts(
            session_id=session_id, tool_name=tool_name, args=current_args)
        if relay_args != current_args:
            current_args = _safe_copy(relay_args)
            trace.append({"source": "nemo_relay"})

    from hermes_cli.plugins import has_middleware

    if not has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args if not trace else current_args, original_payload=args,
            changed=bool(trace), trace=trace,
        )
    return _apply_request_chain(
        TOOL_REQUEST_MIDDLEWARE, "args", trace, original_args,
        tool_name=tool_name, args=current_args, original_args=original_args, **context,
    )


def run_llm_execution_middleware(
    request: dict[str, Any], next_call: Callable[[dict[str, Any]], Any], **context: Any) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE, next_call,
        request=request, original_request=context.pop("original_request", request), **context)


from hermes_cli.local_gpu_admission_middleware import (
    call_local_model_admission_backend,
    _admission_scope_string,
    _is_type_only_json_schema,
    _valid_response_format,
    require_local_model_admission,
    release_local_gpu_admission_leases,
    run_llm_admission_middleware,
)
from hermes_cli.local_model_routes import (
    _host_resolves_to_local_address,
    _is_known_official_provider_host,
    is_local_llm_route as _is_local_llm_route,
)

def verify_local_model_task_completion(**context: Any) -> dict[str, str] | None:
    """Require exactly one trusted controller to verify a local task before it can complete.

    The controller must re-read the pinned task/run, validate the active lease and task digest,
    constrain artifacts to the task workspace, verify the pinned external oracle, and return an
    exact scope echo plus immutable evidence identifiers. This helper never accepts a model's
    self-reported PASS.
    """
    required = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board", "kanban_db",
        "kanban_workspace", "capability_id", "task_contract_sha256", "workspace_path", "assignee",
        "model_override", "provider_override", "completion_contract", "artifacts",
    )
    scope = {key: str(context.get(key) or "") for key in required if key != "artifacts"}
    artifacts = context.get("artifacts")
    required_identity = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board", "kanban_db",
        "capability_id", "task_contract_sha256", "workspace_path", "assignee",
    )
    if (not all(scope.get(key) for key in required_identity)
            or not isinstance(artifacts, (list, tuple)) or not artifacts):
        raise RuntimeError("Local task completion lacks an exact task/run or artifact scope")
    normalized_artifacts = [str(path) for path in artifacts]
    if any(not path.strip() for path in normalized_artifacts):
        raise RuntimeError("Local task completion contains an empty artifact path")
    expected_scope = {**scope, "artifacts": normalized_artifacts}

    try:
        results, callback_count = call_local_model_admission_backend(
            "verify_completion", **expected_scope,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Local task verifier discovery failed closed: %s", exc)
        raise RuntimeError("Local-model controller could not verify this task") from exc
    if callback_count == 0:
        raise RuntimeError("No local-model controller is available to verify this task")
    if callback_count != 1:
        raise RuntimeError("Exactly one local-model controller is required for task verification")
    result = results[0]
    if not isinstance(result, dict):
        raise RuntimeError("Local-model controller returned no verification receipt")
    expected_text = ("policy_id", "reason", "evidence_id", "verifier_sha256", "artifact_sha256")
    if (result.get("decision") != "verified"
            or result.get("scope") != expected_scope
            or any(not isinstance(result.get(key), str) or not result[key].strip()
                   for key in expected_text)):
        reason = result.get("reason") if isinstance(result.get("reason"), str) else "invalid receipt"
        raise RuntimeError(f"Independent local-task verification refused completion: {reason}")
    return {key: result[key].strip() for key in expected_text}


def run_tool_execution_middleware(
    tool_name: str, args: dict[str, Any], next_call: Callable[[dict[str, Any]], Any], **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware."""
    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE, next_call,
        tool_name=tool_name, args=args, original_args=context.pop("original_args", args), **context)


class _DownstreamExecutionError(Exception):
    """Marks an exception raised BELOW a middleware frame so the frame's own failure handling
    (skip-and-continue) doesn't swallow it."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _run_execution_chain(kind: str, terminal_call: Callable[[Any], Any], **kwargs: Any) -> Any:
    from hermes_cli.plugins import _delivery_manager

    payload_key = "request" if "request" in kwargs else "args"
    manager = _delivery_manager()
    callbacks = list(manager._middleware.get(kind, []))
    if not callbacks:
        return terminal_call(kwargs[payload_key])

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # Single-use per frame: a second call would re-run the downstream provider/tool, so it
            # is a contract violation, not a retry.
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            next_called = True
            try:
                next_result = call_at(index + 1, payload if next_payload is None else next_payload)
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            return callback(**call_kwargs)
        except _DownstreamExecutionError as exc:
            raise exc.original
        except Exception as exc:
            # Runs once per tool/LLM call: a mis-declared callback fails identically every time,
            # so it goes through the manager's warn-once reporter (#111922).
            manager._report_hook_failure(kind, callback, call_kwargs, exc, surface="Middleware")
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])
