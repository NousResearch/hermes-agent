"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by
rewriting a request or wrapping the actual execution callback. Keep the small
contract helpers here so agent-loop call sites and plugins share one vocabulary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"
ASSISTANT_RESPONSE_MIDDLEWARE = "assistant_response"

# Back-compat aliases for older PoC branches that used API terminology.
API_REQUEST_MIDDLEWARE = LLM_REQUEST_MIDDLEWARE
API_EXECUTION_MIDDLEWARE = LLM_EXECUTION_MIDDLEWARE

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE,
    ASSISTANT_RESPONSE_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)
    control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponseMiddlewareResult:
    """Structured decision from assistant-response validation middleware."""

    action: str = "pass"
    response_text: str = ""
    message: str = ""
    feedback: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    max_retries: Optional[int] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    raw_decision: Dict[str, Any] = field(default_factory=dict)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    Request payloads are normally plain JSON-shaped dicts, but an LLM request
    can occasionally carry non-deepcopyable objects (clients, callbacks, file
    handles). A hard ``deepcopy`` failure there would otherwise abort the whole
    request-middleware pass. Fall back to a shallow ``dict`` copy so middleware
    still runs and the original nested objects are shared by reference rather
    than corrupting the live payload.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        if isinstance(payload, dict):
            return dict(payload)
        return payload


def apply_llm_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered LLM request middleware.

    Middleware may return ``{"request": {...}}`` to replace the effective
    provider kwargs before Hermes sends them.
    """
    if not _has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=request,
            original_payload=request,
            changed=False,
            trace=[],
            control={},
        )

    original_request = _safe_copy(request)
    current_request = _safe_copy(original_request)
    trace: List[Dict[str, Any]] = []
    control: Dict[str, Any] = {}

    for result in _invoke_middleware(
        LLM_REQUEST_MIDDLEWARE,
        request=current_request,
        original_request=original_request,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_request = result.get("request")
        if isinstance(next_request, dict):
            current_request = _safe_copy(next_request)
        next_control = result.get("control")
        if isinstance(next_control, dict):
            control.update(_safe_copy(next_control))
        if isinstance(next_request, dict) or isinstance(next_control, dict):
            trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_request,
        original_payload=original_request,
        changed=bool(trace),
        trace=trace,
        control=control,
    )


def apply_tool_request_middleware(
    tool_name: str,
    args: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware.

    Middleware may return ``{"args": {...}}`` to replace the effective tool
    arguments before hooks, guardrails, approvals, and execution see them.
    """
    if not _has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args,
            original_payload=args,
            changed=False,
            trace=[],
        )

    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: List[Dict[str, Any]] = []

    for result in _invoke_middleware(
        TOOL_REQUEST_MIDDLEWARE,
        tool_name=tool_name,
        args=current_args,
        original_args=original_args,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_args = result.get("args")
        if not isinstance(next_args, dict):
            continue
        current_args = _safe_copy(next_args)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_args,
        original_payload=original_args,
        changed=bool(trace),
        trace=trace,
    )


def apply_api_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Compatibility wrapper for older ``api_request`` naming."""
    return apply_llm_request_middleware(request, **context)


def apply_assistant_response_middleware(
    response_text: str,
    **context: Any,
) -> AssistantResponseMiddlewareResult:
    """Apply final-draft assistant response validation middleware.

    Middleware may return a structured decision dict with action:
    ``pass``, ``rewrite``, ``retry_with_feedback``, ``require_tool``, or
    ``block``. The first decisive non-pass action wins; pass decisions are
    traced and evaluation continues so multiple validators can observe.
    """
    original_text = response_text or ""
    if not _has_middleware(ASSISTANT_RESPONSE_MIDDLEWARE):
        return AssistantResponseMiddlewareResult(
            action="pass",
            response_text=original_text,
            trace=[],
        )

    trace: List[Dict[str, Any]] = []
    for result in _invoke_middleware(
        ASSISTANT_RESPONSE_MIDDLEWARE,
        response_text=original_text,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        decision = _normalize_assistant_response_decision(result, original_text)
        trace.append(_trace_entry(result))
        decision.trace = list(trace)
        if decision.action != "pass":
            return decision

    return AssistantResponseMiddlewareResult(
        action="pass",
        response_text=original_text,
        trace=trace,
    )


def run_llm_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    callbacks = _get_middleware_callbacks(LLM_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return next_call(request)
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        request=request,
        original_request=context.pop("original_request", request),
        **context,
    )


def run_tool_execution_middleware(
    tool_name: str,
    args: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware."""
    callbacks = _get_middleware_callbacks(TOOL_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return next_call(args)
    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        tool_name=tool_name,
        args=args,
        original_args=context.pop("original_args", args),
        **context,
    )


def run_api_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Compatibility wrapper for older ``api_execution`` naming."""
    return run_llm_execution_middleware(request, next_call, **context)


def _normalize_tool_call(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    name = value.get("name") or value.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None
    raw_args = value.get("args", value.get("arguments", {}))
    if not isinstance(raw_args, dict):
        raw_args = {}
    normalized: Dict[str, Any] = {
        "name": name.strip(),
        "args": _safe_copy(raw_args),
    }
    reason = value.get("reason")
    if isinstance(reason, str) and reason.strip():
        normalized["reason"] = reason.strip()
    read_only = value.get("read_only")
    if isinstance(read_only, bool):
        normalized["read_only"] = read_only
    return normalized


def _normalize_tool_calls(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: List[Dict[str, Any]] = []
    for item in value:
        call = _normalize_tool_call(item)
        if call is not None:
            calls.append(call)
    return calls


def _normalize_assistant_response_decision(
    result: Dict[str, Any],
    original_text: str,
) -> AssistantResponseMiddlewareResult:
    action = str(result.get("action") or result.get("verdict") or "pass").strip().lower()
    aliases = {
        "allow": "pass",
        "ok": "pass",
        "revise": "rewrite",
        "retry": "retry_with_feedback",
        "retry_with_feedback": "retry_with_feedback",
        "require_evidence": "require_tool",
        "evidence_required": "require_tool",
        "tool": "require_tool",
        "deny": "block",
    }
    action = aliases.get(action, action)
    if action not in {"pass", "rewrite", "retry_with_feedback", "require_tool", "block"}:
        action = "block"

    message = result.get("message", "")
    replacement = result.get("response_text", result.get("revised_response"))
    if replacement is None and action == "rewrite":
        replacement = message or original_text
    elif replacement is None:
        replacement = original_text
    response_text = str(replacement) if replacement is not None else original_text
    feedback = result.get("feedback", "")
    try:
        max_retries = result.get("max_retries")
        max_retries = int(max_retries) if max_retries is not None else None
    except Exception:
        max_retries = None

    return AssistantResponseMiddlewareResult(
        action=action,
        response_text=response_text,
        message=str(message or ""),
        feedback=str(feedback or ""),
        tool_calls=_normalize_tool_calls(result.get("tool_calls")),
        max_retries=max_retries,
        raw_decision=_safe_copy(result),
    )


def _invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    from hermes_cli.plugins import invoke_middleware

    return invoke_middleware(kind, **middleware_payload(**kwargs))


def _has_middleware(kind: str) -> bool:
    from hermes_cli.plugins import has_middleware

    return has_middleware(kind)


def _get_middleware_callbacks(kind: str) -> List[Callable]:
    from hermes_cli.plugins import get_plugin_manager

    return list(get_plugin_manager()._middleware.get(kind, []))


def _run_execution_chain(
    kind: str,
    callbacks: List[Callable],
    terminal_call: Callable[[Any], Any],
    **kwargs: Any,
) -> Any:
    payload_key = "request" if "request" in kwargs else "args"

    class _DownstreamExecutionError(Exception):
        def __init__(self, original: BaseException) -> None:
            super().__init__(str(original))
            self.original = original

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # ``next_call`` is single-use per middleware frame. Calling it more
            # than once would re-run the downstream provider/tool, so a second
            # invocation is a contract violation rather than a retry. Surface it
            # instead of silently executing the terminal call twice.
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
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])


def _trace_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    for key in ("source", "reason", "name"):
        value = result.get(key)
        if isinstance(value, str) and value:
            entry[key] = value
    if not entry:
        entry["source"] = "plugin"
    return entry
