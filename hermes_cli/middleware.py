"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by
rewriting a request or wrapping the actual execution callback. Keep the small
contract helpers here so agent-loop call sites and plugins share one vocabulary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from agent.memory_manager import sanitize_recall_payload

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"

# Back-compat aliases for older PoC branches that used API terminology.
API_REQUEST_MIDDLEWARE = LLM_REQUEST_MIDDLEWARE
API_EXECUTION_MIDDLEWARE = LLM_EXECUTION_MIDDLEWARE

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


class _SanitizedString(str):
    """Marker subclass so execution middleware can see scrubbed text safely."""


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


def _restore_sanitized_payload(raw_payload: Any, safe_payload: Any, current_payload: Any) -> Any:
    """Reapply scrubbed branches when middleware leaves them unchanged."""
    if (
        hasattr(raw_payload, "arguments")
        and hasattr(raw_payload, "name")
        and hasattr(current_payload, "function")
        and hasattr(safe_payload, "function")
    ):
        restored = _safe_copy(raw_payload)
        current_function = getattr(current_payload, "function", None)
        safe_function = getattr(safe_payload, "function", None)
        current_name = getattr(current_payload, "name", None)
        current_arguments = getattr(current_payload, "arguments", None)
        safe_name = getattr(safe_payload, "name", None)
        safe_arguments = getattr(safe_payload, "arguments", None)
        if current_function is not None and safe_function is not None:
            if getattr(current_function, "name", None) != getattr(safe_function, "name", None):
                current_name = getattr(current_function, "name", None)
            if getattr(current_function, "arguments", None) != getattr(safe_function, "arguments", None):
                current_arguments = getattr(current_function, "arguments", None)
            try:
                setattr(
                    restored,
                    "name",
                    _restore_sanitized_payload(
                        getattr(raw_payload, "name", None),
                        safe_name,
                        current_name,
                    ),
                )
            except Exception:
                pass
            try:
                setattr(
                    restored,
                    "arguments",
                    _restore_sanitized_payload(
                        getattr(raw_payload, "arguments", None),
                        safe_arguments,
                        current_arguments,
                    ),
                )
            except Exception:
                pass
        current_dict = vars(current_payload) if hasattr(current_payload, "__dict__") else {}
        safe_dict = vars(safe_payload) if hasattr(safe_payload, "__dict__") else {}
        raw_dict = vars(raw_payload) if hasattr(raw_payload, "__dict__") else {}
        for key, value in current_dict.items():
            if key in {"function", "name", "arguments"}:
                continue
            if key in raw_dict and key in safe_dict:
                merged = _restore_sanitized_payload(raw_dict[key], safe_dict[key], value)
            else:
                merged = value
            try:
                setattr(restored, key, merged)
            except Exception:
                pass
        return restored

    if isinstance(current_payload, SimpleNamespace):
        restored = _safe_copy(raw_payload) if hasattr(raw_payload, "__dict__") else SimpleNamespace()
        safe_dict = vars(safe_payload) if hasattr(safe_payload, "__dict__") else {}
        raw_dict = vars(raw_payload) if hasattr(raw_payload, "__dict__") else {}
        for key, value in vars(current_payload).items():
            if key in raw_dict and key in safe_dict:
                merged = _restore_sanitized_payload(raw_dict[key], safe_dict[key], value)
            else:
                merged = value
            try:
                setattr(restored, key, merged)
            except Exception:
                pass
        return restored

    if isinstance(current_payload, dict):
        restored: Dict[str, Any] = {}
        for key, value in current_payload.items():
            if (
                isinstance(raw_payload, dict)
                and isinstance(safe_payload, dict)
                and key in raw_payload
                and key in safe_payload
            ):
                restored[key] = _restore_sanitized_payload(
                    raw_payload[key],
                    safe_payload[key],
                    value,
                )
            else:
                restored[key] = value
        return restored

    if isinstance(current_payload, list):
        restored_list: List[Any] = []
        for idx, value in enumerate(current_payload):
            if (
                isinstance(raw_payload, list)
                and isinstance(safe_payload, list)
                and idx < len(raw_payload)
                and idx < len(safe_payload)
            ):
                restored_list.append(
                    _restore_sanitized_payload(raw_payload[idx], safe_payload[idx], value)
                )
            else:
                restored_list.append(value)
        return restored_list

    if current_payload == safe_payload and raw_payload != safe_payload:
        return _safe_copy(raw_payload)
    return current_payload


def _sanitize_execution_result(value: Any) -> Any:
    if isinstance(value, str):
        sanitized = sanitize_recall_payload(value)
        return _SanitizedString(sanitized)
    if isinstance(value, list):
        return [_sanitize_execution_result(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_execution_result(item)
            for key, item in value.items()
        }
    if hasattr(value, "arguments") and hasattr(value, "name"):
        sanitized_arguments = sanitize_recall_payload(getattr(value, "arguments", None))
        provider_data = _sanitize_execution_result(getattr(value, "provider_data", None))
        tool_call = SimpleNamespace(
            id=getattr(value, "id", None),
            type=getattr(value, "type", None) or "function",
            name=getattr(value, "name", None),
            arguments=sanitized_arguments,
            provider_data=provider_data,
            function=SimpleNamespace(
                name=getattr(value, "name", None),
                arguments=sanitized_arguments,
            ),
        )
        for attr_name in ("call_id", "response_item_id", "extra_content"):
            attr_value = getattr(value, attr_name, None)
            if attr_value is not None:
                setattr(tool_call, attr_name, _sanitize_execution_result(attr_value))
        return tool_call
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(
            **{
                key: _sanitize_execution_result(val)
                for key, val in value.__dict__.items()
            }
        )
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return SimpleNamespace(
            **{
                key: _sanitize_execution_result(val)
                for key, val in vars(value).items()
            }
        )
    return value


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
        )

    original_request = _safe_copy(request)
    safe_original_request = sanitize_recall_payload(_safe_copy(original_request))
    current_request = _safe_copy(safe_original_request)
    trace: List[Dict[str, Any]] = []

    for result in _invoke_middleware(
        LLM_REQUEST_MIDDLEWARE,
        request=current_request,
        original_request=safe_original_request,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_request = result.get("request")
        if not isinstance(next_request, dict):
            continue
        current_request = _safe_copy(next_request)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=_restore_sanitized_payload(request, safe_original_request, current_request),
        original_payload=safe_original_request,
        changed=bool(trace),
        trace=trace,
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
    safe_original_args = sanitize_recall_payload(_safe_copy(original_args))
    current_args = _safe_copy(safe_original_args)
    trace: List[Dict[str, Any]] = []

    for result in _invoke_middleware(
        TOOL_REQUEST_MIDDLEWARE,
        tool_name=tool_name,
        args=current_args,
        original_args=safe_original_args,
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
        payload=_restore_sanitized_payload(args, safe_original_args, current_args),
        original_payload=safe_original_args,
        changed=bool(trace),
        trace=trace,
    )


def apply_api_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Compatibility wrapper for older ``api_request`` naming."""
    return apply_llm_request_middleware(request, **context)


def run_llm_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    callbacks = _get_middleware_callbacks(LLM_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return next_call(request)
    original_request = context.pop("original_request", request)
    safe_request = sanitize_recall_payload(_safe_copy(request))
    safe_original_request = sanitize_recall_payload(_safe_copy(original_request))
    exposed_results: Dict[int, tuple[Any, Any]] = {}

    def _terminal_call(next_request: Dict[str, Any]) -> Any:
        restored_request = _restore_sanitized_payload(request, safe_request, next_request)
        raw_result = next_call(restored_request)
        exposed_result = _sanitize_execution_result(raw_result)
        if exposed_result is raw_result:
            return raw_result
        exposed_results[id(exposed_result)] = (raw_result, _safe_copy(exposed_result))
        return exposed_result

    def _unwrap_result(value: Any) -> Any:
        stored = exposed_results.get(id(value))
        if stored is None:
            return value
        raw_result, safe_snapshot = stored
        return (
            raw_result
            if value == safe_snapshot
            else _restore_sanitized_payload(raw_result, safe_snapshot, value)
        )

    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE,
        callbacks,
        _terminal_call,
        unwrap_result=_unwrap_result,
        request=safe_request,
        original_request=safe_original_request,
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
    original_args = context.pop("original_args", args)
    safe_args = sanitize_recall_payload(_safe_copy(args))
    safe_original_args = sanitize_recall_payload(_safe_copy(original_args))
    exposed_results: Dict[int, tuple[Any, Any]] = {}

    def _terminal_call(next_args: Dict[str, Any]) -> Any:
        restored_args = _restore_sanitized_payload(args, safe_args, next_args)
        raw_result = next_call(restored_args)
        exposed_result = _sanitize_execution_result(raw_result)
        if exposed_result is raw_result:
            return raw_result
        exposed_results[id(exposed_result)] = (raw_result, _safe_copy(exposed_result))
        return exposed_result

    def _unwrap_result(value: Any) -> Any:
        stored = exposed_results.get(id(value))
        if stored is None:
            return value
        raw_result, safe_snapshot = stored
        return (
            raw_result
            if value == safe_snapshot
            else _restore_sanitized_payload(raw_result, safe_snapshot, value)
        )

    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE,
        callbacks,
        _terminal_call,
        unwrap_result=_unwrap_result,
        tool_name=tool_name,
        args=safe_args,
        original_args=safe_original_args,
        **context,
    )


def run_api_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Compatibility wrapper for older ``api_execution`` naming."""
    return run_llm_execution_middleware(request, next_call, **context)


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
    unwrap_result = kwargs.pop("unwrap_result", None)

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
            result = callback(**call_kwargs)
            return unwrap_result(result) if callable(unwrap_result) else result
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
                return unwrap_result(next_result) if callable(unwrap_result) else next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    result = call_at(0, kwargs[payload_key])
    return unwrap_result(result) if callable(unwrap_result) else result


def _trace_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    for key in ("source", "reason", "name"):
        value = result.get(key)
        if isinstance(value, str) and value:
            entry[key] = value
    if not entry:
        entry["source"] = "plugin"
    return entry
