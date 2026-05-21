"""Hermes middleware contract helpers.

This module is the central integration surface for backend-neutral observer
hooks and execution middleware. Runtime call sites stay in the agent loop and
tool dispatcher, but contract constants, payload normalization, request
rewrites, and execution wrappers live here so plugin authors do not need to
reverse-engineer scattered call sites.
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
API_REQUEST_MIDDLEWARE = "api_request"
API_EXECUTION_MIDDLEWARE = "api_execution"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    API_REQUEST_MIDDLEWARE,
    API_EXECUTION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    """Return hook kwargs with the observer schema version populated."""
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
    """Return middleware kwargs with schema markers populated."""
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def invoke_observer_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a plugin hook through the shared observer contract."""
    from hermes_cli.plugins import invoke_hook

    return invoke_hook(hook_name, **observer_payload(**kwargs))


def apply_tool_request_middleware(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware.

    Middleware may return ``{"args": {...}}`` to replace the effective tool
    arguments. Invalid return values are ignored and middleware exceptions are
    fail-open.
    """
    original_args = deepcopy(args or {})
    current_args = deepcopy(original_args)
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
        current_args = deepcopy(next_args)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_args,
        original_payload=original_args,
        changed=current_args != original_args,
        trace=trace,
    )


def apply_api_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered API request middleware.

    Middleware may return ``{"request": {...}}`` to replace the effective API
    kwargs before Hermes sends them to the provider.
    """
    original_request = deepcopy(request)
    current_request = deepcopy(original_request)
    trace: List[Dict[str, Any]] = []

    for result in _invoke_middleware(
        API_REQUEST_MIDDLEWARE,
        request=current_request,
        original_request=original_request,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_request = result.get("request")
        if not isinstance(next_request, dict):
            continue
        current_request = deepcopy(next_request)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_request,
        original_payload=original_request,
        changed=current_request != original_request,
        trace=trace,
    )


def run_tool_execution_middleware(
    tool_name: str,
    args: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run tool execution through registered execution middleware."""
    callbacks = _get_middleware_callbacks(TOOL_EXECUTION_MIDDLEWARE)
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
    """Run API execution through registered execution middleware."""
    callbacks = _get_middleware_callbacks(API_EXECUTION_MIDDLEWARE)
    return _run_execution_chain(
        API_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        request=request,
        original_request=context.pop("original_request", request),
        **context,
    )


def _invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    from hermes_cli.plugins import invoke_middleware

    return invoke_middleware(kind, **middleware_payload(**kwargs))


def _get_middleware_callbacks(kind: str) -> List[Callable]:
    from hermes_cli.plugins import get_plugin_manager

    return list(get_plugin_manager()._middleware.get(kind, []))


def _run_execution_chain(
    kind: str,
    callbacks: List[Callable],
    terminal_call: Callable[[Any], Any],
    **kwargs: Any,
) -> Any:
    payload_key = "args" if "args" in kwargs else "request"

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]

        def next_call(next_payload: Any = None) -> Any:
            return call_at(index + 1, payload if next_payload is None else next_payload)

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            return callback(**call_kwargs)
        except Exception as exc:
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
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
