"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
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

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE, LLM_EXECUTION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallAuthorization:
    """Hermes's decision for the tool call a ``tool_execution`` middleware is handling (``authorize_tool_call``).

    ``allowed``: the call may run. ``args``: the arguments it may run with (after any ``pre_tool_call`` ``modify``
    hook). ``result``: when refused, the result the model must receive; its terminal ``post_tool_call`` was already
    emitted, so the middleware returns it unchanged and does not run the call.
    """

    allowed: bool
    args: dict[str, Any]
    result: Any = None


_TOOL_CALL_AUTHORIZER: ContextVar[Optional[Callable[[Optional[dict[str, Any]]], ToolCallAuthorization]]] = ContextVar(
    "hermes_tool_call_authorizer", default=None)


def authorize_tool_call(args: Optional[dict[str, Any]] = None) -> ToolCallAuthorization:
    """Run Hermes's authorization of the current tool call without running the call.

    For a ``tool_execution`` middleware that runs the call somewhere else (another machine, a sandbox) and so never
    calls ``next_call``. In the agent loop (``authorization="pending"``) this runs Hermes's policy once: the hosted-scope
    refusal, the tool-scope block, ``pre_tool_call`` hooks (block / approve / modify), the pruned-argument guard and the
    loop guardrails. Where Hermes already authorized the call before the middleware (``authorization="done"``:
    ``handle_function_call``) it returns that decision. Repeated calls return the first decision. ``args``: the arguments
    the middleware intends to run (default: those the middleware chain started with).

    A result the middleware returns without calling ``next_call`` gets the same terminal ``post_tool_call`` and
    ``transform_tool_result`` as a local result. If the middleware calls ``next_call`` after authorizing, the decision
    is reused (policy does not run twice) and only the authorized arguments may run.

    Raises ``RuntimeError`` outside a ``tool_execution`` middleware call.
    """
    authorizer = _TOOL_CALL_AUTHORIZER.get()
    if authorizer is None:
        raise RuntimeError("authorize_tool_call() works only inside a tool_execution middleware call")
    return authorizer(args)


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
    tool_name: str, args: dict[str, Any], *, original_args: dict[str, Any] | None = None, **context: Any
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware; ``{"args": {...}}`` replaces the effective tool
    arguments before hooks, guardrails, approvals, and execution see them. ``original_args`` is the
    model's pre-Relay call when ``args`` already carries Relay rewrites (defaults to ``args``)."""
    original_args = _safe_copy(args if original_args is None else original_args)
    current_args = _safe_copy(args)
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


def run_tool_execution_middleware(
    tool_name: str, args: dict[str, Any], next_call: Callable[[dict[str, Any]], Any], *,
    authorize: Optional[Callable[[dict[str, Any]], ToolCallAuthorization]] = None, **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware.

    ``authorize``: the caller's authorization of this call when it has not run yet (the agent loop runs it inside
    ``next_call``); middleware reach it through ``authorize_tool_call`` and see ``authorization="pending"``. Without
    it, Hermes authorized the call before the middleware (``authorization="done"``).
    """
    def _already_authorized(requested: Optional[dict[str, Any]]) -> ToolCallAuthorization:
        if requested is not None and requested != args:
            raise RuntimeError("authorize_tool_call(): this tool call was authorized with the arguments it arrived "
                               "with; other arguments cannot be authorized here")
        return ToolCallAuthorization(allowed=True, args=dict(args))

    if authorize is None:
        authorization, authorizer = "done", _already_authorized
    else:
        authorization = "pending"

        def authorizer(requested: Optional[dict[str, Any]]) -> ToolCallAuthorization:
            return authorize(args if requested is None else requested)
    token = _TOOL_CALL_AUTHORIZER.set(authorizer)
    try:
        return _run_execution_chain(
            TOOL_EXECUTION_MIDDLEWARE, next_call,
            tool_name=tool_name, args=args, original_args=context.pop("original_args", args),
            authorization=authorization, **context)
    finally:
        _TOOL_CALL_AUTHORIZER.reset(token)


class _DownstreamExecutionError(Exception):
    """Marks an exception raised BELOW a middleware frame so the frame's own failure handling
    (skip-and-continue) doesn't swallow it."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _run_execution_chain(kind: str, terminal_call: Callable[[Any], Any], **kwargs: Any) -> Any:
    from hermes_cli.plugins import _delivery_manager
    from hermes_cli.plugins_dispatch import PluginDispatchMixin

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
            return callback(**PluginDispatchMixin._hook_callback_kwargs(callback, call_kwargs))  # narrow signatures, as hooks
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
