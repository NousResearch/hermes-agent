"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

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


class LLMExecutionBlocked(Exception):
    """Raised by ``llm_execution`` middleware to unconditionally prevent an LLM
    API call from proceeding.

    Plain ``Exception`` subclasses raised without calling ``next_call`` are
    caught by ``_run_execution_chain``'s general fallthrough handler and
    silently routed to the next middleware (or the terminal call) — the
    correct behavior for an unrelated failure, but it defeats an intentional
    block. ``_run_execution_chain`` re-raises ``LLMExecutionBlocked``
    explicitly before that fallthrough, so raising it from any middleware in
    the chain — directly, or from code called via ``next_call``'s downstream
    execution — reliably propagates to the caller of
    ``run_llm_execution_middleware`` instead of being swallowed.

    Args:
        reason: Human-readable explanation for the block.
        metadata: Optional structured data passed to the caller. By
            convention (see docs/plugins/hook-taxonomy.md), deny-path
            metadata may carry:

            - ``checked_by``: the plugin/middleware that made the block
              decision. If omitted, ``_run_execution_chain`` backfills it
              from the raising callback's own registered name — a plugin
              is never required to set this itself, so existing raises
              don't need updating.
            - ``decision``: e.g. ``"block"``.
            - ``chain``: optional list of middleware names that ran before
              the block, in order.

            Pure observer hooks are not part of this envelope.

    Example::

        def budget_guard(request, next_call, session_id, **ctx):
            if cost_exceeded(session_id):
                raise LLMExecutionBlocked(
                    "Session cost budget exceeded",
                    metadata={"budget_usd": 5.0, "session_id": session_id},
                )
            return next_call(request)
    """

    def __init__(self, reason: str, *, metadata: Dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
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
    kind: str, payload_key: str, trace: List[Dict[str, Any]], original: Any, **kwargs: Any
) -> RequestMiddlewareResult:
    """Feed ``kwargs[payload_key]`` through every ``kind`` middleware; each may return ``{payload_key: {...}}``."""
    from hermes_cli.plugins import invoke_middleware

    current = kwargs[payload_key]
    for result in invoke_middleware(kind, **middleware_payload(**kwargs)):
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


def apply_llm_request_middleware(request: Dict[str, Any], **context: Any) -> RequestMiddlewareResult:
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
    tool_name: str, args: Dict[str, Any], **context: Any
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware; ``{"args": {...}}`` replaces the effective tool
    arguments before hooks, guardrails, approvals, and execution see them."""
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: List[Dict[str, Any]] = []

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
    request: Dict[str, Any], next_call: Callable[[Dict[str, Any]], Any], **context: Any) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE, next_call,
        request=request, original_request=context.pop("original_request", request), **context)


def run_tool_execution_middleware(
    tool_name: str, args: Dict[str, Any], next_call: Callable[[Dict[str, Any]], Any], **context: Any,
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
        except LLMExecutionBlocked as exc:
            # Explicit re-raise before the general fallthrough below — an
            # intentional block must not be treated like an unrelated
            # middleware failure and routed to the next callback in the chain.
            # Backfill checked_by from the raising callback's own registered
            # name if the plugin didn't set it — required on the deny-path
            # per the checked_by envelope convention, but never fail closed
            # on a missing key; the host fills it in instead.
            exc.metadata.setdefault(
                "checked_by", getattr(callback, "__name__", repr(callback))
            )
            raise
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
