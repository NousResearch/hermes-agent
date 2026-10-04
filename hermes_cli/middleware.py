"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
from copy import copy, deepcopy
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
    """Copy a request graph, tolerating non-deepcopyable members.

    An LLM request can carry clients/callbacks/file handles; a hard ``deepcopy`` failure would
    otherwise abort the whole request-middleware pass. The fallback isolates surrounding
    containers and preserves their aliases and cycles; opaque leaves retain their identity.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:
        logger.debug("deepcopy failed for request payload (%s); retaining opaque leaves", exc)

    # Walk containers without Python recursion: an opaque leaf or a deeply nested
    # request must not share surrounding containers or bypass the middleware pass. Shallow
    # copies retain container subclasses (including defaultdict's factory).
    memo: Dict[int, Any] = {}
    pending: Dict[int, List[Callable[[Any], None]]] = {}
    tasks: List[Callable[[], None]] = []
    root: List[Any] = []

    def copy_state(value: Any, result: Any) -> None:
        state = getattr(value, "__dict__", None)
        if state is not None and hasattr(result, "__dict__"):
            tasks.append(lambda: copy_value(state, lambda copied: setattr(result, "__dict__", copied)))

    def copy_value(value: Any, assign: Callable[[Any], None]) -> None:
        if type(value) in (str, bytes, int, float, bool, complex, type(None)):
            assign(value)
            return
        identity = id(value)
        if identity in memo:
            assign(memo[identity])
            return
        if identity in pending:
            pending[identity].append(assign)
            return
        result: Any
        if isinstance(value, (dict, list, set)):
            factory = dict if isinstance(value, dict) else list if isinstance(value, list) else set
            try:
                result = copy(value)
                if result is value or not isinstance(result, factory):
                    result = factory()
                result.clear()
            except Exception:
                result = factory()
            memo[identity] = result
            assign(result)
            copy_state(value, result)
            if isinstance(result, dict):
                for key, item in reversed(list(value.items())):
                    pair: List[Any] = [None, None]
                    tasks.append(lambda pair=pair, target=result: target.__setitem__(pair[0], pair[1]))
                    tasks.append(lambda item=item, pair=pair: copy_value(
                        item, lambda copied: pair.__setitem__(1, copied)))
                    tasks.append(lambda key=key, pair=pair: copy_value(
                        key, lambda copied: pair.__setitem__(0, copied)))
            elif isinstance(result, list):
                list.extend(result, [None] * len(value))
                for index, item in reversed(list(enumerate(value))):
                    tasks.append(lambda index=index, item=item, target=result: copy_value(
                        item, lambda copied: list.__setitem__(target, index, copied)))
            else:
                for item in value:
                    tasks.append(lambda item=item, target=result: copy_value(
                        item, lambda copied: set.add(target, copied)))
            return
        if isinstance(value, (tuple, frozenset)):
            pending[identity] = [assign]
            items: List[Any] = [None] * len(value)

            def finish() -> None:
                factory = tuple if isinstance(value, tuple) else frozenset
                result = factory.__new__(type(value), items)
                memo[identity] = result
                for setter in pending.pop(identity):
                    setter(result)
                copy_state(value, result)

            # Mutable children are allocated before their contents, so tuple/list
            # cycles can be completed after the tuple itself becomes available.
            tasks.append(finish)
            for index, item in reversed(list(enumerate(value))):
                tasks.append(lambda index=index, item=item: copy_value(
                    item, lambda copied: items.__setitem__(index, copied)))
            return
        # A failing __deepcopy__ may leave partial containers in its memo.
        # Publish that memo only on success, never after an opaque leaf fails.
        leaf_memo = memo.copy()
        try:
            result = deepcopy(value, leaf_memo)
        except Exception:
            result = value
        else:
            memo.update(leaf_memo)
        memo[identity] = result
        assign(result)

    tasks.append(lambda: copy_value(payload, root.append))
    while tasks:
        tasks.pop()()
    return root[0]


def _apply_request_chain(
    kind: str, payload_key: str, trace: List[Dict[str, Any]], original: Any, **kwargs: Any
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

    # Execution-only plugins may rewrite the caller's payload in place, so freeze
    # its original before entering the first frame, then isolate each frame's view.
    original_key = "original_" + payload_key
    kwargs[original_key] = _safe_copy(kwargs[original_key])

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
        call_kwargs[original_key] = _safe_copy(kwargs[original_key])
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
