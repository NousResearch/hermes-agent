"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
import re
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
PRE_INFERENCE_ROUTE_MIDDLEWARE = "pre_inference_route"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE, LLM_EXECUTION_MIDDLEWARE,
    PRE_INFERENCE_ROUTE_MIDDLEWARE,
}

_ROUTE_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ROUTE_MAX_MODEL_LEN = 199
_ROUTE_KEY_LIKE = ("sk-", "sk_", "bearer ", "api_key=", "api-key=", "ghp_", "gho_", "github_pat_",
                    "nvapi-", "hf_")
_ROUTE_MAX_STRING = 500
_ROUTE_MAX_RECEIPT_ITEMS = 32
_ROUTE_METADATA_LIMITS = {
    "session_id": 256, "task_id": 256, "turn_id": 256, "platform": 64,
    "hermes_profile": 128, "requested_model": 199, "requested_provider": 64,
}
_ROUTE_METADATA_DEFAULT_LIMIT = 256


def _bounded_str(value: Any, limit: int) -> str:
    text = str(value if value is not None else "")
    text = "".join(char for char in text if ord(char) >= 32 and ord(char) != 127)
    return text[:limit]


def _bounded_pre_inference_route_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    bounded: Dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = _bounded_str(key, 64)
        limit = _ROUTE_METADATA_LIMITS.get(str(key), _ROUTE_METADATA_DEFAULT_LIMIT)
        if isinstance(value, str):
            bounded[key_text] = _bounded_str(value, limit)
        elif isinstance(value, bool) or value is None:
            bounded[key_text] = value
        elif isinstance(value, (int, float)):
            bounded[key_text] = value
        else:
            bounded[key_text] = _bounded_str(value, limit)
    return bounded


def _bounded_pre_inference_route_receipt(receipt: Any) -> Dict[str, Any]:
    if not isinstance(receipt, dict):
        return {}
    bounded: Dict[str, Any] = {}
    for key, value in list(receipt.items())[:_ROUTE_MAX_RECEIPT_ITEMS]:
        key_text = _bounded_str(key, 64)
        if isinstance(value, str):
            bounded[key_text] = _bounded_str(_sanitize_pre_inference_route_value(value), _ROUTE_MAX_STRING)
        else:
            bounded[key_text] = _sanitize_pre_inference_route_value(value)
    return bounded


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreInferenceRouteResult:
    """Outcome of the pre-inference selection pass (issue #104 host extension).

    ``status`` is one of:
      * ``"route"``       — build the provider client for ``provider``/``model``.
      * ``"refuse"``      — stop before any provider request is issued.
      * ``"passthrough"`` — proceed exactly as if no pre-inference consumer existed.
    """

    status: str
    provider: Optional[str] = None
    model: Optional[str] = None
    reason: str = ""
    receipt: Dict[str, Any] = field(default_factory=dict)


def _sanitize_pre_inference_route_value(value: Any) -> Any:
    """Defense in depth: metadata delivered to pre-inference consumers may never
    carry credential-like material. Any string carrying a known key shape collapses
    to ``"[redacted]"``."""
    if isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in _ROUTE_KEY_LIKE):
            return "[redacted]"
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_pre_inference_route_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_pre_inference_route_value(v) for v in value]
    return value


def _parse_pre_inference_route_result(result: Any) -> Optional[PreInferenceRouteResult]:
    """Validate a consumer's return value. Malformed results are dropped."""
    if not isinstance(result, dict):
        return None
    status = result.get("status")
    if not isinstance(status, str):
        return None
    status = status.strip().lower()
    reason = _sanitize_pre_inference_route_value(str(result.get("reason") or ""))[:500]
    receipt = _bounded_pre_inference_route_receipt(result.get("receipt"))
    if status == "passthrough":
        return PreInferenceRouteResult(status="passthrough", reason=reason or "passthrough")
    if status == "refuse":
        return PreInferenceRouteResult(
            status="refuse",
            reason=reason or "route refused",
            receipt=receipt,
        )
    if status != "route":
        return None
    provider = result.get("provider")
    model = result.get("model")
    if not isinstance(provider, str) or not isinstance(model, str):
        return None
    provider = provider.strip().lower()
    model = model.strip()
    if not _ROUTE_PROVIDER_RE.match(provider):
        return None
    if not model or len(model) > _ROUTE_MAX_MODEL_LEN:
        return None
    if any(ord(char) < 32 for char in model):
        return None
    if any(fragment in model.lower() for fragment in _ROUTE_KEY_LIKE):
        return None
    return PreInferenceRouteResult(
        status="route", provider=provider, model=model,
        reason=reason or "route selected",
        receipt=receipt,
    )


def apply_pre_inference_route_middleware(metadata: Dict[str, Any], **context: Any) -> Optional["PreInferenceRouteResult"]:
    """Invoke ``pre_inference_route`` consumers with non-secret session metadata.

    ``metadata`` is the ONLY thing a consumer sees — never credentials, clients or
    config objects. Returns ``None`` when no consumer is registered (stock behavior,
    byte-identical to pre-extension Hermes). Otherwise the first *valid* consumer
    result wins; a refusal is absolute and the caller must stop before any provider
    request is issued. If consumers ran but every result was malformed, returns a
    fail-closed ``refuse`` rather than silently proceeding stock.
    """
    from hermes_cli.plugins import has_middleware, invoke_middleware

    if not isinstance(metadata, dict):
        metadata = {}
    if not has_middleware(PRE_INFERENCE_ROUTE_MIDDLEWARE):
        return None
    safe_metadata = _sanitize_pre_inference_route_value(_bounded_pre_inference_route_metadata(dict(metadata)))
    try:
        results = invoke_middleware(PRE_INFERENCE_ROUTE_MIDDLEWARE, **safe_metadata)
    except Exception as exc:
        logger.warning("pre_inference_route consumer failed unexpectedly; refusing", exc_info=True)
        return PreInferenceRouteResult(
            status="refuse",
            reason="pre_inference_route consumer failed unexpectedly; refusing before any provider request",
        )
    malformed = 0
    for result in results:
        parsed = _parse_pre_inference_route_result(result)
        if parsed is not None:
            return parsed
        malformed += 1
    if malformed:
        return PreInferenceRouteResult(
            status="refuse",
            reason="pre_inference_route consumer returned a malformed result; refusing before any provider request",
        )
    # A consumer was registered but produced no usable result at all (e.g. it
    # raised and the delivery layer isolated the failure). That is a failure,
    # not an opt-out — refuse. The only provable opt-out is an explicit
    # 'passthrough' result, which returns above.
    return PreInferenceRouteResult(
        status="refuse",
        reason="pre_inference_route consumer(s) registered but returned no result; refusing before any provider request",
    )


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
