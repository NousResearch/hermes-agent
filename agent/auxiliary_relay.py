"""Relay-backed execution of auxiliary LLM calls: per-call identity, route tracking, wire attempts.

Moved out of ``agent/auxiliary_client.py`` (facade): the logical-call ContextVar and every
helper that reads it live here, so relay bookkeeping has one owner. Facade-resident helpers
(``_create_with_progress``, ``_relay_auxiliary_metadata``, ...) are late-imported inside the
functions: the facade stays the monkeypatch seam tests and production patch against.
"""

import contextlib
import contextvars
import functools
import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional

from agent.sdk_transform_bypass import bypass_chat_sdk_request_transform

logger = logging.getLogger(__name__)


_RELAY_AUX_CALL_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_relay_call", default=None)
)


@contextlib.contextmanager
def _relay_aux_call_scope(args: tuple, kwargs: dict):
    """Bind a fresh relay call context for one auxiliary call; mark it failed on any exception."""
    task = args[0] if args else kwargs.get("task")
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": str(task or "unknown"),
        "request_id": f"aux-{uuid.uuid4().hex}",
        "attempt_count": 0,
        "provider": "",
        "model": "",
        "response_model": None,
        "api_mode": "chat_completions",
    })
    try:
        yield
    except BaseException as exc:
        _fail_relay_auxiliary_call(exc)
        raise
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def _relay_auxiliary_call(callback):
    """Give every physical retry in one auxiliary call a shared Relay identity."""
    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        with _relay_aux_call_scope(args, kwargs):
            return callback(*args, **kwargs)
    return wrapped


def _relay_auxiliary_call_async(callback):
    """Async counterpart to :func:`_relay_auxiliary_call`."""
    @functools.wraps(callback)
    async def wrapped(*args, **kwargs):
        with _relay_aux_call_scope(args, kwargs):
            return await callback(*args, **kwargs)
    return wrapped


def _set_relay_auxiliary_route(
    provider: str | None,
    model: str | None,
    api_mode: str | None,
    *,
    route_callback: Optional[Callable[[str, Optional[str], str], None]] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> None:
    from agent.auxiliary_client import _extract_url_query_params

    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    context["provider"] = str(provider or "auxiliary")
    context["model"] = str(model or "unknown")
    context["response_model"] = None
    context["api_mode"] = str(api_mode or "chat_completions")
    context["route_callback"] = route_callback
    runtime = main_runtime or {}
    runtime_base = str(runtime.get("base_url") or "")
    try:
        runtime_base, _ = _extract_url_query_params(runtime_base)
    except ValueError:
        runtime_base = runtime_base.split("?", 1)[0]
    # Stored query-stripped even though the only reader today compares
    # hostnames — a future logger of this context must not be able to leak
    # credentials some proxies carry as ?key=... (#72636).
    context["main_runtime"] = {
        "provider": str(runtime.get("provider") or ""),
        "base_url": runtime_base,
    }


def _wire_provider_name(
    provider: str | None,
    base_url: str,
    main_runtime: Optional[Dict[str, Any]],
) -> str:
    """Return the concrete provider label for one physical wire attempt."""
    from agent.auxiliary_client import _normalize_aux_provider, base_url_hostname

    raw_provider = str(provider or "").strip()
    wrapped = re.search(r"\(([^()]*)\)$", raw_provider)
    if wrapped:
        raw_provider = wrapped.group(1).strip()
    normalized = _normalize_aux_provider(raw_provider)
    if normalized not in {"", "auto"}:
        return raw_provider or normalized

    runtime = main_runtime or {}
    runtime_provider = str(runtime.get("provider") or "").strip()
    runtime_base = str(runtime.get("base_url") or "").strip()
    if (
        runtime_provider
        and runtime_provider not in {"auto"}
        and runtime_base
        and base_url_hostname(runtime_base) == base_url_hostname(base_url)
    ):
        return runtime_provider

    inferred = ""
    try:
        from agent.model_metadata import _infer_provider_from_url

        inferred = _infer_provider_from_url(base_url) or ""
    except (ImportError, ValueError):
        inferred = ""
    if inferred:
        return inferred

    # An auto-selected HTTP client with an otherwise unknown endpoint is a
    # custom route. Reporting "custom" is more actionable than preserving the
    # config-layer "auto" label, which never went over the wire.
    return "custom" if base_url else "auto"


def _notify_relay_auxiliary_route(
    client: Any,
    kwargs: Dict[str, Any],
    provider: str | None,
) -> None:
    """Publish the latest physical route without affecting request success."""
    from agent.auxiliary_client import _extract_url_query_params

    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    route_callback = context.get("route_callback")
    if not callable(route_callback):
        return

    base_url = str(getattr(client, "base_url", "") or "")
    try:
        clean_base, _ = _extract_url_query_params(base_url)
        route_callback(
            _wire_provider_name(
                provider or context.get("provider"),
                clean_base,
                context.get("main_runtime"),
            ),
            str(kwargs.get("model") or context.get("model") or "") or None,
            clean_base,
        )
    except Exception:
        logger.debug("route_callback error in auxiliary wire attempt", exc_info=True)


def _record_route_info(
    route_info: Optional[Dict[str, str]], provider: Optional[str], model: Optional[str]
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


def _relay_auxiliary_metadata(
    *, provider: str | None = None, api_mode: str | None = None
) -> tuple[str, str, dict[str, Any]] | None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return None
    attempt_count = int(context.get("attempt_count") or 0)
    context["attempt_count"] = attempt_count + 1
    provider_name = str(provider or context.get("provider") or "auxiliary")
    model_name = str(context.get("model") or "unknown")
    return provider_name, model_name, {
        "api_mode": str(api_mode or context.get("api_mode") or "chat_completions"),
        "api_request_id": str(context["request_id"]),
        "call_role": f"auxiliary:{context['task']}",
        "retry_count": attempt_count,
        "auxiliary_task": str(context["task"]),
    }


def _relay_sync_completion(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None,
    api_mode: str | None = None, create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    # Late import: the facade is the monkeypatch seam for these names (tests patch
    # agent.auxiliary_client._relay_auxiliary_metadata / _create_with_progress and
    # expect the relay path to observe the patched binding at call time).
    from agent.auxiliary_client import (
        _create_with_progress,
        _relay_auxiliary_metadata,
        _run_protected_sync_provider_call,
    )
    from agent.auxiliary_wire import prepare_chat_messages

    kwargs = prepare_chat_messages(client, kwargs)
    # The progress hook is installed per TASK, so every attempt (retries, recovery rungs, fallbacks)
    # must stream through _create_with_progress or the compression watchdog sees silence (#98466).
    # Recovery rungs / credential retries keep the task's ``no_progress_timeout`` window; the
    # first-token window uses this attempt's provider (fallbacks name theirs) for its stale timeout.
    relay_context = _RELAY_AUX_CALL_CONTEXT.get() or {}
    task = relay_context.get("task")
    relay_context["stream_provider"] = provider or relay_context.get("provider")
    callback = create or (lambda request: _create_with_progress(client, request, task))
    _notify_relay_auxiliary_route(client, kwargs, provider)
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    # Isolate only the provider callback so the owning thread can unwind its lease/DB
    # transaction on hard cancel without touching the shared client.
    if route is None:
        return _run_protected_sync_provider_call(callback, kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    from agent.auxiliary_hooks import run_with_aux_hooks
    model_name = str(kwargs.get("model") or fallback_model)
    try:
        return run_with_aux_hooks(
            lambda: relay_llm.execute_current(
                kwargs, lambda request: _run_protected_sync_provider_call(callback, request),
                name=provider_name, model_name=model_name, metadata=metadata,
                defer_logical_completion=True,
            ),
            aux_task=str(metadata.get("auxiliary_task") or ""), metadata=metadata, client=client, kwargs=kwargs,
            provider=provider_name, model=model_name, api_mode=str(metadata.get("api_mode") or ""),
        )
    except Exception as exc:
        _note_relay_auxiliary_error(exc)
        raise


async def _relay_async_completion(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None,
    api_mode: str | None = None, create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    from agent.auxiliary_client import _acreate_with_progress, _relay_auxiliary_metadata
    from agent.auxiliary_wire import prepare_chat_messages

    kwargs = prepare_chat_messages(client, kwargs)
    # Async twin of the seam default above (#98466).
    callback = create or (lambda request: _acreate_with_progress(client, request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return await callback(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    from agent.auxiliary_hooks import arun_with_aux_hooks
    model_name = str(kwargs.get("model") or fallback_model)
    try:
        return await arun_with_aux_hooks(
            lambda: relay_llm.execute_current_async(
                kwargs, callback, name=provider_name, model_name=model_name,
                metadata=metadata, defer_logical_completion=True,
            ),
            aux_task=str(metadata.get("auxiliary_task") or ""), metadata=metadata, client=client, kwargs=kwargs,
            provider=provider_name, model=model_name, api_mode=str(metadata.get("api_mode") or ""),
        )
    except Exception as exc:
        _note_relay_auxiliary_error(exc)
        raise


def _relay_sync_stream(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None, api_mode: str | None = None
) -> Any:
    from agent.auxiliary_client import _relay_auxiliary_metadata
    from agent.auxiliary_wire import prepare_chat_messages

    kwargs = prepare_chat_messages(client, kwargs)
    # The bypass runs inside the provider callback, AFTER Relay has seen (and possibly
    # rewritten) the real conversation; applying it to `kwargs` would hand Relay an empty one.
    create = lambda request: client.chat.completions.create(**bypass_chat_sdk_request_transform(request, client))  # noqa: E731
    _notify_relay_auxiliary_route(client, kwargs, provider)
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return create(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    from agent.auxiliary_hooks import run_with_aux_hooks
    model_name = str(kwargs.get("model") or fallback_model)
    return run_with_aux_hooks(
        lambda: relay_llm.stream_current(
            kwargs, create, name=provider_name, model_name=model_name, finalizer=dict,
            metadata=metadata, completed_response_predicate=lambda value: hasattr(value, "choices"),
        ),
        aux_task=str(metadata.get("auxiliary_task") or ""), metadata=metadata, client=client, kwargs=kwargs,
        provider=provider_name, model=model_name, api_mode=str(metadata.get("api_mode") or ""), streaming=True,
    )


def _complete_relay_auxiliary_call(*, outcome: str = "success", error_class: Optional[str] = None) -> None:
    """Close one auxiliary logical call after acceptance or terminal failure. A success
    reports the last attempt error it recovered from (``none`` when the first attempt held)."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    from agent import relay_llm
    relay_llm.complete_logical_call(
        str(context.get("request_id") or ""), outcome=outcome,
        model_name=str(context.get("model") or "unknown"),
        provider_name=str(context.get("provider") or "auxiliary"),
        response_model_name=context.get("response_model"),
        error_class=error_class or context.get("error_class") or "none",
    )


def _note_relay_auxiliary_error(exc: BaseException) -> None:
    """Remember one failed attempt's classified reason on the logical call."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is not None and isinstance(exc, Exception):
        from agent import auxiliary_call_outcome
        context["error_class"] = auxiliary_call_outcome.error_class(
            exc, provider=str(context.get("provider") or ""), model=str(context.get("model") or ""))


def _fail_relay_auxiliary_call(exc: BaseException) -> None:
    """Close a call that ended in ``exc`` without replacing it: a Hermes abort is ``cancelled``,
    anything else ``failed`` with the classifier's reason for the error that ended it."""
    try:
        from agent import auxiliary_call_outcome
        if auxiliary_call_outcome.is_cancellation(exc):
            _complete_relay_auxiliary_call(outcome="cancelled", error_class="none")
            return
        _note_relay_auxiliary_error(exc)
        _complete_relay_auxiliary_call(outcome="failed")
    except Exception:
        logger.warning("Relay auxiliary failure finalization failed", exc_info=True)
