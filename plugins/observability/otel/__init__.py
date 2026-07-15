"""OpenTelemetry OTLP trace export for Hermes observer trajectories."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .config import load_config
from .exporter import create_runtime
from .span_builder import SpanBuilder


logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_RUNTIME: Any = None
_BUILDER: SpanBuilder | None = None
_INIT_FAILED = object()


def _get_builder() -> SpanBuilder | None:
    global _RUNTIME, _BUILDER
    with _LOCK:
        if _RUNTIME is _INIT_FAILED:
            return None
        if _BUILDER is not None:
            return _BUILDER

        config = load_config()
        if not config.enabled:
            _RUNTIME = _INIT_FAILED
            return None
        try:
            _RUNTIME = create_runtime(config)
            _BUILDER = SpanBuilder(_RUNTIME.tracer)
        except Exception:
            logger.debug("OpenTelemetry observer initialization failed", exc_info=True)
            _RUNTIME = _INIT_FAILED
            _BUILDER = None
        return _BUILDER


def _safe(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        logger.debug("OpenTelemetry observer hook failed", exc_info=True)


def _dispatch(method: str, kwargs: dict[str, Any]) -> None:
    builder = _get_builder()
    if builder is not None:
        _safe(lambda: getattr(builder, method)(kwargs))


def register(ctx: Any) -> None:
    """Register the complete Hermes observer lifecycle."""
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)


def on_session_start(**kwargs: Any) -> None:
    _dispatch("start_session", kwargs)


def on_session_end(**kwargs: Any) -> None:
    _dispatch("mark_session_end", kwargs)


def on_session_finalize(**kwargs: Any) -> None:
    global _RUNTIME, _BUILDER
    with _LOCK:
        runtime = _RUNTIME
        builder = _BUILDER
        remaining_sessions = 0
        if builder is not None:
            try:
                remaining_sessions = builder.finalize_session(kwargs)
            except Exception:
                logger.debug("OpenTelemetry session finalization failed", exc_info=True)
        if runtime not in (None, _INIT_FAILED) and remaining_sessions == 0:
            _safe(runtime.provider.shutdown)
            _RUNTIME = None
            _BUILDER = None


def on_session_reset(**kwargs: Any) -> None:
    _dispatch("finalize_session", kwargs)


def on_pre_llm_call(**kwargs: Any) -> None:
    _dispatch("start_turn", kwargs)


def on_post_llm_call(**kwargs: Any) -> None:
    _dispatch("end_turn", kwargs)


def on_pre_api_request(**kwargs: Any) -> None:
    _dispatch("start_llm_request", kwargs)


def on_post_api_request(**kwargs: Any) -> None:
    builder = _get_builder()
    if builder is not None:
        _safe(lambda: builder.end_llm_request(kwargs))


def on_api_request_error(**kwargs: Any) -> None:
    builder = _get_builder()
    if builder is not None:
        _safe(lambda: builder.end_llm_request(kwargs, error=True))


def on_pre_tool_call(**kwargs: Any) -> None:
    _dispatch("start_tool", kwargs)


def on_post_tool_call(**kwargs: Any) -> None:
    _dispatch("end_tool", kwargs)


def on_subagent_start(**kwargs: Any) -> None:
    _dispatch("start_subagent", kwargs)


def on_subagent_stop(**kwargs: Any) -> None:
    _dispatch("end_subagent", kwargs)


def on_pre_approval_request(**kwargs: Any) -> None:
    _dispatch("start_approval", kwargs)


def on_post_approval_response(**kwargs: Any) -> None:
    _dispatch("end_approval", kwargs)
