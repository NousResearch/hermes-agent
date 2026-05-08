"""telemetry — Hermes plugin for local OTEL-style span tracing + memory pods.

Traces LLM calls and tool executions to a local SQLite database
(~/.hermes/telemetry.db) AND writes to Hermes memory pods for persistent
cross-session memory.

No external service required — everything stays on-disk.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Check availability of core telemetry ────────────────────────────────────

_TE: Optional[Any] = None
_TE_LOCK = threading.Lock()

try:
    import sys as _sys

    # Try hermes home scripts FIRST (~/.hermes/scripts), then fallback to
    # bundled scripts/core. Adding the *parent* dir so Python finds the 'core'
    # package within it. This ensures the telemetry plugin always uses
    # the latest core modules regardless of how Hermes was launched.
    import os as _os
    _hermes_scripts = _os.path.expanduser("~/.hermes/scripts")
    _bundled_scripts = _os.path.join(_os.path.dirname(__file__), "..", "..", "scripts")
    for _path in [_hermes_scripts, _bundled_scripts]:
        if _path and _path not in _sys.path:
            _sys.path.insert(0, _path)

    from core import TelemetryEngine, SpanKind, TraceStatus, get_pod, pod_health_report

    _TE_AVAILABLE = True
except Exception as exc:
    logger.debug("telemetry plugin: core not available (%s). Plugin inert.", exc)
    TelemetryEngine = None
    SpanKind = None
    TraceStatus = None
    get_pod = None
    pod_health_report = None
    _TE_AVAILABLE = False


# ── Env config ───────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        if _env(name).lower() in {"1", "true", "yes", "on"}:
            return True
    return False


_ENABLED = _env_bool("HERMES_TELEMETRY_ENABLED", "HERMES_TELEMETRY")
_DEBUG = _env_bool("HERMES_TELEMETRY_DEBUG")

_DB_PATH = _env("HERMES_TELEMETRY_DB", "") or "sqlite+aiosqlite:////home/ubuntu/.hermes/telemetry.db"

# Memory pod integration: write to pods on every hook
_MEM_PODS = _env_bool("HERMES_MEM_PODS", "HERMES_TELEMETRY")


# ── Per-session state ─────────────────────────────────────────────────────────

_session_engines: dict[str, Any] = {}
_SESSION_LOCK = threading.Lock()

# Pending LLM spans keyed by session_id
_pending_llm: dict[str, tuple[str, float]] = {}

# Tool call counts for session summary
_tool_counts: dict[str, int] = {}
_tool_lock = threading.Lock()


def _debug(msg: str, *args) -> None:
    if _DEBUG:
        logger.info("telemetry: " + msg, *args)


# ── Engine lifecycle ────────────────────────────────────────────────────────

def _get_engine(session_id: str, conversation_id: Optional[str] = None):
    if not _TE_AVAILABLE:
        return None

    with _SESSION_LOCK:
        if session_id not in _session_engines:
            te = TelemetryEngine(
                trace_id=session_id,
                conversation_id=conversation_id,
                db_path=_DB_PATH,
                metadata={"source": "hermes-plugin", "session_id": session_id},
            )
            _session_engines[session_id] = te
            _start_trace_bg(te)
        return _session_engines[session_id]


def _start_trace_bg(te):
    def _inner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(te.start_trace())
            loop.close()
        except Exception as exc:
            logger.debug("start_trace failed: %s", exc)

    t = threading.Thread(target=_inner, daemon=True)
    t.start()


# ── Memory Pod writers ───────────────────────────────────────────────────────

def _write_episodic(session_id: str, event_type: str, content: dict) -> None:
    """Write a record to the episodic pod (session events, tool calls, etc)."""
    if not get_pod:
        return
    try:
        pod = get_pod("episodic")
        if pod:
            pod.add({
                "type": event_type,
                "content": content,
                "session_id": session_id,
                "timestamp": time.time(),
            })
    except Exception as exc:
        logger.debug("_write_episodic error: %s", exc)


def _write_entity(entity_type: str, name: str, description: str = "", properties: dict = None) -> None:
    """Write an entity to the entity pod (knowledge graph)."""
    if not get_pod:
        return
    try:
        pod = get_pod("entity")
        if pod:
            pod.add({
                "entity_type": entity_type,
                "name": name,
                "description": description,
                "properties": properties or {},
            })
    except Exception as exc:
        logger.debug("_write_entity error: %s", exc)


def _write_working(session_id: str, key: str, value: Any) -> None:
    """Write a key-value to the working memory pod."""
    if not get_pod:
        return
    try:
        pod = get_pod("working")
        if pod:
            pod.add(
                {"key": key, "value": value},
                session_id=session_id
            )
    except Exception as exc:
        logger.debug("_write_working error: %s", exc)


# ── LLM span helpers ─────────────────────────────────────────────────────────

def _llm_kind() -> str:
    return SpanKind.LLM if hasattr(SpanKind, "LLM") else "llm"


def _tool_kind() -> str:
    return SpanKind.TOOL if hasattr(SpanKind, "TOOL") else "tool"


# ── Hook: pre_llm_call ──────────────────────────────────────────────────────

def on_pre_llm_call(
    *,
    session_id: str = "",
    task_id: str = "",
    tool_name: str = "",
    args: Any = None,
    model: str = "",
    **kwargs,
) -> None:
    if not _ENABLED or not _TE_AVAILABLE:
        return
    if not session_id:
        return
    try:
        te = _get_engine(session_id)
        if te is None:
            return
        span_id = _run_sync(te.start_span(
            name=f"llm:{model or tool_name or 'unknown'}",
            kind=_llm_kind(),
            attributes={"model": model or "", "tool": tool_name, "task_id": task_id},
        ))
        with _SESSION_LOCK:
            _pending_llm[session_id] = (span_id, time.time())

        # Write to memory pods
        if _MEM_PODS:
            _write_episodic(session_id, "llm_call_started", {
                "model": model or "",
                "tool": tool_name,
                "task_id": task_id,
            })

        _debug("pre_llm_call span=%s model=%s", span_id, model)
    except Exception as exc:
        logger.debug("on_pre_llm_call error: %s", exc)


# ── Hook: post_llm_call ──────────────────────────────────────────────────────

def on_post_llm_call(
    *,
    session_id: str = "",
    task_id: str = "",
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    error: Any = None,
    model: str = "",
    **kwargs,
) -> None:
    if not _ENABLED or not _TE_AVAILABLE:
        return
    if not session_id:
        return
    try:
        te = _get_engine(session_id)
        if te is None:
            return
        span_id = None
        with _SESSION_LOCK:
            data = _pending_llm.pop(session_id, None)
            if data:
                span_id = data[0]

        if span_id:
            # Record token usage
            if hasattr(result, "usage") and result.usage is not None:
                try:
                    _run_sync(te.set_attribute(span_id, "prompt_tokens", result.usage.prompt_tokens))
                    _run_sync(te.set_attribute(span_id, "completion_tokens", result.usage.completion_tokens))
                    _run_sync(te.record_metric(span_id, "total_tokens", result.usage.total_tokens))
                except Exception:
                    pass

            # End span
            status = (TraceStatus.ERROR if hasattr(TraceStatus, "ERROR") else "error") if error else (TraceStatus.OK if hasattr(TraceStatus, "OK") else "ok")
            _run_sync(te.end_span(span_id, status=status, error_message=str(error) if error else None))

        # Write to memory pods
        if _MEM_PODS:
            # Summarize response into entity
            response_text = ""
            if hasattr(result, "content") and result.content:
                response_text = str(result.content)[:500]
            elif hasattr(result, "text") and result.text:
                response_text = str(result.text)[:500]
            if response_text:
                _write_entity(
                    entity_type="llm_response",
                    name=f"llm_response_{session_id[:8]}",
                    description=response_text,
                    properties={
                        "model": model or "",
                        "tool": tool_name,
                        "session_id": session_id,
                        "tokens": getattr(getattr(result, "usage", None), "total_tokens", 0) if hasattr(result, "usage") else 0,
                    }
                )
            # Working memory: last LLM response summary
            if response_text:
                _write_working(session_id, "last_llm_response", response_text[:200])

        _debug("post_llm_call span=%s done", span_id)
    except Exception as exc:
        logger.debug("on_post_llm_call error: %s", exc)


# ── Hook: pre_tool_call ──────────────────────────────────────────────────────

def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **kwargs,
) -> None:
    if not _ENABLED or not _TE_AVAILABLE:
        return
    if not session_id:
        return
    try:
        te = _get_engine(session_id)
        if te is None:
            return
        # Build safe args preview
        safe_args = {}
        if isinstance(args, dict):
            safe_args = {k: (str(v)[:200] if len(str(v)) > 200 else str(v)) for k, v in args.items()}
        attrs = {"tool": tool_name, "task_id": task_id, "args_preview": str(safe_args)[:500]}

        span_id = _run_sync(te.start_span(
            name=f"tool:{tool_name}",
            kind=_tool_kind(),
            attributes=attrs,
        ))

        with _SESSION_LOCK:
            _pending_llm[f"tool:{tool_call_id}"] = (span_id, time.time())

            # Tool call count
            _tool_counts[session_id] = _tool_counts.get(session_id, 0) + 1

        # Write to episodic pod
        if _MEM_PODS:
            _write_episodic(session_id, "tool_call", {
                "tool": tool_name,
                "args": safe_args,
                "tool_call_id": tool_call_id,
            })

        _debug("pre_tool_call span=%s tool=%s", span_id, tool_name)
    except Exception as exc:
        logger.debug("on_pre_tool_call error: %s", exc)


# ── Hook: post_tool_call ─────────────────────────────────────────────────────

def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    error: Any = None,
    **kwargs,
) -> None:
    if not _ENABLED or not _TE_AVAILABLE:
        return
    if not session_id:
        return
    try:
        te = _get_engine(session_id)
        if te is None:
            return
        span_id = None
        with _SESSION_LOCK:
            data = _pending_llm.pop(f"tool:{tool_call_id}", None) if tool_call_id else None
            if data:
                span_id = data[0]

        if span_id:
            # Truncate result preview
            result_str = str(result) if result is not None else ""
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            if result is not None:
                _run_sync(te.set_attribute(span_id, "result_preview", result_str))

            status = (TraceStatus.ERROR if hasattr(TraceStatus, "ERROR") else "error") if error else (TraceStatus.OK if hasattr(TraceStatus, "OK") else "ok")
            _run_sync(te.end_span(span_id, status=status, error_message=str(error) if error else None))

        # Write to memory pods
        if _MEM_PODS:
            _write_episodic(session_id, "tool_result", {
                "tool": tool_name,
                "result": str(result)[:500] if result else "",
                "error": str(error) if error else None,
                "tool_call_id": tool_call_id,
            })
            # Update working memory with tool count
            with _tool_lock:
                count = _tool_counts.get(session_id, 0)
            _write_working(session_id, "tool_count", count)
            _write_working(session_id, "last_tool", tool_name)

        _debug("post_tool_call span=%s done", span_id)
    except Exception as exc:
        logger.debug("on_post_tool_call error: %s", exc)


# ── Sync helper ──────────────────────────────────────────────────────────────

def _run_sync(coro):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as exc:
        logger.debug("_run_sync error: %s", exc)
        return None


# ── Plugin registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    if not _ENABLED:
        _debug("Plugin disabled (HERMES_TELEMETRY_ENABLED not set)")
        return
    if not _TE_AVAILABLE:
        logger.warning("telemetry plugin: core TelemetryEngine not available. Install aiosqlite.")
        return

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)

    pod_info = ""
    if _MEM_PODS and get_pod:
        try:
            report = pod_health_report()
            parts = [f"{k}({v['count']})" for k, v in report.items()]
            pod_info = " | pods: " + ", ".join(parts)
        except Exception:
            pass

    logger.info("telemetry plugin: OTEL spans + memory pods enabled → %s%s", _DB_PATH, pod_info)