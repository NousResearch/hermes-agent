"""Telemetry plugin — wires Hermes lifecycle hooks to the local telemetry emitter.

This is the *only* instrumentation seam. It registers observational hooks (which core
already invokes fail-open) and translates each into a typed local telemetry event
handed to ``agent.telemetry.emitter``. There are zero edits to core call sites:
the hooks already carry model/provider/usage/duration/tool data.

Everything here is best-effort and fail-open — a raised exception in a hook callback is
swallowed by core, and we additionally guard each callback so a telemetry bug can never
disturb a session. No content, no network: local telemetry only.

Hooks consumed:
  on_session_start    -> begin a run context (trace_id/run_id + root span id)
  post_api_request    -> one model_call event + its timing span (tokens, latency)
  api_request_error   -> one error event
  post_tool_call      -> one tool_call event + its timing span (duration, result class)
  on_session_finalize -> finalize the run row + emit the run's root span
  subagent_start/stop -> (reserved) lineage markers

Each model/tool call emits a SpanEvent (timing + parent = the run's root span) keyed by
the same span_id as its detail row, so tel_spans reconstructs a run -> calls trace tree.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Per-run accumulators keyed by run_id, so on_session_finalize can roll up counts.
_runs: Dict[str, Dict[str, Any]] = {}
_runs_lock = threading.Lock()


def _safe(fn):
    """Decorator: never let a telemetry hook raise into core."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.debug("telemetry hook %s failed", getattr(fn, "__name__", "?"), exc_info=True)
            return None
    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    return wrapper


def _run_key(session_id: Optional[str], task_id: Optional[str]) -> str:
    return session_id or task_id or "default"


# ── on_session_start ────────────────────────────────────────────────────────
@_safe
def _on_session_start(**kw: Any) -> None:
    from agent.telemetry import spans

    session_id = kw.get("session_id") or ""
    platform = kw.get("platform") or kw.get("source") or ""
    ctx = spans.start_run()
    now = time.time_ns()
    root_span_id = spans.new_span_id()
    key = _run_key(session_id, kw.get("task_id"))
    with _runs_lock:
        _runs[key] = {
            "run_id": ctx.run_id,
            "trace_id": ctx.trace_id,
            "root_span_id": root_span_id,
            "session_id": session_id or None,
            "entrypoint": _entrypoint_for(platform, kw.get("source")),
            "platform": platform or None,
            "start_ns": now,
            "model_call_count": 0,
            "tool_call_count": 0,
            "error_count": 0,
        }


def _ensure_run(session_id: Optional[str], task_id: Optional[str], platform: str = "") -> Dict[str, Any]:
    """Return the run accumulator, lazily creating one if session_start was missed."""
    from agent.telemetry import spans

    key = _run_key(session_id, task_id)
    with _runs_lock:
        run = _runs.get(key)
        if run is None:
            rid = spans.current_run_id() or spans.new_id()
            tid = spans.current_trace_id() or spans.new_id()
            run = {
                "run_id": rid,
                "trace_id": tid,
                "root_span_id": spans.new_span_id(),
                "session_id": session_id or None,
                "entrypoint": _entrypoint_for(platform),
                "platform": platform or None,
                "start_ns": time.time_ns(),
                "model_call_count": 0,
                "tool_call_count": 0,
                "error_count": 0,
            }
            _runs[key] = run
        return run


def _emit_call_span(run: Dict[str, Any], span_id: str, name: str, kind: str,
                    duration_ms: Optional[int], status: Optional[str]) -> None:
    """Emit the timing/lineage span for a model or tool call.

    The call hooks fire on completion, so end_ns is ~now and start_ns is reconstructed
    from the measured duration (end - duration). The span is parented to the run's root
    so a 2-level run -> calls waterfall can be reconstructed from tel_spans.
    """
    from agent.telemetry import emitter
    from agent.telemetry.events import SpanEvent

    end_ns = time.time_ns()
    dur_ns = int(duration_ms) * 1_000_000 if isinstance(duration_ms, (int, float)) else 0
    start_ns = end_ns - dur_ns
    emitter.emit(SpanEvent(
        span_id=span_id,
        trace_id=run["trace_id"],
        run_id=run["run_id"],
        parent_span_id=run.get("root_span_id"),
        name=name,
        kind=kind,
        start_ns=start_ns,
        end_ns=end_ns,
        status=status,
    ))


def _entrypoint_for(platform: Optional[str], source: Optional[str] = None) -> str:
    """Coarse entrypoint label (cli / gateway / tui / api / cron …).

    This is a workflow *surface* label, not model/tool anonymization — it answers
    "where did this run come from", which is genuinely categorical.
    """
    s = (source or platform or "").lower()
    if s in ("", "chat", "interactive", "cli", "desktop"):
        return "cli"
    if s in ("telegram", "discord", "slack", "whatsapp", "signal", "matrix",
             "email", "sms", "teams", "feishu", "wecom", "line", "google_chat"):
        return "gateway"
    if s == "tui":
        return "tui"
    if s in ("api", "api_server", "openai_api"):
        return "api"
    if s == "cron":
        return "cron"
    if s == "batch":
        return "batch"
    if s == "acp":
        return "acp"
    return "cli"


# ── post_api_request -> model_call event ────────────────────────────────────
@_safe
def _on_post_api_request(**kw: Any) -> None:
    from agent.telemetry import emitter, spans
    from agent.telemetry.events import ModelCallEvent

    session_id = kw.get("session_id") or ""
    platform = kw.get("platform") or ""
    run = _ensure_run(session_id, kw.get("task_id"), platform)

    usage = kw.get("usage") or {}
    duration = kw.get("api_duration")
    latency_ms = int(duration * 1000) if isinstance(duration, (int, float)) else None

    span_id = spans.new_span_id()
    evt = ModelCallEvent(
        span_id=span_id,
        run_id=run["run_id"],
        provider=kw.get("provider"),       # raw
        model=kw.get("model"),             # raw
        base_url=kw.get("base_url"),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        latency_ms=latency_ms,
        end_reason="completed",
    )
    with _runs_lock:
        run["model_call_count"] += 1
    _emit_call_span(run, span_id, name=kw.get("model") or "model_call",
                    kind="model", duration_ms=latency_ms, status="ok")
    emitter.emit(evt)


# ── api_request_error -> error event ────────────────────────────────────────
@_safe
def _on_api_request_error(**kw: Any) -> None:
    from agent.telemetry import emitter
    from agent.telemetry.events import ErrorEvent

    session_id = kw.get("session_id") or ""
    run = _ensure_run(session_id, kw.get("task_id"), kw.get("platform") or "")
    error_class = _coarse_error_class(kw.get("error_type") or kw.get("error") or "")
    with _runs_lock:
        run["error_count"] += 1
    emitter.emit(ErrorEvent(
        run_id=run["run_id"],
        error_class=error_class,
        subsystem="model_api",
        recovery=None,
    ))


def _coarse_error_class(raw: Any) -> str:
    s = str(raw).lower()
    if "timeout" in s:
        return "provider_timeout"
    if "rate" in s and "limit" in s:
        return "rate_limit"
    if any(k in s for k in ("auth", "401", "403", "unauthorized", "forbidden")):
        return "auth"
    if any(k in s for k in ("connection", "network", "dns", "socket")):
        return "network"
    if "context" in s and ("length" in s or "overflow" in s or "token" in s):
        return "context_overflow"
    if any(k in s for k in ("500", "502", "503", "server error", "provider")):
        return "provider_error"
    return "unknown"


# ── post_tool_call -> tool_call event ───────────────────────────────────────
@_safe
def _on_post_tool_call(**kw: Any) -> None:
    from agent.telemetry import emitter, spans
    from agent.telemetry.events import ToolCallEvent

    session_id = kw.get("session_id") or ""
    run = _ensure_run(session_id, kw.get("task_id"), kw.get("platform") or "")

    function_name = kw.get("function_name") or kw.get("tool_name")
    duration_ms = kw.get("duration_ms")
    result = kw.get("result")
    result_class = _tool_result_class(result)

    with _runs_lock:
        run["tool_call_count"] += 1
        if result_class == "error":
            run["error_count"] += 1

    dur_int = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
    span_id = spans.new_span_id()
    _emit_call_span(run, span_id, name=function_name or "tool_call",
                    kind="tool", duration_ms=dur_int, status=result_class)
    emitter.emit(ToolCallEvent(
        span_id=span_id,
        run_id=run["run_id"],
        tool_name=function_name,             # raw tool name
        duration_ms=dur_int,
        result_class=result_class,
    ))


def _tool_result_class(result: Any) -> str:
    """Classify a tool result without retaining content — error vs ok vs blocked."""
    try:
        import json
        if isinstance(result, str):
            r = result.strip()
            if r.startswith("{"):
                obj = json.loads(r)
                if isinstance(obj, dict):
                    if obj.get("error") or obj.get("blocked"):
                        return "blocked" if obj.get("blocked") else "error"
                    if obj.get("timeout"):
                        return "timeout"
            return "ok"
        if isinstance(result, dict):
            if result.get("error"):
                return "error"
            return "ok"
    except Exception:
        return "ok"
    return "ok"


# ── on_session_finalize -> finalize the run row ─────────────────────────────
@_safe
def _on_session_finalize(**kw: Any) -> None:
    from agent.telemetry import emitter, spans
    from agent.telemetry.events import RunEvent, SpanEvent

    session_id = kw.get("session_id") or ""
    key = _run_key(session_id, kw.get("task_id"))
    with _runs_lock:
        run = _runs.pop(key, None)
    if run is None:
        run = _ensure_run(session_id, kw.get("task_id"), kw.get("platform") or "")
        with _runs_lock:
            _runs.pop(key, None)

    end_ns = time.time_ns()
    start_ns = run.get("start_ns", end_ns)
    end_reason = _coarse_end_reason(kw)
    # Root span for the run — the trace root the call spans hang off of.
    emitter.emit(SpanEvent(
        span_id=run.get("root_span_id") or spans.new_span_id(),
        trace_id=run["trace_id"],
        run_id=run["run_id"],
        parent_span_id=None,
        name=f"run:{run.get('entrypoint', 'cli')}",
        kind="run",
        start_ns=start_ns,
        end_ns=end_ns,
        status=end_reason,
    ))
    emitter.emit(RunEvent(
        run_id=run["run_id"],
        trace_id=run["trace_id"],
        entrypoint=run.get("entrypoint", "cli"),
        session_id=run.get("session_id"),
        platform=run.get("platform"),
        start_ns=start_ns,
        end_ns=end_ns,
        end_reason=end_reason,
        model_call_count=run.get("model_call_count", 0),
        tool_call_count=run.get("tool_call_count", 0),
        error_count=run.get("error_count", 0),
        estimated_cost_usd=_as_float(kw.get("estimated_cost_usd")),
        cost_status=kw.get("cost_status"),
    ))
    spans.clear_run()


def _coarse_end_reason(kw: Dict[str, Any]) -> str:
    if kw.get("interrupted"):
        return "interrupted"
    if kw.get("failed"):
        return "failed"
    reason = str(kw.get("turn_exit_reason") or "").lower()
    if "max_iteration" in reason:
        return "max_iterations"
    if "timeout" in reason:
        return "timeout"
    return "completed"


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── subagent lineage (reserved) ─────────────────────────────────────────────
# A delegated subagent runs its own ``run_conversation`` with its own session id, so
# its model/tool calls are already captured as a separate tel_runs row via the normal
# hooks — no subagent activity is lost. These hooks fire with the parent<->child bridge
# (parent_session_id, child_session_id, child_role, child_goal); they are reserved for
# recording parent->child *lineage* (linking a child run back to its parent), which
# needs a tel_runs.parent_run_id column. Deferred until a consumer needs the delegation
# tree; left registered as the attachment point.
@_safe
def _on_subagent_start(**kw: Any) -> None:
    return None


@_safe
def _on_subagent_stop(**kw: Any) -> None:
    return None


# ── registration ────────────────────────────────────────────────────────────
def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    logger.debug("telemetry plugin registered 7 hooks")
