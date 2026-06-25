"""observeco — Hermes plugin for ObserveCo observability.

Exports telemetry to an ObserveCo OTEL listener via OTLP/HTTP spans.
Covers token usage, session lifecycle, tool calls, errors, and subagents.

Activation:
  hermes plugins enable observability/observeco

Required env vars (set in ~/.hermes/.env):
  HERMES_OBSERVECO_ENDPOINT  — OTLP HTTP endpoint (default: http://127.0.0.1:4318)

Optional env vars:
  HERMES_OBSERVECO_DISABLED  — set "true" to disable export without removing plugin
  HERMES_OBSERVECO_SERVICE   — service.name in resource (default: hermes-agent)

Backward compat:
  HERMES_OTEL_ENDPOINT is used as fallback if HERMES_OBSERVECO_ENDPOINT is not set.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_endpoint() -> str:
    """Return endpoint, with backward-compat fallback to HERMES_OTEL_ENDPOINT."""
    return (
        _env("HERMES_OBSERVECO_ENDPOINT")
        or _env("HERMES_OTEL_ENDPOINT")
        or "http://127.0.0.1:4318"
    ).rstrip("/")


def _get_service_name() -> str:
    return _env("HERMES_OBSERVECO_SERVICE", "hermes-agent")


def _is_disabled() -> bool:
    return _env("HERMES_OBSERVECO_DISABLED", "").lower() in ("true", "1", "yes")


# ── OTLP span builder ──────────────────────────────────────────────────


def _build_otlp_payload(
    *,
    service_name: str,
    span_name: str,
    span_id: str,
    trace_id: str,
    start_time_unix_nano: int,
    end_time_unix_nano: int,
    attributes: dict[str, Any],
    status_code: int = 1,  # 0=UNSET, 1=OK, 2=ERROR
    status_message: str = "",
) -> dict:
    """Build a minimal OTLP/HTTP JSON payload with one span."""
    span_attrs: list[dict] = []
    for k, v in attributes.items():
        if isinstance(v, str):
            span_attrs.append({"key": k, "value": {"stringValue": v}})
        elif isinstance(v, int):
            span_attrs.append({"key": k, "value": {"intValue": v}})
        elif isinstance(v, float):
            span_attrs.append({"key": k, "value": {"doubleValue": v}})

    status = {"code": status_code}
    if status_message:
        status["message"] = status_message

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
                        {"key": "telemetry.sdk.name", "value": {"stringValue": "hermes-observeco-plugin"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "hermes-agent", "version": "1.0.0"},
                        "spans": [
                            {
                                "name": span_name,
                                "spanId": span_id,
                                "traceId": trace_id,
                                "startTimeUnixNano": start_time_unix_nano,
                                "endTimeUnixNano": end_time_unix_nano,
                                "status": status,
                                "attributes": span_attrs,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _send_span(payload: dict, endpoint: str) -> None:
    """Fire-and-forget POST to OTLP HTTP endpoint."""
    import urllib.request
    import urllib.error

    url = f"{endpoint}/v1/traces"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status >= 400:
                logger.warning("ObserveCo export got HTTP %d from %s", resp.status, url)
    except urllib.error.URLError as exc:
        logger.debug("ObserveCo export failed (endpoint may be offline): %s", exc)
    except Exception as exc:
        logger.debug("ObserveCo export error: %s", exc)


# ── Helpers ─────────────────────────────────────────────────────────────


def _now_ns() -> int:
    return int(time.time() * 1_000_000_000)


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def _trace_id() -> str:
    return uuid.uuid4().hex[:32]


def _safe_int(v: Any, default: int = 0) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    try:
        return int(v) if v not in (None, "") else default
    except (ValueError, TypeError):
        return default


# ── Hook: post_api_request (token usage) ───────────────────────────────


def on_post_api_request(
    *,
    task_id: str = "",
    session_id: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    model: str = "",
    api_call_count: int = 0,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    usage: Any = None,
    api_duration: float = 0.0,
    finish_reason: str = "",
    **_: Any,
) -> None:
    """Emit an OTLP span for this API call with real token usage data."""
    if _is_disabled():
        return
    if not isinstance(usage, dict) or not usage:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()

    input_tokens = _safe_int(usage.get("input_tokens", 0))
    output_tokens = _safe_int(usage.get("output_tokens", 0))
    cache_read = _safe_int(usage.get("cache_read_tokens", 0))
    cache_write = _safe_int(usage.get("cache_write_tokens", 0))
    reasoning = _safe_int(usage.get("reasoning_tokens", 0))
    cost = float(usage.get("estimated_cost_usd", 0) or 0)

    if input_tokens == 0 and output_tokens == 0:
        return

    now_ns = _now_ns()
    duration_ns = int(api_duration * 1_000_000_000) if api_duration else 0

    attrs: dict[str, object] = {
        # OpenInference token conventions
        "llm.usage.token_count.prompt": input_tokens,
        "llm.usage.token_count.completion": output_tokens,
        "llm.usage.cache_creation_input_tokens": cache_write,
        "llm.usage.cache_read_input_tokens": cache_read,
        # Provider / model metadata
        "gen_ai.system": provider or "unknown",
        "gen_ai.request.model": model or "unknown",
        "llm.provider": provider or "unknown",
        # Hermes session context
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.api_call_count": api_call_count,
        "hermes.assistant_content_chars": assistant_content_chars,
        "hermes.assistant_tool_call_count": assistant_tool_call_count,
        "hermes.api_duration_ms": int(api_duration * 1000) if api_duration else 0,
        "hermes.finish_reason": finish_reason or "",
        "hermes.cost_usd": round(cost, 8) if cost else 0,
        "hermes.reasoning_tokens": reasoning,
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name=f"llm.{model or 'call'}",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns - duration_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
        status_code=1,
    )

    _send_span(payload, endpoint)


# ── Hook: api_request_error ────────────────────────────────────────────


def on_api_request_error(
    *,
    task_id: str = "",
    session_id: str = "",
    provider: str = "",
    model: str = "",
    error: Any = None,
    api_duration: float = 0.0,
    **_: Any,
) -> None:
    """Emit an OTLP span for a failed API request."""
    if _is_disabled():
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    error_str = str(error) if error is not None else "unknown"

    attrs: dict[str, Any] = {
        "gen_ai.system": provider or "unknown",
        "gen_ai.request.model": model or "unknown",
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.error": error_str,
        "hermes.api_duration_ms": int(api_duration * 1000) if api_duration else 0,
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name=f"error.llm.{model or 'call'}",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
        status_code=2,  # ERROR
        status_message=error_str[:200],
    )

    _send_span(payload, endpoint)


# ── Hook: on_session_start ───────────────────────────────────────────────


def on_session_start(
    *,
    session_id: str = "",
    task_id: str = "",
    model: str = "",
    provider: str = "",
    **_: Any,
) -> None:
    """Emit a span marking the start of a Hermes session."""
    if _is_disabled():
        return
    if not session_id:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id,
        "hermes.task_id": task_id or "",
        "hermes.event": "session_start",
        "gen_ai.request.model": model or "",
        "gen_ai.system": provider or "",
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="session.start",
        span_id=_span_id(),
        trace_id=session_id.replace("-", "")[:32] or _trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: on_session_end ───────────────────────────────────────────────


def on_session_end(
    *,
    session_id: str = "",
    task_id: str = "",
    model: str = "",
    provider: str = "",
    **_: Any,
) -> None:
    """Emit a span marking the end of a Hermes session."""
    if _is_disabled():
        return
    if not session_id:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id,
        "hermes.task_id": task_id or "",
        "hermes.event": "session_end",
        "gen_ai.request.model": model or "",
        "gen_ai.system": provider or "",
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="session.end",
        span_id=_span_id(),
        trace_id=session_id.replace("-", "")[:32] or _trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: on_session_finalize ──────────────────────────────────────────


def on_session_finalize(
    *,
    session_id: str = "",
    task_id: str = "",
    **_: Any,
) -> None:
    """Emit a span when a session is finalized (resources cleaned up)."""
    if _is_disabled():
        return
    if not session_id:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id,
        "hermes.task_id": task_id or "",
        "hermes.event": "session_finalize",
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="session.finalize",
        span_id=_span_id(),
        trace_id=session_id.replace("-", "")[:32] or _trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: pre_tool_call ─────────────────────────────────────────────────


def on_pre_tool_call(
    *,
    tool_name: str = "",
    tool_args: Any = None,
    session_id: str = "",
    task_id: str = "",
    **_: Any,
) -> None:
    """Emit a span when a tool is about to be called."""
    if _is_disabled():
        return
    if not tool_name:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.tool_name": tool_name,
        "hermes.event": "tool_start",
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name=f"tool.{tool_name}",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: post_tool_call ────────────────────────────────────────────────


def on_post_tool_call(
    *,
    tool_name: str = "",
    tool_args: Any = None,
    result: Any = None,
    session_id: str = "",
    task_id: str = "",
    **_: Any,
) -> None:
    """Emit a span when a tool call completes."""
    if _is_disabled():
        return
    if not tool_name:
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    # Summarise result — don't dump full payload
    result_summary = ""
    if isinstance(result, dict):
        if "error" in result:
            result_summary = f"error: {str(result['error'])[:100]}"
        elif "output" in result:
            result_summary = f"ok ({len(str(result['output']))} chars)"
        else:
            result_summary = f"ok ({len(json.dumps(result))} chars)"
    elif isinstance(result, str):
        result_summary = f"ok ({len(result)} chars)"
    elif result is None:
        result_summary = "ok (empty)"
    else:
        result_summary = f"ok ({type(result).__name__})"

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.tool_name": tool_name,
        "hermes.event": "tool_end",
        "hermes.tool_result": result_summary,
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name=f"tool.{tool_name}",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: subagent_start ────────────────────────────────────────────────


def on_subagent_start(
    *,
    session_id: str = "",
    task_id: str = "",
    parent_session_id: str = "",
    goal: str = "",
    **_: Any,
) -> None:
    """Emit a span when a subagent is spawned."""
    if _is_disabled():
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.parent_session_id": parent_session_id or "",
        "hermes.event": "subagent_start",
        "hermes.subagent_goal": (goal or "")[:200],
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="subagent.start",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: subagent_stop ────────────────────────────────────────────────


def on_subagent_stop(
    *,
    session_id: str = "",
    task_id: str = "",
    parent_session_id: str = "",
    **_: Any,
) -> None:
    """Emit a span when a subagent completes."""
    if _is_disabled():
        return

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    attrs: dict[str, Any] = {
        "hermes.session_id": session_id or "",
        "hermes.task_id": task_id or "",
        "hermes.parent_session_id": parent_session_id or "",
        "hermes.event": "subagent_stop",
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="subagent.stop",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)


# ── Hook: pre_gateway_dispatch ──────────────────────────────────────────


def on_pre_gateway_dispatch(
    *,
    event: Any = None,
    gateway: Any = None,
    session_store: Any = None,
    **_: Any,
) -> dict | None:
    """Monitor gateway dispatch events.

    This hook fires for every incoming message. It emits a span with
    routing info but does NOT modify dispatch behaviour (returns None).
    """
    if _is_disabled():
        return None

    endpoint = _get_endpoint()
    service_name = _get_service_name()
    now_ns = _now_ns()

    # Extract safe metadata from the event
    platform = ""
    chat_id = ""
    user_id = ""
    text_preview = ""
    topic_id = ""
    if event is not None:
        platform = getattr(event, "platform", "") or ""
        chat_id = str(getattr(event, "chat_id", "") or "")
        user_id = str(getattr(event, "user_id", "") or "")
        text = getattr(event, "text", "") or ""
        text_preview = text[:100]
        topic_id = str(getattr(event, "topic_id", "") or "")

    attrs: dict[str, Any] = {
        "hermes.event": "gateway_dispatch",
        "hermes.platform": platform,
        "hermes.chat_id": chat_id,
        "hermes.user_id": user_id,
        "hermes.topic_id": topic_id,
        "hermes.text_preview": text_preview,
    }

    payload = _build_otlp_payload(
        service_name=service_name,
        span_name="gateway.dispatch",
        span_id=_span_id(),
        trace_id=_trace_id(),
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns,
        attributes=attrs,
    )

    _send_span(payload, endpoint)
    return None  # Do not modify dispatch


# ── Plugin registration ─────────────────────────────────────────────────


def register(ctx) -> None:
    """Register all ObserveCo hooks."""
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
    # Legacy hook name variants for backward compat
    ctx.register_hook("post_llm_call", on_post_api_request)
    logger.info(
        "ObserveCo plugin registered — will export spans to %s",
        _get_endpoint(),
    )
