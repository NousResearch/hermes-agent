"""Fail-open Laminar trace export helpers for Hermes agent-system events."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.request
from typing import Any, Dict, Optional


SECRET_KEY_RE = re.compile(r"(secret|token|api[_-]?key|authorization|credential|password)", re.IGNORECASE)
MAX_ATTR_CHARS = 512


def laminar_export_enabled() -> bool:
    return str(os.getenv("HERMES_LAMINAR_EXPORT_ENABLED") or "").lower() in {"1", "true", "yes", "on"}


def export_subagent_event(event: Dict[str, Any], *, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Export one subagent event as a small OTLP/JSON span. Never raises."""

    if not laminar_export_enabled():
        return {"ok": False, "status": "disabled"}
    sample_rate = _env_float("HERMES_LAMINAR_SAMPLE_RATE", 1.0)
    if sample_rate < 1.0 and random.random() > max(sample_rate, 0.0):
        return {"ok": False, "status": "sampled_out"}
    try:
        attrs = trace_attributes_for_event(event)
        target = (endpoint or os.getenv("HERMES_LAMINAR_OTLP_HTTP_ENDPOINT") or "http://127.0.0.1:8000/v1/traces").rstrip("/")
        body = json.dumps(_otlp_payload(attrs, event)).encode("utf-8")
        request = urllib.request.Request(
            target,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=_env_float("HERMES_LAMINAR_EXPORT_TIMEOUT_SECONDS", 2.0)):
            pass
        return {"ok": True, "status": "exported", "attributes": attrs}
    except Exception as exc:
        return {"ok": False, "status": "failed_open", "warning": str(exc)}


def trace_attributes_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
    attrs = {
        "hermes_signal_domain": "agent-system",
        "plan_id": event.get("launch_plan_id") or event.get("plan_id"),
        "task_id": event.get("launch_task_id") or event.get("task_id"),
        "session_id": event.get("session_id") or event.get("ao_session_id") or event.get("runtime_session_id"),
        "runtime": event.get("runtime"),
        "status": event.get("status"),
        "verification_verdict": event.get("verification_verdict") or event.get("verification_status"),
        "worker_confidence": event.get("worker_confidence"),
        "output_contract_score": event.get("output_contract_score"),
        "cost_usd": event.get("cost_usd"),
        "duration_seconds": event.get("duration_seconds"),
        "cluster_key": _cluster_key_hint(event),
        "cluster_title": _cluster_title_hint(event),
    }
    return redact_attributes(attrs)


def redact_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if SECRET_KEY_RE.search(str(key)):
            redacted[key] = "[REDACTED]"
            continue
        if isinstance(value, (int, float, bool)):
            redacted[key] = value
            continue
        text = str(value)
        if SECRET_KEY_RE.search(text):
            text = SECRET_KEY_RE.sub("[REDACTED]", text)
        redacted[key] = text[:MAX_ATTR_CHARS]
    return redacted


def _otlp_payload(attrs: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    now_ns = int(float(event.get("created_at") or time.time()) * 1_000_000_000)
    trace_id = hashlib.sha256(str(attrs.get("session_id") or now_ns).encode("utf-8")).hexdigest()[:32]
    span_id = hashlib.sha256(str(event.get("event_id") or now_ns).encode("utf-8")).hexdigest()[:16]
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attr("service.name", "hermes-agent")]},
            "scopeSpans": [{
                "scope": {"name": "hermes.dev.production_signals"},
                "spans": [{
                    "traceId": trace_id,
                    "spanId": span_id,
                    "name": str(event.get("event") or "subagent.event"),
                    "kind": 1,
                    "startTimeUnixNano": now_ns,
                    "endTimeUnixNano": now_ns,
                    "attributes": [_attr(key, value) for key, value in attrs.items()],
                }],
            }],
        }],
    }


def _attr(key: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _cluster_key_hint(event: Dict[str, Any]) -> str:
    status = str(event.get("status") or "").lower()
    if status in {"failed", "needs_review", "error", "timeout", "timed_out"}:
        return f"terminal_status:{status}"
    verdict = str(event.get("verification_verdict") or event.get("verification_status") or "").lower()
    if verdict in {"failed", "partial", "unverifiable", "needs_review"}:
        return f"verification:{verdict}"
    return "agent-system:event"


def _cluster_title_hint(event: Dict[str, Any]) -> str:
    return str(event.get("goal") or event.get("event") or "Hermes agent-system event")[:MAX_ATTR_CHARS]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default
