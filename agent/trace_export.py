"""Render a :class:`~agent.trace_builder.Trace` into portable file formats.

Two formats, both hand-built (no OTel SDK dependency):

* **OTLP/JSON** with OpenInference semantic conventions — the industry standard
  for LLM/agent traces. Ingestible by Arize Phoenix and any OpenTelemetry
  backend, so we can confirm our spans are correct in a real viewer.
* **Chrome Trace Event format** — each session becomes its own track, viewable
  by dropping the file into https://ui.perfetto.dev or ``chrome://tracing``.

Keeping these as plain dict/JSON builders means the trace layer has zero new
third-party dependencies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from agent.trace_builder import STATUS_ERROR, STATUS_OK, Trace

_SERVICE_NAME = "hermes-agent"
_SCOPE_NAME = "hermes.tracing"


def _hex_id(value: str, nbytes: int) -> str:
    """Deterministic hex id of ``nbytes`` bytes from an arbitrary string."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[: nbytes * 2]


def _otlp_any_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _otlp_attributes(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, value in attrs.items():
        if value is None:
            continue
        out.append({"key": key, "value": _otlp_any_value(value)})
    return out


def _otlp_status(status: str) -> Dict[str, Any]:
    if status == STATUS_OK:
        return {"code": 1}
    if status == STATUS_ERROR:
        return {"code": 2}
    return {"code": 0}


def _to_nanos(seconds: float) -> str:
    return str(int(seconds * 1_000_000_000))


def to_otlp_json(trace: Trace) -> Dict[str, Any]:
    """Build an OTLP/JSON ``TracesData`` document with OpenInference attributes."""
    trace_hex = _hex_id(trace.trace_id, 16)
    otlp_spans: List[Dict[str, Any]] = []

    for span in trace.spans:
        attributes = dict(span.attributes)
        # OpenInference: the GenAI span kind travels as an attribute; the OTLP
        # SpanKind stays INTERNAL (1).
        attributes["openinference.span.kind"] = span.kind
        if span.session_id:
            attributes.setdefault("session.id", span.session_id)

        otlp_span: Dict[str, Any] = {
            "traceId": trace_hex,
            "spanId": _hex_id(span.span_id, 8),
            "name": span.name,
            "kind": 1,
            "startTimeUnixNano": _to_nanos(span.start),
            "endTimeUnixNano": _to_nanos(span.end),
            "attributes": _otlp_attributes(attributes),
            "status": _otlp_status(span.status),
        }
        if span.parent_id:
            otlp_span["parentSpanId"] = _hex_id(span.parent_id, 8)
        otlp_spans.append(otlp_span)

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _otlp_attributes(
                        {
                            "service.name": _SERVICE_NAME,
                            "session.id": trace.root_session_id,
                            "hermes.source": trace.metadata.get("source"),
                        }
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": _SCOPE_NAME},
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }


def to_chrome_trace(trace: Trace) -> Dict[str, Any]:
    """Build a Chrome Trace Event document (one track per session)."""
    base = trace.start
    events: List[Dict[str, Any]] = []

    # Stable, compact track ids per session — each session is its own lane.
    tids: Dict[str, int] = {}

    def tid_for(session_id: str) -> int:
        if session_id not in tids:
            tids[session_id] = len(tids) + 1
        return tids[session_id]

    for span in trace.spans:
        sid = span.session_id or trace.root_session_id
        events.append(
            {
                "name": span.name,
                "cat": span.kind,
                "ph": "X",
                "ts": (span.start - base) * 1_000_000,
                "dur": max(0.0, span.duration) * 1_000_000,
                "pid": 1,
                "tid": tid_for(sid),
                "args": {k: v for k, v in span.attributes.items() if v is not None},
            }
        )

    # Name each track after its session (root first) for legible lanes.
    for session_id, tid in tids.items():
        label = "root" if session_id == trace.root_session_id else f"subagent {session_id[:8]}"
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": tid,
                "args": {"name": label},
            }
        )

    return {"traceEvents": events, "displayTimeUnit": "ms"}


def dumps(trace: Trace, fmt: str = "otlp", *, indent: int = 2) -> str:
    """Serialize ``trace`` to a JSON string in the requested format."""
    fmt = (fmt or "otlp").lower()
    if fmt in {"otlp", "otlp-json", "openinference"}:
        doc = to_otlp_json(trace)
    elif fmt in {"chrome", "perfetto", "trace-event"}:
        doc = to_chrome_trace(trace)
    else:
        raise ValueError(f"unknown trace format: {fmt!r} (use 'otlp' or 'chrome')")
    return json.dumps(doc, ensure_ascii=False, indent=indent, default=str)


__all__ = ["dumps", "to_chrome_trace", "to_otlp_json"]
