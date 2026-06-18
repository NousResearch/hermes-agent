"""Execution receipt construction and hook emission.

This module is intentionally small and passive. It builds metadata-only
execution receipts and emits them to the plugin system when a plugin has opted
in to the ``execution_receipt`` observer hook. It must never affect tool
execution: all hook/serialization failures are fail-open.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "hermes.execution_receipt.v0"
REDACTION_POLICY_VERSION = "execution_receipts.v0"
HOOK_NAME = "execution_receipt"


def emit_tool_execution_receipt(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    status: str = "ok",
    duration_ms: int | None = None,
    args: Any = None,
    result: Any = None,
    sequence_number: int | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    links: Sequence[Mapping[str, Any]] | None = None,
    evidence_gaps: Sequence[str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any] | None:
    """Emit a metadata-only tool-complete execution receipt.

    Returns the receipt when it was built, or ``None`` when no plugin listens
    for receipt events. The caller must not depend on delivery: plugin hook
    failures are logged at debug level and swallowed.
    """
    try:
        from hermes_cli import plugins

        if not plugins.has_hook(HOOK_NAME):
            return None
    except Exception as exc:
        logger.debug("execution receipt hook check failed: %s", exc)
        return None

    receipt = _build_tool_execution_receipt(
        session_id=session_id,
        task_id=task_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=status,
        duration_ms=duration_ms,
        args=args,
        result=result,
        sequence_number=sequence_number,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        links=links,
        evidence_gaps=evidence_gaps,
        error_type=error_type,
        error_message=error_message,
    )

    try:
        plugins.invoke_hook(HOOK_NAME, receipt=receipt)
    except Exception as exc:
        logger.debug("execution receipt hook failed: %s", exc, exc_info=True)
    return receipt


def _build_tool_execution_receipt(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    status: str = "ok",
    duration_ms: int | None = None,
    args: Any = None,
    result: Any = None,
    sequence_number: int | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    links: Sequence[Mapping[str, Any]] | None = None,
    evidence_gaps: Sequence[str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": f"receipt-{uuid.uuid4().hex}",
        "receipt_type": "tool_complete",
        "trace_id": trace_id or f"trace-{uuid.uuid4().hex}",
        "span_id": f"span-{uuid.uuid4().hex}",
        "parent_span_id": parent_span_id,
        "sequence_number": int(sequence_number or 0),
        "timestamp": _now_iso(),
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "turn_id": str(turn_id or ""),
        "api_request_id": str(api_request_id or ""),
        "tool_call_id": str(tool_call_id or ""),
        "tool_name": str(tool_name or ""),
        "status": str(status or "ok"),
        "duration_ms": _coerce_duration_ms(duration_ms),
        "args": _redacted_payload_metadata(args),
        "result": _redacted_payload_metadata(result),
        "links": _normalize_links(links),
        "evidence_gaps": _normalize_evidence_gaps(evidence_gaps),
        "redaction_policy_version": REDACTION_POLICY_VERSION,
        "redaction_status": "ok",
        "error_type": str(error_type) if error_type else None,
        "error_message": None,
        "error_message_metadata": _redacted_error_message_metadata(error_message),
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce_duration_ms(duration_ms: int | None) -> int | None:
    if duration_ms is None:
        return None
    try:
        coerced = int(duration_ms)
    except Exception:
        return None
    return max(coerced, 0)


def _redacted_payload_metadata(payload: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "redacted": True,
        "kind": type(payload).__name__ if payload is not None else "none",
        "size_bytes": _payload_size_bytes(payload),
    }
    if isinstance(payload, Mapping):
        metadata["field_names"] = sorted(str(key) for key in payload.keys())[:50]
        metadata["field_count"] = len(payload)
    elif isinstance(payload, (list, tuple, set, frozenset)):
        metadata["item_count"] = len(payload)
    elif isinstance(payload, str):
        metadata["char_count"] = len(payload)
    return metadata


def _payload_size_bytes(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8", errors="replace"))
    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(type(payload).__name__)
    return len(serialized.encode("utf-8", errors="replace"))


def _normalize_links(links: Sequence[Mapping[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not links:
        return normalized
    for link in links[:50]:
        if not isinstance(link, Mapping):
            continue
        item: dict[str, str] = {}
        for key in ("type", "id", "trace_id", "session_id", "task_id", "subagent_id"):
            value = link.get(key)
            if value is not None:
                item[key] = str(value)
        if item:
            normalized.append(item)
    return normalized


def _normalize_evidence_gaps(evidence_gaps: Sequence[str] | None) -> list[str]:
    if not evidence_gaps:
        return []
    return [str(gap) for gap in evidence_gaps if gap][:50]


def _redacted_error_message_metadata(error_message: str | None) -> dict[str, Any] | None:
    if not error_message:
        return None
    text = str(error_message)
    return {
        "redacted": True,
        "kind": type(error_message).__name__,
        "char_count": len(text),
        "size_bytes": len(text.encode("utf-8", errors="replace")),
    }


__all__ = [
    "HOOK_NAME",
    "REDACTION_POLICY_VERSION",
    "SCHEMA_VERSION",
    "emit_tool_execution_receipt",
]
