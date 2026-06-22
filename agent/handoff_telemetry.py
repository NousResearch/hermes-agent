"""Durable telemetry helpers for agent handoffs/delegation.

This module intentionally stays small and dependency-light: callers hand us
already-normalized counters from the running agent, and we append one JSONL
record per handoff execution.  The JSONL file is a practical baseline that can
later feed richer observability plugins or dashboards without changing the
runtime call sites again.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home

try:
    from agent.redact import redact_sensitive_text
except Exception:  # pragma: no cover - import fallback for early bootstrap/tests
    def redact_sensitive_text(text: str, *, force: bool = False, code_file: bool = False) -> str:
        return text

logger = logging.getLogger(__name__)

TELEMETRY_FILENAME = "handoff_telemetry.jsonl"


def new_trace_id(prefix: str = "handoff") -> str:
    """Return a compact correlation id for a handoff execution."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    return int(_number(value, 0.0))


def _redact_jsonish(value: Any) -> Any:
    """Return a JSON-safe copy with string leaves force-redacted."""
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            redact_sensitive_text(str(key), force=True): _redact_jsonish(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_jsonish(item) for item in value]
    return redact_sensitive_text(str(value), force=True)


def build_handoff_telemetry_event(
    *,
    trace_id: str,
    subagent_id: Optional[str],
    parent_session_id: Optional[str],
    parent_task_id: Optional[str],
    parent_subagent_id: Optional[str],
    task_index: int,
    status: str,
    exit_reason: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    agent_id: Optional[str] = None,
    assigned_model: Optional[str] = None,
    assigned_provider: Optional[str] = None,
    effective_model: Optional[str] = None,
    effective_provider: Optional[str] = None,
    model_source: Optional[str] = None,
    model_resolution_warnings: Optional[Any] = None,
    api_mode: Optional[str] = None,
    api_calls: Any = 0,
    duration_seconds: Any = 0,
    input_tokens: Any = 0,
    output_tokens: Any = 0,
    cache_read_tokens: Any = 0,
    cache_write_tokens: Any = 0,
    reasoning_tokens: Any = 0,
    prompt_tokens: Any = 0,
    total_tokens: Any = 0,
    estimated_cost_usd: Any = None,
    cost_status: Optional[str] = None,
    cost_source: Optional[str] = None,
    result: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a redaction-safe handoff telemetry event."""
    in_tokens = _int(input_tokens)
    out_tokens = _int(output_tokens)
    cache_read = _int(cache_read_tokens)
    cache_write = _int(cache_write_tokens)
    reasoning = _int(reasoning_tokens)
    prompt = _int(prompt_tokens) or (in_tokens + cache_read + cache_write)
    total = _int(total_tokens) or (prompt + out_tokens)

    warnings_list = (
        list(model_resolution_warnings)
        if isinstance(model_resolution_warnings, (list, tuple, set))
        else ([str(model_resolution_warnings)] if model_resolution_warnings else [])
    )

    event: dict[str, Any] = {
        "schema_version": 1,
        "event_type": "agent_handoff",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "subagent_id": subagent_id,
        "parent_session_id": parent_session_id,
        "parent_task_id": parent_task_id,
        "parent_subagent_id": parent_subagent_id,
        "task_index": int(task_index),
        "status": status,
        "exit_reason": exit_reason,
        "model": model,
        "provider": provider,
        "agent_id": agent_id,
        "assigned_model": assigned_model,
        "assigned_provider": assigned_provider,
        "effective_model": effective_model or model,
        "effective_provider": effective_provider or provider,
        "model_source": model_source,
        "model_resolution_warnings": warnings_list,
        "api_mode": api_mode,
        "api_calls": _int(api_calls),
        "duration_seconds": round(_number(duration_seconds, 0.0), 3),
        "tokens": {
            "input": in_tokens,
            "output": out_tokens,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "reasoning": reasoning,
            "prompt": prompt,
            "total": total,
        },
        "cost": {
            "estimated_usd": None
            if estimated_cost_usd is None
            else round(_number(estimated_cost_usd, 0.0), 8),
            "status": cost_status,
            "source": cost_source,
        },
        "result": _redact_jsonish(dict(result or {})),
    }
    return event


def telemetry_log_path() -> Path:
    return get_hermes_home() / "logs" / TELEMETRY_FILENAME


def record_handoff_telemetry(event: Mapping[str, Any]) -> Optional[Path]:
    """Append one handoff telemetry event to ~/.hermes/logs/handoff_telemetry.jsonl.

    Telemetry must never break agent execution; failures are logged at debug and
    otherwise ignored.
    """
    try:
        path = telemetry_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _redact_jsonish(dict(event))
        # If a caller supplied a non-JSON value, preserve useful text rather than
        # failing the entire telemetry write.
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return path
    except Exception as exc:
        logger.debug("handoff telemetry write failed: %s", exc, exc_info=True)
        return None


def file_mtime_ns(path: Path) -> int:
    """Tiny test/helper utility for callers that want to verify a write moved."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0
