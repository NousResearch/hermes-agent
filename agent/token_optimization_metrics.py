"""Measurement-only token optimization metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_TRUE_VALUES = {"1", "true", "yes", "on"}


def metrics_enabled() -> bool:
    """Return whether opt-in Phase 1 token metrics are enabled."""
    return os.getenv("HERMES_TOKEN_OPTIMIZATION_METRICS", "").strip().lower() in _TRUE_VALUES


def metrics_path() -> Path:
    """Return the profile-aware JSONL metrics path."""
    return get_hermes_home() / "token-optimization" / "metrics.jsonl"


def estimate_tokens(char_count: int) -> int:
    """Estimate tokens from result size without inspecting prompt text."""
    return int(math.ceil(char_count / 4)) if char_count > 0 else 0


def _arg_keys(function_args: Any) -> list[str]:
    if not isinstance(function_args, dict):
        return []
    return sorted(str(key) for key in function_args)


def _json_shape(result: str) -> dict[str, Any]:
    try:
        parsed = json.loads(result)
    except Exception:
        return {
            "is_json": False,
            "top_level_type": "invalid_json",
            "has_error_key": False,
            "has_non_empty_error": False,
            "truncated_by_tool": False,
            "result_class": "invalid_json",
        }

    if isinstance(parsed, dict):
        top_level_type = "object"
        has_error_key = "error" in parsed
        error_value = parsed.get("error")
        has_non_empty_error = error_value not in (None, "", False)
        truncated_by_tool = any(
            parsed.get(key) is True
            for key in ("truncated_by_tool", "truncated", "is_truncated")
        )
        if has_non_empty_error:
            result_class = "runtime_error"
        elif truncated_by_tool:
            result_class = "truncated"
        elif not parsed:
            result_class = "empty"
        else:
            result_class = "success"
    elif isinstance(parsed, list):
        top_level_type = "array"
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "empty" if not parsed else "success"
    elif isinstance(parsed, str):
        top_level_type = "string"
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "empty" if not parsed else "success"
    elif isinstance(parsed, bool):
        top_level_type = "boolean"
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "success"
    elif isinstance(parsed, (int, float)):
        top_level_type = "number"
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "success"
    elif parsed is None:
        top_level_type = "null"
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "empty"
    else:
        top_level_type = type(parsed).__name__
        has_error_key = False
        has_non_empty_error = False
        truncated_by_tool = False
        result_class = "success"

    return {
        "is_json": True,
        "top_level_type": top_level_type,
        "has_error_key": has_error_key,
        "has_non_empty_error": has_non_empty_error,
        "truncated_by_tool": truncated_by_tool,
        "result_class": result_class,
    }


def build_tool_result_metric(
    *,
    tool_name: str,
    function_args: Any,
    result: Any,
    task_id: str | None,
    session_id: str | None,
    tool_call_id: str | None,
    duration_ms: int | None,
) -> dict[str, Any]:
    """Build metadata for a tool result without storing raw payloads.

    The sha256 value identifies identical payloads; it is not anonymization.
    """
    result_text = result if isinstance(result, str) else str(result)
    result_bytes = result_text.encode("utf-8", errors="replace")
    char_count = len(result_text)
    metric: dict[str, Any] = {
        "schema_version": 2,
        "event_type": "tool_result",
        "created_at_ms": int(time.time() * 1000),
        "session_id": session_id or "",
        "task_id": task_id or "",
        "tool_call_id": tool_call_id or "",
        "tool_name": tool_name,
        "duration_ms": duration_ms if isinstance(duration_ms, int) else None,
        "result_char_count": char_count,
        "result_byte_count": len(result_bytes),
        "estimated_tokens": estimate_tokens(char_count),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
        "arg_keys": _arg_keys(function_args),
    }
    metric.update(_json_shape(result_text))
    return metric


def record_tool_result_metric(
    *,
    tool_name: str,
    function_args: Any,
    result: Any,
    task_id: str | None,
    session_id: str | None,
    tool_call_id: str | None,
    duration_ms: int | None,
) -> None:
    """Append one opt-in metric row.

    This function is deliberately fail-open: metrics must never break tool
    dispatch, and Phase 1 records measurement metadata only.
    """
    try:
        if not metrics_enabled():
            return

        metric = build_tool_result_metric(
            tool_name=tool_name,
            function_args=function_args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
        )
        path = metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metric, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        return
