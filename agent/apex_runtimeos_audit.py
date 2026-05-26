"""APEX RuntimeOS audit trail writer.

Writes compact JSONL checkpoint records without prompts, messages, local report
paths, raw errors, or credentials. This is a best-effort audit channel: failures
must not break user-facing turns unless future enforce logic explicitly requires
it elsewhere.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_ALLOWED_TOP_LEVEL = {
    "stage",
    "enabled",
    "mode",
    "blocking",
    "context_unchanged",
    "results",
}
_ALLOWED_RESULT_FIELDS = {"enabled", "mode", "point", "status", "blocking", "context_unchanged", "output", "control", "elapsed_ms"}
_ALLOWED_CONTROL_FIELDS = {
    "auto_control_enabled",
    "mode",
    "min_severity",
    "severity",
    "action",
    "blocking",
    "applied",
    "mutates_runtime",
    "reason",
}
_ALLOWED_OUTPUT_FIELDS = {
    "model",
    "provider",
    "api_mode",
    "tool_count",
    "message_count",
    "api_call_count",
    "approx_input_tokens",
    "request_char_count",
    "max_tokens",
    "completed",
    "interrupted",
    "turn_exit_reason",
    "final_response_present",
    "decision",
    "mutates_route",
    "mutates_plan",
    "mutates_memory_or_skills",
}
_SECRET_HINTS = ("key", "token", "secret", "password", "authorization", "credential")
_PATH_HINTS = ("path", "dir", "file")


def _audit_enabled() -> bool:
    return os.environ.get("APEX_RUNTIMEOS_AUDIT_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off", ""}


def _audit_path() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_AUDIT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("HERMES_APEX_RUNTIMEOS_AUDIT_DIR", str(Path.home() / ".hermes" / "apex_runtimeos_audit")))
    return base / "checkpoints.jsonl"


def _safe_scalar(key: str, value: Any) -> Any:
    lower = key.lower()
    if any(hint in lower for hint in _SECRET_HINTS):
        return "[REDACTED]"
    if any(hint in lower for hint in _PATH_HINTS):
        return "[REDACTED_PATH]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 200:
            return value[:197] + "..."
        return value
    return str(value)[:200]


def _sanitize_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key in _ALLOWED_TOP_LEVEL:
        if key not in checkpoint:
            continue
        if key != "results":
            safe[key] = _safe_scalar(key, checkpoint.get(key))
            continue
        results = checkpoint.get("results")
        if not isinstance(results, Mapping):
            continue
        safe_results: Dict[str, Any] = {}
        for organ, result in results.items():
            if not isinstance(result, Mapping):
                continue
            item: Dict[str, Any] = {}
            for r_key in _ALLOWED_RESULT_FIELDS:
                if r_key not in result:
                    continue
                if r_key == "output":
                    output = result.get("output")
                    if isinstance(output, Mapping):
                        item["output"] = {
                            o_key: _safe_scalar(o_key, output.get(o_key))
                            for o_key in _ALLOWED_OUTPUT_FIELDS
                            if o_key in output
                        }
                    continue
                if r_key == "control":
                    control = result.get("control")
                    if isinstance(control, Mapping):
                        item["control"] = {
                            c_key: _safe_scalar(c_key, control.get(c_key))
                            for c_key in _ALLOWED_CONTROL_FIELDS
                            if c_key in control
                        }
                    continue
                item[r_key] = _safe_scalar(r_key, result.get(r_key))
            safe_results[str(organ)] = item
        safe["results"] = safe_results
    return safe


def persist_checkpoint(stage: str, checkpoint: Mapping[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Append a sanitized checkpoint to JSONL and return write metadata."""
    if not _audit_enabled():
        return {"audit_enabled": False, "written": False}
    safe_checkpoint = _sanitize_checkpoint(checkpoint)
    record = {
        "schema": "ApexRuntimeOSCheckpointAudit/v1",
        "ts": time.time(),
        "stage": stage,
        "session_id": session_id or "",
        "checkpoint": safe_checkpoint,
    }
    out = _audit_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"audit_enabled": True, "written": True, "audit_path": str(out)}


__all__ = ["persist_checkpoint"]
