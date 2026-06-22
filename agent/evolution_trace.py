"""Explicit run traces for the Hermes evolution loop.

This module is intentionally standalone and non-invasive: it writes profile-scoped
JSON trace files only when called by a CLI/tool/test path. It does not hook the
main agent loop, mutate prompts, change toolsets, or infer success from LLM text.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

_TRACE_ROOT_PARTS = ("logs", "evolution", "runs")
_VALID_FINAL_STATUSES = {"running", "success", "failed", "blocked", "unknown"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    result: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _as_question_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, dict)):
        values = [values]
    result: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts = [f"{str(k).strip()}: {str(v).strip()}" for k, v in value.items()]
            text = "; ".join(part for part in parts if part.strip())
        else:
            text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_grill_gate(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "purpose": str(value.get("purpose") or ""),
        "required_questions": _as_question_list(value.get("required_questions")),
    }


def _trace_root() -> Path:
    root = get_hermes_home()
    for part in _TRACE_ROOT_PARTS:
        root = root / part
    return root


def _trace_dir(run_id: str) -> Path:
    clean = str(run_id or "").strip()
    if not clean or any(sep in clean for sep in ("/", "\\")) or clean in {".", ".."}:
        raise ValueError("run_id must be a single path segment")
    return _trace_root() / clean


def _trace_path(run_id: str) -> Path:
    return _trace_dir(run_id) / "trace.json"


def _new_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _write_trace(payload: Dict[str, Any]) -> Path:
    path = _trace_path(payload["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_trace(run_id: str) -> Dict[str, Any]:
    """Read a trace payload by run id."""
    path = _trace_path(run_id)
    return json.loads(path.read_text(encoding="utf-8"))


def start_trace(
    *,
    mission: Optional[str] = None,
    session_id: Optional[str] = None,
    design_statement: Optional[str] = None,
    skills_loaded: Optional[Iterable[Any]] = None,
    allowed_side_effects: Optional[Iterable[Any]] = None,
    blocked_side_effects: Optional[Iterable[Any]] = None,
    evaluation_gates: Optional[Iterable[Any]] = None,
    evidence_roots: Optional[Iterable[Any]] = None,
    must_read_notes: Optional[Iterable[Any]] = None,
    grill_gate: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a profile-scoped evolution trace and return its id/path.

    The trace is a measurement artifact only. Callers decide what to record; this
    helper only persists structured evidence for later reports/evals.
    """
    trace_id = str(run_id or _new_run_id())
    started_at = _now_iso()
    payload: Dict[str, Any] = {
        "run_id": trace_id,
        "mission": mission or "default",
        "session_id": session_id,
        "design_statement": design_statement,
        "skills_loaded": _as_list(skills_loaded),
        "tools_called": [],
        "files_changed": [],
        "tests_run": [],
        "success_signals": [],
        "failure_signals": [],
        "evidence_sources": [],
        "gate_results": [],
        "decisions": [],
        "context_packs": [],
        "change_packets": [],
        "allowed_side_effects": _as_list(allowed_side_effects),
        "blocked_side_effects": _as_list(blocked_side_effects),
        "evaluation_gates": _as_list(evaluation_gates),
        "evidence_roots": _as_list(evidence_roots),
        "must_read_notes": _as_list(must_read_notes),
        "grill_gate": _normalize_grill_gate(grill_gate),
        "events": [],
        "started_at": started_at,
        "completed_at": None,
        "final_status": "running",
        "summary": "",
    }
    path = _write_trace(payload)
    return {"run_id": trace_id, "trace_path": str(path), "trace_dir": str(path.parent)}


def _append_unique(items: List[Any], item: Any) -> None:
    if item is None:
        return
    if item not in items:
        items.append(item)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _apply_event_aggregates(payload: Dict[str, Any], event: Dict[str, Any]) -> None:
    event_type = event.get("type")
    status = str(event.get("status") or "").lower()
    if event_type == "tool_call":
        _append_unique(payload.setdefault("tools_called", []), event.get("tool"))
    elif event_type == "file_change":
        _append_unique(payload.setdefault("files_changed", []), event.get("path"))
    elif event_type == "test_run":
        command = event.get("command")
        if command:
            _append_unique(
                payload.setdefault("tests_run", []),
                {
                    "command": command,
                    "status": event.get("status") or "unknown",
                    "summary": event.get("summary") or "",
                },
            )
    elif event_type == "signal":
        signal = event.get("signal") or event.get("summary")
        if status in {"success", "passed", "pass", "ok"}:
            _append_unique(payload.setdefault("success_signals", []), signal)
        elif status in {"failure", "failed", "fail", "blocked", "error"}:
            _append_unique(payload.setdefault("failure_signals", []), signal)
    elif event_type == "evidence_source":
        source = {
            "source_type": event.get("source_type") or "unknown",
            "path": event.get("path"),
            "title": event.get("title"),
            "summary": event.get("summary") or "",
        }
        _append_unique(payload.setdefault("evidence_sources", []), source)
    elif event_type == "gate":
        gate = event.get("gate") or event.get("signal") or event.get("summary")
        row = {"gate": gate, "status": event.get("status") or "unknown", "summary": event.get("summary") or ""}
        _append_unique(payload.setdefault("gate_results", []), row)
        if status in {"success", "passed", "pass", "ok"}:
            _append_unique(payload.setdefault("success_signals", []), gate)
        elif status in {"failure", "failed", "fail", "blocked", "error"}:
            _append_unique(payload.setdefault("failure_signals", []), gate)
    elif event_type == "decision":
        _append_unique(
            payload.setdefault("decisions", []),
            {
                "decision": event.get("decision") or "unknown",
                "status": event.get("status") or "unknown",
                "summary": event.get("summary") or "",
            },
        )
    elif event_type == "context_pack":
        pack = {
            "path": event.get("path"),
            "topic": event.get("topic") or "",
            "status": event.get("status") or "unknown",
            "source_count": _optional_int(event.get("source_count")),
            "missing_evidence": event.get("missing_evidence") or "",
            "summary": event.get("summary") or "",
        }
        _append_unique(payload.setdefault("context_packs", []), pack)
    elif event_type == "change_packet":
        packet = {
            "path": event.get("path"),
            "topic": event.get("topic") or "",
            "objective": event.get("objective") or "",
            "status": event.get("status") or "unknown",
            "summary": event.get("summary") or "",
        }
        _append_unique(payload.setdefault("change_packets", []), packet)


def record_event(run_id: str, event_type: str, **fields: Any) -> Dict[str, Any]:
    """Append one event to a trace and update simple aggregate fields."""
    payload = read_trace(run_id)
    event = {
        "type": str(event_type),
        "at": _now_iso(),
        **{k: v for k, v in fields.items() if v is not None},
    }
    payload.setdefault("events", []).append(event)
    _apply_event_aggregates(payload, event)
    path = _write_trace(payload)
    return {"run_id": run_id, "trace_path": str(path), "event_count": len(payload["events"])}


def finish_trace(run_id: str, *, final_status: str, summary: str = "") -> Dict[str, Any]:
    """Mark a trace terminal without inferring success automatically."""
    status = str(final_status or "").strip().lower()
    if status not in _VALID_FINAL_STATUSES - {"running"}:
        raise ValueError("final_status must be one of success, failed, blocked, unknown")
    payload = read_trace(run_id)
    payload["final_status"] = status
    payload["summary"] = str(summary or "")
    payload["completed_at"] = _now_iso()
    path = _write_trace(payload)
    return {"run_id": run_id, "trace_path": str(path), "final_status": status}


def load_mission_manifest(name: str) -> Dict[str, Any]:
    """Load a mission manifest from HERMES_HOME/mission-manifests/<name>.yaml."""
    clean = str(name or "").strip()
    if not clean or any(sep in clean for sep in ("/", "\\")) or clean in {".", ".."}:
        raise ValueError("mission name must be a single path segment")
    path = get_hermes_home() / "mission-manifests" / f"{clean}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Mission manifest not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - dependency is present in normal Hermes installs
        raise RuntimeError("PyYAML is required to read mission manifests") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mission manifest must be a mapping: {path}")
    manifest = dict(data)
    manifest.setdefault("name", clean)
    for key in ("default_skills", "allowed_side_effects", "blocked_side_effects", "evaluation_gates", "evidence_roots", "must_read_notes"):
        manifest[key] = _as_list(manifest.get(key))
    manifest["grill_gate"] = _normalize_grill_gate(manifest.get("grill_gate"))
    manifest["design_statement"] = str(manifest.get("design_statement") or "")
    manifest["path"] = str(path)
    return manifest


def start_trace_from_manifest(name: str, *, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Start a trace using policy fields from a mission manifest."""
    manifest = load_mission_manifest(name)
    result = start_trace(
        mission=manifest["name"],
        session_id=session_id,
        design_statement=manifest.get("design_statement") or None,
        skills_loaded=manifest.get("default_skills"),
        allowed_side_effects=manifest.get("allowed_side_effects"),
        blocked_side_effects=manifest.get("blocked_side_effects"),
        evaluation_gates=manifest.get("evaluation_gates"),
        evidence_roots=manifest.get("evidence_roots"),
        must_read_notes=manifest.get("must_read_notes"),
        grill_gate=manifest.get("grill_gate"),
    )
    result["manifest_path"] = manifest["path"]
    return result


def list_recent_traces(*, limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent trace payloads, newest first, skipping malformed traces."""
    root = _trace_root()
    if not root.exists():
        return []
    traces: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/trace.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("run_id"):
                traces.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
        if len(traces) >= limit:
            break
    return traces
