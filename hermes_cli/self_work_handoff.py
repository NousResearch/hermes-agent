"""Durable restart-safe self-work handoff records for Cockpit/Biff.

This module promotes the existing Discord operator-checklist shape from a
rendering primitive into a small local persistence contract.  It is intentionally
local, display-safe, and read-mostly: Cockpit can read the latest record after a
dashboard/gateway restart without needing the model's pre-restart context.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from gateway.operator_checklist import format_operator_checklist
from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
_MAX_FIELD_CHARS = 320
_MAX_LONG_FIELD_CHARS = 1200
_MAX_LIST_ITEMS = 20
_HANDOFF_DIR = "cockpit"
_LATEST_FILENAME = "self_work_handoff_latest.json"
_HISTORY_FILENAME = "self_work_handoff_history.jsonl"

_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|token|secret|password|passwd|authorization|credential|session[_-]?key)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+)\b"
)
_LOCAL_CREDENTIAL_PATH_RE = re.compile(r"(?:^|\s)(?:~|/Users/[^\s]+)/(?:\.local/bin/get_credential\.sh|[^\s]*credentials/[^\s]*)")


def _handoff_dir() -> Path:
    return get_hermes_home() / _HANDOFF_DIR


def latest_handoff_path() -> Path:
    return _handoff_dir() / _LATEST_FILENAME


def history_handoff_path() -> Path:
    return _handoff_dir() / _HISTORY_FILENAME


def _clean_text(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str:
    text = " ".join(str(value or "").replace("`", "'").split())
    text = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[redacted]", text)
    text = _SECRET_VALUE_RE.sub("[redacted key]", text)
    text = _LOCAL_CREDENTIAL_PATH_RE.sub(" [redacted credential path]", text)
    if len(text) > limit:
        return text[: max(1, limit - 1)].rstrip() + "…"
    return text


def _clean_optional(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str | None:
    text = _clean_text(value, limit=limit)
    return text or None


def _clean_list(values: Any, *, limit: int = _MAX_FIELD_CHARS) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raw_items = [values]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            raw_items = [values]
    cleaned = []
    for item in raw_items[:_MAX_LIST_ITEMS]:
        text = _clean_text(item, limit=limit)
        if text:
            cleaned.append(text)
    return cleaned


def _normalise_status(value: Any) -> str:
    raw = str(value or "pending").strip().lower()
    aliases = {
        "complete": "done",
        "completed": "done",
        "success": "done",
        "active": "current",
        "in_progress": "current",
        "running": "current",
        "now": "current",
        "todo": "pending",
        "queued": "pending",
        "waiting": "pending",
        "hold": "blocked",
        "error": "blocked",
        "failed": "blocked",
        "skip": "skipped",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"done", "current", "pending", "blocked", "skipped"} else "pending"


def _normalise_checklist(payload: Mapping[str, Any]) -> dict[str, Any]:
    checklist = payload.get("operator_checklist") or payload.get("checklist") or {}
    if not isinstance(checklist, Mapping):
        checklist = {}
    steps_value = checklist.get("steps") or payload.get("steps") or []
    steps = []
    if not isinstance(steps_value, (str, bytes)):
        try:
            iterable = list(steps_value)
        except TypeError:
            iterable = []
        for item in iterable[:_MAX_LIST_ITEMS]:
            if isinstance(item, Mapping):
                label = item.get("label") or item.get("name") or item.get("step")
                status = item.get("status")
            elif isinstance(item, (list, tuple)) and item:
                label = item[0]
                status = item[1] if len(item) > 1 else "pending"
            else:
                label = item
                status = "pending"
            label_text = _clean_text(label, limit=80)
            if label_text:
                steps.append({"label": label_text, "status": _normalise_status(status)})
    current_index = checklist.get("current_index", payload.get("current_index"))
    try:
        current_index = int(current_index) if current_index is not None else None
    except (TypeError, ValueError):
        current_index = None
    return {
        "title": _clean_text(checklist.get("title") or payload.get("title") or payload.get("issue_identifier") or "Self-work handoff", limit=80),
        "current_index": current_index,
        "steps": steps,
        "max_steps": min(max(int(checklist.get("max_steps", 8) or 8), 1), 12),
    }


def normalise_self_work_handoff(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a display-safe durable handoff record.

    The returned shape is suitable for local JSON persistence and for Cockpit's
    read-only resume brief.  It deliberately stores concise work-position fields,
    not raw transcripts, credentials, or platform/channel identifiers.
    """

    if not isinstance(payload, Mapping):
        payload = {}
    checklist = _normalise_checklist(payload)
    record = {
        "schema_version": SCHEMA_VERSION,
        "handoff_id": _clean_text(payload.get("handoff_id") or f"handoff-{int(time.time())}", limit=80),
        "created_at": _clean_text(payload.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), limit=40),
        "updated_at": _clean_text(payload.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), limit=40),
        "issue_identifier": _clean_optional(payload.get("issue_identifier"), limit=32),
        "linear_url": _clean_optional(payload.get("linear_url"), limit=160),
        "session_id": _clean_optional(payload.get("session_id"), limit=120),
        "parent_session_id": _clean_optional(payload.get("parent_session_id"), limit=120),
        "goal": _clean_optional(payload.get("goal") or payload.get("title"), limit=160),
        "current_phase": _clean_optional(payload.get("current_phase") or payload.get("phase"), limit=120),
        "last_action": _clean_optional(payload.get("last_action"), limit=_MAX_LONG_FIELD_CHARS),
        "next_safe_step": _clean_optional(payload.get("next_safe_step"), limit=_MAX_LONG_FIELD_CHARS),
        "touched_files": _clean_list(payload.get("touched_files") or payload.get("files"), limit=160),
        "touched_routes": _clean_list(payload.get("touched_routes") or payload.get("routes"), limit=160),
        "completed_verification": _clean_list(payload.get("completed_verification"), limit=220),
        "pending_verification": _clean_list(payload.get("pending_verification"), limit=220),
        "known_failures": _clean_list(payload.get("known_failures") or payload.get("unresolved_failures"), limit=260),
        "running_processes": _clean_list(payload.get("running_processes"), limit=160),
        "operator_checklist": checklist,
    }
    record["rendered_checklist"] = format_operator_checklist(
        checklist["steps"],
        title=checklist["title"],
        current_index=checklist.get("current_index"),
        max_steps=checklist.get("max_steps", 8),
        fenced=False,
    )
    return {k: v for k, v in record.items() if v not in (None, [], {})}


def write_self_work_handoff(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the latest handoff atomically and append a JSONL history row."""

    record = normalise_self_work_handoff(payload)
    directory = _handoff_dir()
    directory.mkdir(parents=True, exist_ok=True)
    latest = latest_handoff_path()
    fd, tmp_name = tempfile.mkstemp(prefix="self_work_handoff_", suffix=".json", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(record, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, latest)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    history = history_handoff_path()
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        os.chmod(history, 0o600)
    except OSError:
        pass
    return record


def read_latest_self_work_handoff() -> dict[str, Any] | None:
    path = latest_handoff_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, Mapping):
        return None
    return normalise_self_work_handoff(data)


def cockpit_self_work_handoff_payload() -> dict[str, Any]:
    record = read_latest_self_work_handoff()
    return {
        "schema_version": SCHEMA_VERSION,
        "read_only": True,
        "actions_enabled": False,
        "mutation_enabled": False,
        "source": "local_self_work_handoff",
        "has_handoff": record is not None,
        "handoff": record,
    }
