"""Discord approval-button helpers for human Kanban gates.

This module is intentionally free of discord.py imports so both the gateway
adapter and small watcher scripts can share the same routing/formatting logic.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb

CUSTOM_ID_PREFIX = "hermes:kanban-approval:"
APPROVAL_ACTIONS = ("approve", "deny", "needs_changes")
REVIEW_REQUIRED_RE = re.compile(r"\breview[-_ ]required\b", re.I)
HUMAN_GATE_RE = re.compile(
    r"\b(human[-_ ]gate|human[-_ ]approval|approval[-_ ]required|matthew[-_ ]approval|manual[-_ ]approval)\b",
    re.I,
)
SYNOPSIS_FIELD_LIMIT = 220
SECRET_FIELD_RE = re.compile(
    r"(?i)\b([\w.-]*(?:api[-_ ]?key|auth[-_ ]?token|access[-_ ]?token|refresh[-_ ]?token|secret|password|passwd|pwd|credential|private[-_ ]?key|token)[\w.-]*)\b\s*([:=])\s*([^\s,;\]\)\}]+|\"[^\"]*\"|'[^']*')"
)
SECRET_JSON_RE = re.compile(
    r"(?i)([\"']?[\w.-]*(?:api[-_ ]?key|auth[-_ ]?token|access[-_ ]?token|refresh[-_ ]?token|secret|password|passwd|pwd|credential|private[-_ ]?key)[\w.-]*[\"']?\s*:\s*)([\"'][^\"']*[\"']|[^\s,}\]]+)"
)
SECRET_TOKEN_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,}|gh[pousr]_[A-Za-z0-9_]{20,}|(?:bearer\s+)[A-Za-z0-9._~+/=-]{16,}|[A-Za-z0-9._~+/=-]{32,})\b"
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def redact_synopsis_field(value: Any, *, limit: int = SYNOPSIS_FIELD_LIMIT) -> str:
    """Return a compact, Discord-safe synopsis with obvious secrets redacted."""
    if value is None:
        text = ""
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    text = SECRET_FIELD_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    text = SECRET_JSON_RE.sub(lambda m: f"{m.group(1)}[redacted]", text)
    text = SECRET_TOKEN_RE.sub("[redacted]", text)
    text = EMAIL_RE.sub("[redacted]", text)
    if len(text) > limit:
        return text[: max(0, limit - 12)].rstrip() + " …[truncated]"
    return text


@dataclass(frozen=True)
class ApprovalRequest:
    task_id: str
    title: str
    reason: str
    project_context: str
    what_is_approved: str
    if_approved: str
    risk_rollback: str
    run_context: str = ""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def parse_payload(raw: Any) -> dict[str, Any]:
    return _as_dict(raw)


def parse_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(payload.get("metadata"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "required"}


def is_autonomous_review_gate(task: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    metadata = parse_metadata(payload)
    reason = str(payload.get("reason") or payload.get("summary") or payload.get("error") or "")
    haystack = "\n".join(
        str(x or "") for x in (
            reason,
            task.get("title"),
            task.get("body"),
            metadata.get("gate"),
            metadata.get("review_gate"),
        )
    )
    return bool(
        _truthy(payload.get("review_required"))
        or _truthy(metadata.get("review_required"))
        or REVIEW_REQUIRED_RE.search(haystack)
    )


def is_human_approval_gate(task: Mapping[str, Any], event_kind: str, payload: Mapping[str, Any]) -> bool:
    if event_kind != "blocked":
        return False
    if is_autonomous_review_gate(task, payload):
        return False
    metadata = parse_metadata(payload)
    reason = str(payload.get("reason") or payload.get("summary") or payload.get("error") or "")
    explicit = (
        payload.get("human_approval_required"),
        payload.get("human_gate"),
        metadata.get("human_approval_required"),
        metadata.get("human_gate"),
        metadata.get("discord_approval_request"),
    )
    if any(_truthy(v) for v in explicit):
        return True
    return bool(HUMAN_GATE_RE.search(reason))


def build_custom_id(task_id: str, action: str) -> str:
    if action not in APPROVAL_ACTIONS:
        raise ValueError(f"unsupported Kanban approval action: {action}")
    return f"{CUSTOM_ID_PREFIX}{task_id}:{action}"


def parse_custom_id(custom_id: str) -> Optional[tuple[str, str]]:
    if not custom_id.startswith(CUSTOM_ID_PREFIX):
        return None
    rest = custom_id[len(CUSTOM_ID_PREFIX):]
    try:
        task_id, action = rest.rsplit(":", 1)
    except ValueError:
        return None
    if not task_id or action not in APPROVAL_ACTIONS:
        return None
    return task_id, action


def build_approval_request(task: Mapping[str, Any], payload: Mapping[str, Any], project_context: Optional[Mapping[str, Any]] = None) -> ApprovalRequest:
    metadata = parse_metadata(payload)
    reason = str(payload.get("reason") or payload.get("summary") or payload.get("error") or "human approval required").strip()
    project_context = project_context or {}
    task_id = str(task.get("id") or payload.get("task_id") or "unknown")
    title = redact_synopsis_field(task.get("title") or task_id, limit=140)
    project = redact_synopsis_field(
        metadata.get("project_title")
        or project_context.get("project_title")
        or project_context.get("project_hub_slug")
        or metadata.get("project")
        or "not provided",
        limit=140,
    )
    run_bits = []
    if payload.get("run_id"):
        run_bits.append(f"run {payload['run_id']}")
    if task.get("assignee"):
        run_bits.append(f"assignee {task['assignee']}")
    return ApprovalRequest(
        task_id=task_id,
        title=title,
        reason=redact_synopsis_field(reason),
        project_context=project,
        run_context=", ".join(run_bits),
        what_is_approved=redact_synopsis_field(metadata.get("what_is_approved") or payload.get("what_is_approved") or reason),
        if_approved=redact_synopsis_field(metadata.get("if_approved") or payload.get("if_approved") or "the task will be unblocked and returned to the Kanban dispatcher"),
        risk_rollback=redact_synopsis_field(metadata.get("risk_rollback") or payload.get("risk_rollback") or "Deny/Needs changes leaves durable evidence and keeps the task blocked for follow-up."),
    )


def format_approval_content(req: ApprovalRequest, mentions: str = "") -> str:
    prefix = f"{mentions}\n" if mentions else ""
    lines = [
        f"{prefix}🛂 **Human Kanban approval requested**",
        f"**Task:** `{req.task_id}` — {req.title}",
        f"**Project/run:** {req.project_context}{(' | ' + req.run_context) if req.run_context else ''}",
        f"**Approve:** {redact_synopsis_field(req.what_is_approved)}",
        f"**If approved:** {redact_synopsis_field(req.if_approved)}",
        f"**Risk / rollback:** {redact_synopsis_field(req.risk_rollback)}",
        f"**Source:** {redact_synopsis_field(req.reason)}",
    ]
    return "\n".join(lines)[:1900]


def build_message_payload(req: ApprovalRequest, mentions: str = "") -> dict[str, Any]:
    return {
        "content": format_approval_content(req, mentions),
        "components": [{
            "type": 1,
            "components": [
                {"type": 2, "style": 3, "label": "Approve", "custom_id": build_custom_id(req.task_id, "approve")},
                {"type": 2, "style": 4, "label": "Deny", "custom_id": build_custom_id(req.task_id, "deny")},
                {"type": 2, "style": 2, "label": "Needs changes", "custom_id": build_custom_id(req.task_id, "needs_changes")},
            ],
        }],
    }


def resolve_db_path(raw: Optional[str] = None) -> Path:
    raw = (raw if raw is not None else os.getenv("KANBAN_DB_PATH") or os.getenv("HERMES_KANBAN_DB") or "").strip()
    if not raw or raw.lower() == "default":
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "kanban.db"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def apply_approval_decision(task_id: str, action: str, actor: str, *, db_path: Optional[Path] = None) -> str:
    if action not in APPROVAL_ACTIONS:
        raise ValueError(f"unsupported Kanban approval action: {action}")
    author = f"discord:{actor}" if actor else "discord:unknown"
    with kb.connect_closing(db_path=db_path or resolve_db_path()) as conn:
        task = conn.execute("SELECT id, status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            raise ValueError(f"unknown task {task_id}")
        label = {"approve": "APPROVED", "deny": "DENIED", "needs_changes": "NEEDS CHANGES"}[action]
        kb.add_comment(conn, task_id, author, f"{label} via Discord approval button at {int(time.time())}.")
        if action == "approve":
            changed = kb.unblock_task(conn, task_id)
            return "approved_unblocked" if changed else "approved_no_status_change"
        return f"{action}_recorded"
