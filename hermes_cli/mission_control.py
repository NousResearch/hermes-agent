"""Read-only Mission Control aggregation facade.

This module builds dashboard/API read models from existing Hermes state. It
must not execute commands, mutate local state, reveal secrets, or interpret
worker output as instructions.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_STATUS_SOURCES: list[dict[str, str]] = [
    {
        "name": "Hermes Ops",
        "project": "Hermes Ops",
        "profile": "default",
        "path": "/home/jenny/ai-ops-brain/ai-ops/PROJECT_STATUS.md",
    },
    {
        "name": "Main Jenny",
        "project": "Main Jenny",
        "profile": "default",
        "path": "/home/jenny/ai-ops-brain/PROJECT_COMMAND_CENTER.md",
    },
    {
        "name": "Research Ops",
        "project": "Research Ops",
        "profile": "default",
        "path": "/home/jenny/ai-ops-brain/ai-ops/research-ops/PROJECT_STATUS.md",
    },
    {
        "name": "Family Hub",
        "project": "Family Hub",
        "profile": "family-hub",
        "path": "/home/jenny/ai-ops-brain/family-hub/PROJECT_STATUS.md",
    },
    {
        "name": "Tool & Tally",
        "project": "Tool & Tally",
        "profile": "no-call-estimateready",
        "path": "/home/jenny/ai-ops-brain/business/no-call-lead-engine-business-os/PROJECT_STATUS.md",
    },
    {
        "name": "VendorProof",
        "project": "VendorProof",
        "profile": "default",
        "path": "/home/jenny/ai-ops-brain/business/vendorproof-agentic-build/PROJECT_STATUS.md",
    },
    {
        "name": "Video Channel",
        "project": "Video Channel",
        "profile": "money-signal-video",
        "path": "/home/jenny/ai-ops-brain/social-video/signal-room/PROJECT_STATUS.md",
    },
    {
        "name": "Impossible Footage",
        "project": "Impossible Footage",
        "profile": "money-signal-video",
        "path": "/home/jenny/ai-ops-brain/social-video/impossible-footage/PROJECT_STATUS.md",
    },
    {
        "name": "Waha Inspection",
        "project": "Waha Inspection",
        "profile": "wahainspection",
        "path": "/home/jenny/ai-ops-brain/waha/PROJECT_STATUS.md",
    },
]

STANDING_GATES: list[dict[str, str]] = [
    {
        "id": "live-gateway-restart",
        "title": "Live gateway/service restart or reset",
        "risk_label": "Live-service",
        "posture": "blocked_until_current_approval",
    },
    {
        "id": "public-payment-customer-action",
        "title": "Public, payment, outreach, or customer action",
        "risk_label": "Money/customer",
        "posture": "blocked_until_current_approval",
    },
    {
        "id": "credential-auth-change",
        "title": "Credential, OAuth, token, or account change",
        "risk_label": "Credential/auth",
        "posture": "blocked_until_current_approval",
    },
    {
        "id": "destructive-data-change",
        "title": "Destructive cleanup, migration, or reorganization",
        "risk_label": "Destructive",
        "posture": "blocked_until_current_approval",
    },
]

_SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|secret|token|cookie|authorization|password|client[_-]?secret|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(Bearer)\s+(sk-[A-Za-z0-9._-]+)"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{8,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password)\s*[:=]\s*['\"]?[^'\"\s,}]+"
    ),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_text(value: str) -> str:
    """Redact likely secrets from untrusted text."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: _redacted_replacement(match.group(0)), text)
    return text


def _redacted_replacement(raw: str) -> str:
    prefix = raw.split(":", 1)[0] if ":" in raw else raw.split("=", 1)[0]
    if _SENSITIVE_KEYS.search(prefix):
        sep = ":" if ":" in raw else "=" if "=" in raw else ""
        return f"{prefix}{sep} [REDACTED]" if sep else "[REDACTED]"
    if raw.lower().startswith("authorization"):
        return "Authorization: Bearer [REDACTED]"
    if raw.lower().startswith("bearer"):
        return "Bearer [REDACTED]"
    return "[REDACTED]"


def redact_value(value: Any, *, key: str = "") -> Any:
    """Recursively redact likely secrets from JSON-compatible data."""
    if _SENSITIVE_KEYS.search(key or ""):
        return "[REDACTED]" if value not in (None, "") else value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): redact_value(v, key=str(k)) for k, v in value.items()}
    return value


def _base_response(source: str) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "source": source,
        "source_refs": [],
        "items": [],
        "warnings": [],
    }


def _read_text_preview(path: Path, *, max_chars: int = 4000) -> tuple[Optional[str], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"
    return redact_text(text[:max_chars]), None


def project_status() -> dict[str, Any]:
    response = _base_response("project_status_files")
    for source in PROJECT_STATUS_SOURCES:
        path = Path(source["path"])
        item = {
            "name": source.get("name"),
            "project": source.get("project"),
            "profile": source.get("profile"),
            "path": str(path),
            "exists": path.exists(),
            "excerpt": None,
        }
        response["source_refs"].append(str(path))
        if path.exists():
            excerpt, warning = _read_text_preview(path)
            if warning:
                response["warnings"].append(warning)
            item["excerpt"] = excerpt
        else:
            response["warnings"].append(f"Project status source missing: {path}")
        response["items"].append(redact_value(item))
    return response


def _kanban_db_path() -> Path:
    from hermes_cli import kanban_db

    return kanban_db.kanban_db_path()


def _open_readonly_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _with_kanban_conn(response: dict[str, Any]) -> Optional[sqlite3.Connection]:
    path = _kanban_db_path()
    response["source_refs"].append(str(path))
    if not path.exists():
        response["warnings"].append(f"Kanban database not found: {path}")
        return None
    try:
        return _open_readonly_sqlite(path)
    except Exception as exc:
        response["warnings"].append(f"Kanban database unavailable: {exc}")
        return None


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def open_tasks(limit: int = 100) -> dict[str, Any]:
    response = _base_response("kanban_tasks")
    conn = _with_kanban_conn(response)
    if conn is None:
        return response
    try:
        rows = conn.execute(
            """
            SELECT id, title, body, assignee, status, priority, created_by,
                   created_at, started_at, completed_at, tenant,
                   last_failure_error, current_run_id
              FROM tasks
             WHERE status NOT IN ('done', 'archived')
             ORDER BY priority DESC, created_at ASC
             LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except Exception as exc:
        response["warnings"].append(f"Could not read Kanban tasks: {exc}")
        return response
    finally:
        conn.close()

    for row in rows:
        body = _row_value(row, "body")
        item = {
            "id": row["id"],
            "title": row["title"],
            "body_preview": str(body)[:500] if body else None,
            "assignee": row["assignee"],
            "status": row["status"],
            "priority": row["priority"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "tenant": row["tenant"],
            "last_failure_error": row["last_failure_error"],
            "current_run_id": row["current_run_id"],
            "trusted_for_execution": False,
        }
        response["items"].append(redact_value(item))
    return response


def latest_worker_results(limit: int = 50) -> dict[str, Any]:
    response = _base_response("kanban_task_runs")
    conn = _with_kanban_conn(response)
    if conn is None:
        return response
    try:
        rows = conn.execute(
            """
            SELECT r.id AS run_id, r.task_id, r.profile, r.step_key, r.status,
                   r.started_at, r.ended_at, r.outcome, r.summary, r.error,
                   r.metadata, t.title, t.assignee, t.status AS task_status
              FROM task_runs r
              LEFT JOIN tasks t ON t.id = r.task_id
             WHERE (r.summary IS NOT NULL AND r.summary != '')
                OR (r.error IS NOT NULL AND r.error != '')
                OR (r.metadata IS NOT NULL AND r.metadata != '')
             ORDER BY COALESCE(r.ended_at, r.started_at) DESC, r.id DESC
             LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except Exception as exc:
        response["warnings"].append(f"Could not read Kanban worker results: {exc}")
        return response
    finally:
        conn.close()

    for row in rows:
        metadata = None
        raw_metadata = row["metadata"]
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except Exception:
                metadata = {"_warning": "metadata was not valid JSON"}
        item = {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "title": row["title"],
            "assignee": row["assignee"],
            "task_status": row["task_status"],
            "profile": row["profile"],
            "step_key": row["step_key"],
            "run_status": row["status"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "outcome": row["outcome"],
            "summary": row["summary"],
            "error": row["error"],
            "metadata": metadata,
            "trusted_for_execution": False,
        }
        response["items"].append(redact_value(item))
    return response


def repo_status() -> dict[str, Any]:
    response = _base_response("repo_status_warning_only")
    response["probing_enabled"] = False
    response["warnings"].append(
        "Repo probing is not implemented in Phase 1; no shell commands were executed."
    )
    paths = [PROJECT_ROOT]
    paths.extend(Path(source["path"]).parent for source in PROJECT_STATUS_SOURCES)
    seen: set[str] = set()
    for path in paths:
        path_text = str(path)
        if path_text in seen:
            continue
        seen.add(path_text)
        response["source_refs"].append(path_text)
        response["items"].append(
            {
                "path": path_text,
                "exists": path.exists(),
                "probing_enabled": False,
                "status": "not_probed",
            }
        )
    return response


def approval_gates() -> dict[str, Any]:
    response = _base_response("ops_approvals")
    response["standing_gates"] = list(STANDING_GATES)
    response["execution_posture"] = {
        "read_only_default": True,
        "decision_records_only": True,
        "fixed_actions_only": True,
        "action_execution_disabled_by_default": True,
    }
    try:
        from hermes_cli.ops_approvals import ApprovalStore

        store = ApprovalStore()
        response["source_refs"].extend([str(store.inbox_path), str(store.audit_path)])
        approvals = store.list(now=datetime.now(timezone.utc))
    except Exception as exc:
        response["warnings"].append(f"Could not read approval inbox: {exc}")
        approvals = []

    pending = [item for item in approvals if item.get("status") == "pending"]
    response["items"] = redact_value(approvals)
    response["summary"] = {
        "pending_count": len(pending),
        "total_count": len(approvals),
        "blocked_execution": True,
    }
    try:
        from hermes_cli.ops_actions import action_registry_status

        response["action_registry"] = redact_value(action_registry_status())
    except Exception as exc:
        response["warnings"].append(f"Could not read fixed action registry: {exc}")
        response["action_registry"] = {
            "execution_enabled": False,
            "allowed_actions": [],
            "actions": [],
            "blocked_action_classes": [],
        }
    return response


def _iter_jsonl(path: Path, warnings: list[str]) -> Iterable[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"Audit log not found: {path}")
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        warnings.append(f"Could not read audit log {path}: {exc}")
        return []
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            malformed += 1
    if malformed:
        warnings.append(f"Skipped {malformed} malformed audit log line(s) in {path}")
    return events


def recent_audit_log(limit: int = 50) -> dict[str, Any]:
    response = _base_response("ops_approval_audit")
    try:
        from hermes_cli.ops_approvals import ApprovalStore

        store = ApprovalStore()
        path = store.audit_path
    except Exception as exc:
        response["warnings"].append(f"Could not resolve approval audit path: {exc}")
        return response
    response["source_refs"].append(str(path))
    events = list(_iter_jsonl(path, response["warnings"]))
    response["items"] = redact_value(events[-int(limit):][::-1])
    return response
