"""Mission Control aggregation and local packet facade.

This module builds dashboard/API read models from existing Hermes state and
saves local review packets. It must not execute commands, start workers, reveal
secrets, or interpret worker output as instructions.
"""

from __future__ import annotations

import json
import os
import re
import secrets
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

PACKET_KINDS = {"codex_prompt", "worker_result", "operator_note", "block_flag"}
PACKET_STATUSES = {"draft", "queued", "imported", "reviewed", "archived"}
BLOCK_FLAGS = {
    "pause_future_outreach",
    "block_all_sends",
    "pause_cron_triggered_outreach",
    "disable_launch_actions",
}
BLOCKED_PACKET_ACTIONS = {
    "send_email",
    "publish_video",
    "activate_payment",
    "delete_files",
    "run_unbounded_codex",
    "autonomous_computer_use",
    "start_bulk_outreach",
    "start_codex",
    "start_worker",
    "start_hermes_run",
    "mouse_control",
    "keyboard_control",
    "browser_control",
}
MAX_PACKET_TEXT_CHARS = 100_000
PREVIEW_CHARS = 2_000

_SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|secret|token|cookie|authorization|password|client[_-]?secret|access[_-]?token|refresh[_-]?token|oauth|smtp|gmail|payment|customer)",
    re.IGNORECASE,
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bCookie\s*:\s*[^,\n\r]+"),
    re.compile(r"(?i)\b(Bearer)\s+(sk-[A-Za-z0-9._-]+)"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{8,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9._-]{8,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|oauth[_-]?token|smtp[_-]?password|gmail[_-]?token|payment[_-]?secret|customer[_-]?secret|dashboard[_-]?session[_-]?token)\s*[:=]\s*['\"]?[^'\"\s,}]+"
    ),
)


class MissionControlPacketError(ValueError):
    """Raised when a Mission Control packet request is invalid."""


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
    if raw.lower().startswith("cookie"):
        return "Cookie: [REDACTED]"
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


def _required_text(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise MissionControlPacketError(f"Missing required field: {key}")
    return value


def _optional_text(data: dict[str, Any], key: str, default: str = "") -> str:
    return str(data.get(key) or default).strip()


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    raw = data.get(key)
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raise MissionControlPacketError(f"{key} must be a list of strings")
    return [redact_text(str(item).strip()) for item in raw if str(item).strip()]


def _mission_control_state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control"


def packet_storage_dir() -> Path:
    return _mission_control_state_dir() / "packets"


def packet_audit_path() -> Path:
    return _mission_control_state_dir() / "packet-audit.jsonl"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _packet_path(packet_id: str) -> Path:
    if not re.fullmatch(r"mcpkt_[0-9TZ]+_[a-f0-9]{12}", packet_id):
        raise MissionControlPacketError("Invalid packet id")
    return packet_storage_dir() / f"{packet_id}.json"


def _new_packet_id(created_at: str) -> str:
    stamp = re.sub(r"[^0-9TZ]", "", created_at.replace("+00:00", "Z"))
    return f"mcpkt_{stamp}_{secrets.token_hex(6)}"


def _sanitize_status(value: Any, default: str) -> str:
    status = str(value or default).strip() or default
    if status not in PACKET_STATUSES:
        raise MissionControlPacketError(f"Invalid status: {status}")
    return status


def _payload_preview(payload: Any) -> str:
    text = json.dumps(redact_value(payload), sort_keys=True, ensure_ascii=False)
    if len(text) > PREVIEW_CHARS:
        return text[:PREVIEW_CHARS] + "...[truncated]"
    return text


def _packet_warnings(data: dict[str, Any], payload: Any) -> list[str]:
    warnings: list[str] = []
    requested_safety = {
        "dry_run": data.get("dry_run"),
        "review_required": data.get("review_required"),
        "trusted_for_execution": data.get("trusted_for_execution"),
    }
    if isinstance(data.get("payload"), dict):
        requested_safety.update(
            {
                "payload.dry_run": data["payload"].get("dry_run"),
                "payload.review_required": data["payload"].get("review_required"),
                "payload.trusted_for_execution": data["payload"].get("trusted_for_execution"),
            }
        )
    if any(value is False or value is True for value in requested_safety.values()):
        warnings.append("Safety fields are forced by Mission Control packet policy.")

    rendered = json.dumps(payload, sort_keys=True, default=str).lower()
    blocked = sorted(action for action in BLOCKED_PACKET_ACTIONS if action in rendered)
    if blocked:
        warnings.append(
            "Blocked requested action text was preserved as inert data only: "
            + ", ".join(blocked)
        )
    return warnings


def _append_packet_audit(
    event: str,
    *,
    packet: Optional[dict[str, Any]] = None,
    project: str = "",
    result: str = "ok",
    warnings: Optional[list[str]] = None,
    actor: str = "dashboard",
    surface: str = "dashboard",
) -> None:
    record = {
        "timestamp": _now_iso(),
        "event": event,
        "actor": redact_text(actor or "dashboard"),
        "surface": redact_text(surface or "dashboard"),
        "packet_id": packet.get("id") if packet else None,
        "packet_kind": packet.get("kind") if packet else None,
        "project": redact_text(project or str(packet.get("project") if packet else "")),
        "dry_run": True if packet is None else bool(packet.get("dry_run") is True),
        "review_required": True if packet is None else bool(packet.get("review_required") is True),
        "trusted_for_execution": False,
        "result": redact_text(result),
        "warnings": redact_value(warnings or []),
    }
    path = packet_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_value(record), sort_keys=True) + "\n")


def _text_field(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    payload = data.get("payload")
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise MissionControlPacketError(f"Missing required field: {keys[0]}")


def _bounded_text(text: str, warnings: list[str]) -> str:
    if len(text) > MAX_PACKET_TEXT_CHARS:
        warnings.append(f"Input text truncated to {MAX_PACKET_TEXT_CHARS} characters.")
        return text[:MAX_PACKET_TEXT_CHARS]
    return text


def parse_worker_result_metadata(text: str) -> dict[str, Any]:
    """Extract display-only metadata from worker text without executing it."""
    safe_text = redact_text(text[:MAX_PACKET_TEXT_CHARS])
    lines = safe_text.splitlines()
    metadata: dict[str, Any] = {
        "repo_path": None,
        "branch": None,
        "head_before": None,
        "head_after": None,
        "commit_refs": [],
        "changed_files": [],
        "tests_run": [],
        "passed_count": None,
        "failed_count": None,
        "risks_blockers": [],
        "next_prompt": None,
        "trusted_for_execution": False,
    }
    section: Optional[str] = None
    next_prompt_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if not stripped:
            continue
        if lower.startswith(("repo path:", "repo:")):
            metadata["repo_path"] = stripped.split(":", 1)[1].strip()
            section = None
            continue
        if lower.startswith("branch:"):
            metadata["branch"] = stripped.split(":", 1)[1].strip()
            section = None
            continue
        if lower.startswith("head before:"):
            metadata["head_before"] = stripped.split(":", 1)[1].strip()
            section = None
            continue
        if lower.startswith("head after:"):
            metadata["head_after"] = stripped.split(":", 1)[1].strip()
            section = None
            continue
        if lower.startswith("commit:"):
            commit = stripped.split(":", 1)[1].strip()
            if commit:
                metadata["commit_refs"].append(commit)
            section = None
            continue
        if lower.startswith("changed files"):
            section = "changed_files"
            continue
        if lower.startswith("tests run"):
            section = "tests_run"
            continue
        if lower.startswith(("risks/blockers", "risks", "blockers")):
            section = "risks_blockers"
            continue
        if lower.startswith("next implementation prompt"):
            section = "next_prompt"
            continue
        if section == "changed_files" and stripped.startswith(("-", "*")):
            metadata["changed_files"].append(stripped.lstrip("-* ").strip())
            continue
        if section == "tests_run" and stripped.startswith(("-", "*")):
            metadata["tests_run"].append(stripped.lstrip("-* ").strip())
            _merge_test_counts(metadata, stripped)
            continue
        if section == "risks_blockers" and stripped.startswith(("-", "*")):
            metadata["risks_blockers"].append(stripped.lstrip("-* ").strip())
            continue
        if section == "next_prompt":
            next_prompt_lines.append(stripped)
            continue
        _merge_test_counts(metadata, stripped)
    if next_prompt_lines:
        metadata["next_prompt"] = "\n".join(next_prompt_lines)[:PREVIEW_CHARS]
    return redact_value(metadata)


def _merge_test_counts(metadata: dict[str, Any], text: str) -> None:
    passed = re.search(r"(\d+)\s+passed", text, re.IGNORECASE)
    failed = re.search(r"(\d+)\s+failed", text, re.IGNORECASE)
    if passed:
        metadata["passed_count"] = int(passed.group(1))
    if failed:
        metadata["failed_count"] = int(failed.group(1))


def _build_packet(
    *,
    kind: str,
    data: dict[str, Any],
    payload: dict[str, Any],
    status: str,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    if kind not in PACKET_KINDS:
        raise MissionControlPacketError(f"Invalid packet kind: {kind}")
    project = _required_text(data, "project")
    title = _required_text(data, "title")
    created_at = _now_iso()
    redacted_payload = redact_value(payload)
    packet_warnings = list(warnings or []) + _packet_warnings(data, redacted_payload)
    packet = {
        "id": _new_packet_id(created_at),
        "kind": kind,
        "project": redact_text(project),
        "title": redact_text(title),
        "payload": redacted_payload,
        "redacted_payload_preview": _payload_preview(redacted_payload),
        "source_refs": _string_list(data, "source_refs"),
        "approval_gates": redact_value(data.get("approval_gates") or list(STANDING_GATES)),
        "dry_run": True,
        "review_required": True,
        "trusted_for_execution": False,
        "status": _sanitize_status(data.get("status"), status),
        "author": redact_text(_optional_text(data, "author", "dashboard") or "dashboard"),
        "created_at": created_at,
        "updated_at": created_at,
        "warnings": redact_value(packet_warnings),
    }
    return packet


def _save_packet(packet: dict[str, Any], audit_events: Iterable[str]) -> dict[str, Any]:
    path = _packet_path(packet["id"])
    if path.exists():
        raise MissionControlPacketError("Packet id already exists")
    _atomic_write_json(path, redact_value(packet))
    for event in ["packet_created", *audit_events]:
        _append_packet_audit(
            event,
            packet=packet,
            project=str(packet.get("project") or ""),
            warnings=list(packet.get("warnings") or []),
        )
    return packet


def save_next_codex_prompt(data: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    prompt = _bounded_text(_text_field(data, "prompt", "codex_prompt", "text"), warnings)
    payload = {
        "prompt": prompt,
        "operator_intent": _optional_text(data, "operator_intent"),
        "trusted_for_execution": False,
        "execution_policy": "saved_for_review_only",
        "blocked_execution_classes": sorted(BLOCKED_PACKET_ACTIONS),
    }
    if isinstance(data.get("payload"), dict):
        payload["request_context"] = data["payload"]
    packet = _build_packet(
        kind="codex_prompt",
        data=data,
        payload=payload,
        status="draft",
        warnings=warnings,
    )
    return _save_packet(packet, ["codex_prompt_saved"])


def import_worker_result(data: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    text = _bounded_text(_text_field(data, "worker_result", "result_text", "text"), warnings)
    payload = {
        "worker_result": text,
        "parsed_metadata": parse_worker_result_metadata(text),
        "trusted_for_execution": False,
        "execution_policy": "imported_as_untrusted_data_only",
    }
    packet = _build_packet(
        kind="worker_result",
        data={**data, "status": "imported"},
        payload=payload,
        status="imported",
        warnings=warnings,
    )
    return _save_packet(packet, ["worker_result_imported"])


def set_block_flag(data: dict[str, Any]) -> dict[str, Any]:
    flag = _required_text(data, "flag")
    if flag not in BLOCK_FLAGS:
        raise MissionControlPacketError(f"Invalid block flag: {flag}")
    payload = {
        "flag": flag,
        "reason": _required_text(data, "reason"),
        "advisory_only": True,
        "local_state_updated": False,
        "safe_state_hook": None,
        "trusted_for_execution": False,
    }
    packet = _build_packet(
        kind="block_flag",
        data=data,
        payload=payload,
        status="draft",
        warnings=["No explicit safe block-flag state hook exists; saved as advisory packet only."],
    )
    return _save_packet(packet, ["block_flag_packet_saved"])


def create_rejection_audit(data: dict[str, Any], detail: str, *, packet_kind: str = "") -> None:
    project = str(data.get("project") or "")
    _append_packet_audit(
        "packet_rejected",
        project=project,
        result=detail,
        warnings=[detail, f"packet_kind={packet_kind}" if packet_kind else ""],
        actor=redact_text(str(data.get("author") or "dashboard")),
    )


def list_packets(limit: int = 100) -> dict[str, Any]:
    response = _base_response("mission_control_packets")
    path = packet_storage_dir()
    response["source_refs"].append(str(path))
    if not path.exists():
        response["warnings"].append(f"Mission Control packet directory not found: {path}")
        return response
    files = sorted(path.glob("mcpkt_*.json"), key=lambda item: item.name, reverse=True)
    for file_path in files[: int(limit)]:
        try:
            packet = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            response["warnings"].append(f"Could not read packet {file_path.name}: {exc}")
            continue
        if not isinstance(packet, dict):
            response["warnings"].append(f"Packet file is not a JSON object: {file_path.name}")
            continue
        response["items"].append(redact_value(_packet_summary(packet)))
    return response


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": packet.get("id"),
        "kind": packet.get("kind"),
        "project": packet.get("project"),
        "title": packet.get("title"),
        "status": packet.get("status"),
        "dry_run": packet.get("dry_run") is True,
        "review_required": packet.get("review_required") is True,
        "trusted_for_execution": False,
        "author": packet.get("author"),
        "created_at": packet.get("created_at"),
        "updated_at": packet.get("updated_at"),
        "redacted_payload_preview": packet.get("redacted_payload_preview"),
        "source_ref_count": len(packet.get("source_refs") or []),
        "warnings": packet.get("warnings") or [],
    }


def get_packet(packet_id: str) -> dict[str, Any]:
    path = _packet_path(packet_id)
    if not path.exists():
        raise FileNotFoundError(packet_id)
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MissionControlPacketError(f"Could not read packet: {exc}") from exc
    if not isinstance(packet, dict):
        raise MissionControlPacketError("Packet file is not a JSON object")
    return {
        "generated_at": _now_iso(),
        "source": "mission_control_packet",
        "source_refs": [str(path)],
        "packet": redact_value(packet),
        "warnings": [],
    }


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
