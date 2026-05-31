"""Durable active task/workspace state for gateway resume recovery."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write


DEFAULT_ACTIVE_TASK_TTL_SECONDS = 48 * 60 * 60
ACTIVE_TASK_STATUSES = {"active", "interrupted", "detached", "unknown"}
logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class ActiveTaskRecord:
    session_key: str
    session_id: Optional[str] = None
    platform: Optional[str] = None
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    repo_path: Optional[str] = None
    branch: Optional[str] = None
    head: Optional[str] = None
    mode: Optional[str] = None
    command: Optional[str] = None
    task_summary: Optional[str] = None
    status: str = "unknown"
    pid: Optional[int] = None
    process_session_id: Optional[str] = None
    latest_log_path: Optional[str] = None
    latest_summary_path: Optional[str] = None
    updated_at: str = ""
    resume_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveTaskRecord":
        fields = {name for name in cls.__dataclass_fields__}
        payload = {key: data.get(key) for key in fields if key in data}
        if not payload.get("updated_at"):
            payload["updated_at"] = _utc_now_iso()
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_fresh(self, ttl_seconds: int = DEFAULT_ACTIVE_TASK_TTL_SECONDS) -> bool:
        updated = _parse_iso_timestamp(self.updated_at)
        if updated is None:
            return False
        age = datetime.now(timezone.utc) - updated
        return age.total_seconds() <= ttl_seconds

    def has_usable_workspace(self) -> bool:
        if self.status not in ACTIVE_TASK_STATUSES:
            return False
        if not self.repo_path:
            return False
        try:
            return Path(self.repo_path).expanduser().exists()
        except OSError:
            return False


class ActiveTaskStore:
    """Small JSON store keyed by gateway session_key."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else get_hermes_home() / "session_active_tasks.json"
        self._lock = threading.Lock()

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug("active-task store file is absent: %s", self.path)
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("failed to parse active-task store %s: %s", self.path, exc)
            return {}
        except OSError as exc:
            logger.warning("failed to read active-task store %s: %s", self.path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        atomic_json_write(self.path, data, indent=2)

    def get(self, session_key: str) -> Optional[ActiveTaskRecord]:
        if not session_key:
            return None
        with self._lock:
            raw = self._read_unlocked().get(session_key)
        if not isinstance(raw, dict):
            return None
        try:
            return ActiveTaskRecord.from_dict(raw)
        except TypeError:
            return None

    def upsert(self, *, session_key: str, **fields: Any) -> ActiveTaskRecord:
        if not session_key:
            raise ValueError("session_key is required")

        with self._lock:
            data = self._read_unlocked()
            existing = data.get(session_key) if isinstance(data.get(session_key), dict) else {}
            payload = dict(existing)
            payload["session_key"] = session_key
            for key, value in fields.items():
                if key not in ActiveTaskRecord.__dataclass_fields__:
                    continue
                if key in {"pid"}:
                    payload[key] = int(value) if value is not None else None
                elif key == "status":
                    payload[key] = _clean_optional_str(value) or "unknown"
                else:
                    payload[key] = _clean_optional_str(value)
            payload["updated_at"] = _utc_now_iso()
            record = ActiveTaskRecord.from_dict(payload)
            data[session_key] = record.to_dict()
            self._write_unlocked(data)
            return record

    def replace_foreground_session(
        self,
        *,
        session_key: str,
        repo_path: str,
        branch: Optional[str] = None,
        head: Optional[str] = None,
    ) -> ActiveTaskRecord:
        if not session_key:
            raise ValueError("session_key is required")

        payload = {
            "session_key": session_key,
            "repo_path": _clean_optional_str(repo_path),
            "branch": _clean_optional_str(branch),
            "head": _clean_optional_str(head),
            "mode": "foreground_session",
            "status": "active",
            "updated_at": _utc_now_iso(),
        }
        record = ActiveTaskRecord.from_dict(payload)
        with self._lock:
            data = self._read_unlocked()
            data[session_key] = payload
            self._write_unlocked(data)
        return record


def resolve_git_branch(repo_path: str | os.PathLike[str] | None) -> Optional[str]:
    if not repo_path:
        return None
    path = Path(repo_path).expanduser()
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def resolve_git_head(repo_path: str | os.PathLike[str] | None) -> Optional[str]:
    if not repo_path:
        return None
    path = Path(repo_path).expanduser()
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    head = (result.stdout or "").strip()
    return head or None


def build_active_task_recovery_note(
    record: ActiveTaskRecord | None,
    resume_reason: str | None = None,
) -> str:
    reason = resume_reason or (record.resume_reason if record else None) or "restart_timeout"
    reason_phrase = (
        "a gateway restart"
        if reason == "restart_timeout"
        else "a gateway shutdown"
        if reason == "shutdown_timeout"
        else "a gateway interruption"
    )
    base = (
        "[System note: Your previous turn in this session was interrupted by "
        f"{reason_phrase}. The conversation history below is intact."
    )

    if record is None:
        return (
            base
            + " Active workspace/process state: unknown. Do not silently default "
            "to the gateway working directory as the active project. Report that "
            "the previous repo, branch, command, and process status are unknown "
            "before continuing. Next safe recovery check: inspect durable active "
            "task/process records for this session_key, then ask before running "
            "project commands.]"
        )

    task = record.task_summary or record.command or "unknown"
    lines = [
        base,
        f" Previous active task: {task}",
        f"Previous repo path: {record.repo_path or 'unknown'}",
        f"Previous branch: {record.branch or 'unknown'}",
        f"Previous HEAD: {record.head or 'unknown'}",
        f"Previous command: {record.command or 'unknown'}",
        f"Process status: {record.status or 'unknown'}",
    ]
    if record.process_session_id or record.pid is not None:
        process_ref = record.process_session_id or f"pid:{record.pid}"
        lines.append(f"Process/session id: {process_ref}")
    if record.latest_log_path:
        lines.append(f"Latest log path: {record.latest_log_path}")
    if record.latest_summary_path:
        lines.append(f"Latest summary path: {record.latest_summary_path}")
    lines.append(
        "Next safe recovery check: verify the process/session status and latest "
        "log or summary path before running continuation commands.]"
    )
    return "\n".join(lines)
