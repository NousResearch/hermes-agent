"""Filesystem-backed approval records for research actions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from research.models import make_id, now_utc
from research.state_store import ResearchStateStore


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ApprovalStore:
    """Manage approval records under each project."""

    def __init__(self, state_store: ResearchStateStore):
        self.state_store = state_store

    def _path(self, project_id: str, approval_id: str) -> Path:
        return self.state_store.approvals_dir(project_id) / f"{approval_id}.json"

    def request(self, project_id: str, action: str, payload: dict[str, Any], reason: str = "") -> dict[str, Any]:
        approval_id = make_id("approval")
        record = {
            "approval_id": approval_id,
            "project_id": project_id,
            "action": action,
            "status": "pending",
            "reason": reason,
            "payload": payload,
            "created_at": now_utc(),
        }
        _atomic_write_json(self._path(project_id, approval_id), record)
        return record

    def approve(self, project_id: str, approval_id: str) -> dict[str, Any]:
        record = self.get(project_id, approval_id)
        record["status"] = "approved"
        record["approved_at"] = now_utc()
        _atomic_write_json(self._path(project_id, approval_id), record)
        return record

    def get(self, project_id: str, approval_id: str) -> dict[str, Any]:
        return json.loads(self._path(project_id, approval_id).read_text(encoding="utf-8"))

    def is_approved(self, project_id: str, action: str, approval_id: str | None) -> bool:
        if not approval_id:
            return False
        try:
            record = self.get(project_id, approval_id)
        except FileNotFoundError:
            return False
        return record.get("action") == action and record.get("status") == "approved"
