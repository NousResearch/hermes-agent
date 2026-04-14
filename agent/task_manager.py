"""Structured task persistence for Hermes Agent."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hermes_constants import get_hermes_home


VALID_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
VALID_KINDS = {"local", "delegated", "system"}
VALID_ASSIGNEES = {"local", "delegate", "system"}

_ALLOWED_TRANSITIONS = {
    "pending": {"running", "cancelled"},
    "running": {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


class TaskManager:
    """Manage structured tasks with JSON persistence."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        session_id: str = "default",
    ):
        self.storage_path = Path(storage_path) if storage_path else get_hermes_home() / "state" / "tasks.json"
        self.session_id = str(session_id or "default")
        self._tasks: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _validate_kind(kind: str) -> str:
        normalized = str(kind).strip().lower()
        if normalized not in VALID_KINDS:
            raise ValueError(f"Invalid task kind: {kind}")
        return normalized

    @staticmethod
    def _validate_assignee(assignee: str | None) -> str | None:
        if assignee is None:
            return None
        normalized = str(assignee).strip().lower()
        if normalized not in VALID_ASSIGNEES:
            raise ValueError(f"Invalid task assignee: {assignee}")
        return normalized

    @staticmethod
    def _validate_status(status: str) -> str:
        normalized = str(status).strip().lower()
        if normalized not in VALID_STATUSES:
            raise ValueError(f"Invalid task status: {status}")
        return normalized

    @staticmethod
    def _copy_metadata(metadata: dict | None) -> dict | None:
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        return copy.deepcopy(metadata)

    @staticmethod
    def _task_session_id(task: dict) -> str:
        return str(task.get("session_id") or "default")

    def create(
        self,
        title,
        kind: str = "local",
        parent_task_id: str | None = None,
        assignee: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        title_text = str(title).strip()
        if not title_text:
            raise ValueError("title is required")

        now = self._now_iso()
        task = {
            "task_id": str(uuid4()),
            "title": title_text,
            "status": "pending",
            "kind": self._validate_kind(kind),
            "session_id": self.session_id,
            "created_at": now,
            "updated_at": now,
            "parent_task_id": str(parent_task_id).strip() if parent_task_id else None,
            "assignee": self._validate_assignee(assignee),
            "result_summary": None,
            "metadata": self._copy_metadata(metadata),
        }
        self._tasks[task["task_id"]] = task
        self._save()
        return copy.deepcopy(task)

    def update(
        self,
        task_id,
        status: str | None = None,
        result_summary: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        task_key = str(task_id).strip()
        task = self._tasks.get(task_key)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")

        updated = copy.deepcopy(task)
        if status is not None:
            next_status = self._validate_status(status)
            current_status = updated["status"]
            if next_status != current_status:
                allowed_statuses = _ALLOWED_TRANSITIONS[current_status]
                if next_status not in allowed_statuses:
                    raise ValueError(f"Invalid status transition: {current_status} -> {next_status}")
                updated["status"] = next_status

        if result_summary is not None:
            updated["result_summary"] = str(result_summary)

        if metadata is not None:
            updated["metadata"] = self._copy_metadata(metadata)

        updated["updated_at"] = self._now_iso()
        self._tasks[task_key] = updated
        self._save()
        return copy.deepcopy(updated)

    def get(self, task_id, include_all: bool = False) -> dict | None:
        task = self._tasks.get(str(task_id).strip())
        if task is None:
            return None
        if not include_all and self._task_session_id(task) != self.session_id:
            return None
        return copy.deepcopy(task)

    def list(
        self,
        status: str | None = None,
        kind: str | None = None,
        include_all: bool = False,
    ) -> list[dict]:
        normalized_status = self._validate_status(status) if status is not None else None
        normalized_kind = self._validate_kind(kind) if kind is not None else None

        results = []
        for task in self._tasks.values():
            if not include_all and self._task_session_id(task) != self.session_id:
                continue
            if normalized_status is not None and task["status"] != normalized_status:
                continue
            if normalized_kind is not None and task["kind"] != normalized_kind:
                continue
            results.append(copy.deepcopy(task))
        return results

    def cancel(self, task_id) -> dict:
        return self.update(task_id, status="cancelled")

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tasks": list(self._tasks.values())}
        self.storage_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._tasks = {}
            return

        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._tasks = {}
            return

        task_items = data.get("tasks", []) if isinstance(data, dict) else data
        if not isinstance(task_items, list):
            self._tasks = {}
            return
        loaded_tasks: dict[str, dict] = {}
        for item in task_items:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", "")).strip()
            if not task_id:
                continue
            loaded_tasks[task_id] = copy.deepcopy(item)
        self._tasks = loaded_tasks
