#!/usr/bin/env python3
"""
InboxManager — external task inbox with JSON persistence.

Stores items at ``<project_root>/.hermes/inbox.json``.
Supports: manual, cli, feishu (stub), discord (stub), scheduler (stub).

Writeback status for each source:
  manual    — N/A (no external source to write back to)
  cli       — available (writes to ~/.hermes/inbox-results/<id>.json)
  feishu    — unavailable (stub, API not connected)
  discord   — unavailable (stub, API not connected)
  scheduler — unavailable (stub, no cron runner)
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from executors.types import (
    InboxItem,
    InboxSource,
    InboxStatus,
    InboxResultCallback,
    TaskDraft,
)

logger = logging.getLogger(__name__)

INBOX_FILENAME = "inbox.json"

# Source writeback availability
_WRITEBACK_AVAILABLE: Dict[InboxSource, bool] = {
    InboxSource.DESKTOP: False,
    InboxSource.CLI: True,
    InboxSource.FEISHU: False,
    InboxSource.DISCORD: False,
    InboxSource.SCHEDULER: False,
}


class InboxManager:
    """Manages inbox items at ``<project_root>/.hermes/inbox.json``."""

    def __init__(self, project_root: Path):
        self._project_root = Path(project_root).resolve()
        self._inbox_dir = self._project_root / ".hermes"
        self._inbox_path = self._inbox_dir / INBOX_FILENAME
        self._items: Optional[List[InboxItem]] = None

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> List[InboxItem]:
        if self._items is not None:
            return self._items
        if self._inbox_path.exists():
            try:
                raw = json.loads(self._inbox_path.read_text())
                self._items = [self._from_dict(it) for it in raw]
                return self._items
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Corrupt inbox.json: %s", e)
        self._items = []
        return self._items

    def _save(self) -> None:
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        data = [self._to_dict(it) for it in (self._items or [])]
        self._inbox_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def add(
        self,
        source: InboxSource,
        title: str,
        body: str,
        raw_payload: Optional[Dict[str, Any]] = None,
        suggested_executor: Optional[str] = None,
        project_hint: Optional[str] = None,
        priority: str = "normal",
        expires_at: Optional[datetime.datetime] = None,
    ) -> InboxItem:
        """Add a new inbox item."""
        items = self._load()
        item_id = f"inbox-{uuid.uuid4().hex[:12]}"

        draft = TaskDraft(
            title=title,
            suggested_prompt=body,
            suggested_executor=suggested_executor,
            project_hint=project_hint,
            priority=priority,
        )

        item = InboxItem(
            id=item_id,
            source=source,
            raw_payload=raw_payload or {"title": title, "body": body},
            draft=draft,
            status=InboxStatus.PENDING,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            expires_at=expires_at,
        )

        items.append(item)
        self._items = items
        self._save()
        logger.info("Inbox item added: %s [%s]", item_id, source.value)
        return item

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_items(
        self,
        status: Optional[InboxStatus] = None,
        source: Optional[InboxSource] = None,
    ) -> List[InboxItem]:
        """List inbox items, optionally filtered."""
        items = self._load()
        if status:
            items = [it for it in items if it.status == status]
        if source:
            items = [it for it in items if it.source == source]
        return items

    def list_pending(self) -> List[InboxItem]:
        return self.list_items(status=InboxStatus.PENDING)

    def get(self, item_id: str) -> Optional[InboxItem]:
        for it in self._load():
            if it.id == item_id:
                return it
        return None

    def count_by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in self._load():
            counts[it.status.value] = counts.get(it.status.value, 0) + 1
        return counts

    def count_pending_by_source(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in self.list_pending():
            counts[it.source.value] = counts.get(it.source.value, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def convert_to_task(self, item_id: str, task_id: str) -> Optional[InboxItem]:
        """Mark item as confirmed and link to a task thread."""
        item = self.get(item_id)
        if item is None:
            return None
        if item.status != InboxStatus.PENDING:
            logger.warning("Item %s not pending: %s", item_id, item.status.value)
            return item

        item.status = InboxStatus.CONFIRMED
        item.linked_task_id = task_id
        self._save()
        logger.info("Inbox item %s confirmed → task %s", item_id, task_id)
        return item

    def reject(self, item_id: str, reason: str = "") -> Optional[InboxItem]:
        """Reject an inbox item."""
        item = self.get(item_id)
        if item is None:
            return None
        item.status = InboxStatus.REJECTED
        item.rejected_reason = reason
        self._save()
        return item

    def archive(self, item_id: str) -> Optional[InboxItem]:
        """Archive an inbox item."""
        item = self.get(item_id)
        if item is None:
            return None
        item.status = InboxStatus.ARCHIVED
        self._save()
        return item

    def expire(self, item_id: str) -> Optional[InboxItem]:
        """Mark item as expired."""
        item = self.get(item_id)
        if item is None:
            return None
        item.status = InboxStatus.EXPIRED
        self._save()
        return item

    def update_draft(
        self, item_id: str,
        title: Optional[str] = None,
        prompt: Optional[str] = None,
        executor: Optional[str] = None,
        project: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Optional[InboxItem]:
        """Edit the task draft before converting."""
        item = self.get(item_id)
        if item is None:
            return None
        if title is not None:
            item.draft.title = title
            item.draft.user_edited = True
        if prompt is not None:
            item.draft.suggested_prompt = prompt
            item.draft.user_edited = True
        if executor is not None:
            item.draft.suggested_executor = executor
        if project is not None:
            item.draft.project_hint = project
        if priority is not None:
            item.draft.priority = priority
        self._save()
        return item

    # ------------------------------------------------------------------
    # Writeback
    # ------------------------------------------------------------------

    def get_writeback_callback(
        self, item_id: str, run_id: str, summary: str,
        status: str = "done", changed_files_count: int = 0,
    ) -> Optional[InboxResultCallback]:
        """Create a result callback for writeback. May be unavailable."""
        item = self.get(item_id)
        if item is None:
            return None

        available = _WRITEBACK_AVAILABLE.get(item.source, False)
        return InboxResultCallback(
            inbox_item_id=item_id,
            run_id=run_id,
            status=status,
            summary=summary[:500],
            changed_files_count=changed_files_count,
            writeback_available=available,
        )

    @staticmethod
    def writeback_destination(item: InboxItem) -> str:
        """Describe where the result would be written back."""
        if item.source == InboxSource.DESKTOP:
            return "N/A (manual entry)"
        if item.source == InboxSource.CLI:
            return f"~/.hermes/inbox-results/{item.id}.json"
        if item.source == InboxSource.FEISHU:
            return "Feishu thread (unavailable — stub)"
        if item.source == InboxSource.DISCORD:
            return "Discord channel (unavailable — stub)"
        if item.source == InboxSource.SCHEDULER:
            return "Scheduler job status (unavailable — stub)"
        return "Unknown"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(item: InboxItem) -> Dict[str, Any]:
        return {
            "id": item.id,
            "source": item.source.value,
            "raw_payload": item.raw_payload,
            "draft": asdict(item.draft),
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
            "expires_at": item.expires_at.isoformat() if item.expires_at else None,
            "linked_task_id": item.linked_task_id,
            "rejected_reason": item.rejected_reason,
        }

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> InboxItem:
        return InboxItem(
            id=d["id"],
            source=InboxSource(d["source"]),
            raw_payload=d.get("raw_payload", {}),
            draft=TaskDraft(**d.get("draft", {})),
            status=InboxStatus(d.get("status", "pending")),
            created_at=datetime.datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.datetime.now(datetime.timezone.utc),
            expires_at=datetime.datetime.fromisoformat(d["expires_at"]) if d.get("expires_at") else None,
            linked_task_id=d.get("linked_task_id"),
            rejected_reason=d.get("rejected_reason"),
        )


def create_inbox_manager(project_root: Path) -> InboxManager:
    return InboxManager(project_root)
