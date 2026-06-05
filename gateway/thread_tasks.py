"""Persistent bindings between messaging threads and Todoist tasks."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform
from gateway.session import SessionSource
from utils import atomic_json_write

_TODOIST_ID_RE = re.compile(
    r"(?:\[(?:todoist|t)[:#]\s*(?P<bracket>\d{5,})\])|"
    r"(?:\b(?:todoist|task|t)[:#]\s*(?P<plain>\d{5,})\b)|"
    r"(?:\b(?:id|task_id)=(?P<query>\d{5,})\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ThreadTaskBinding:
    """A durable task reference for a single platform chat/thread lane."""

    task_id: str
    task_title: Optional[str] = None
    source: str = "manual"
    url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "source": self.source,
            "url": self.url or todoist_task_url(self.task_id),
        }
        if self.task_title:
            data["task_title"] = self.task_title
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadTaskBinding":
        task_id = str(data.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        title = data.get("task_title")
        task_title = str(title).strip() if title is not None else ""
        return cls(
            task_id=task_id,
            task_title=task_title or None,
            source=str(data.get("source") or "manual"),
            url=str(data.get("url") or "").strip() or todoist_task_url(task_id),
        )


def todoist_task_url(task_id: str) -> str:
    return f"https://todoist.com/showTask?id={task_id}"


def parse_todoist_task_id(text: Optional[str]) -> Optional[str]:
    """Extract a Todoist task id from a thread title or command argument."""

    if not text:
        return None
    match = _TODOIST_ID_RE.search(text)
    if not match:
        return None
    return (
        match.group("bracket")
        or match.group("plain")
        or match.group("query")
        or ""
    ).strip() or None


def _platform_value(platform: Platform | str | None) -> str:
    value = getattr(platform, "value", platform)
    return str(value or "").strip().lower()


def thread_binding_key(source: SessionSource) -> Optional[str]:
    """Return a stable key for a chat/thread lane, or None when unavailable."""

    if not source or not source.chat_id:
        return None
    platform = _platform_value(source.platform)
    if not platform:
        return None
    chat_id = str(source.chat_id)
    thread_id = str(source.thread_id or "")
    parent_chat_id = str(source.parent_chat_id or "")
    guild_id = str(source.guild_id or "")
    return "|".join((platform, guild_id, parent_chat_id, chat_id, thread_id))


class ThreadTaskBindingStore:
    """Small JSON-backed store for platform-thread to Todoist-task bindings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = raw if isinstance(raw, dict) else {}
            except FileNotFoundError:
                self._data = {}
            except Exception:
                self._data = {}
            self._loaded = True

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.path, self._data)

    def bind(
        self,
        source: SessionSource,
        task_id: str,
        *,
        task_title: Optional[str] = None,
        source_label: str = "manual",
    ) -> ThreadTaskBinding:
        key = thread_binding_key(source)
        if not key:
            raise ValueError("cannot bind task without a platform chat id")
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("task id is required")
        binding = ThreadTaskBinding(
            task_id=task_id,
            task_title=(task_title or "").strip() or None,
            source=source_label,
            url=todoist_task_url(task_id),
        )
        with self._lock:
            self._ensure_loaded()
            self._data[key] = binding.to_dict()
            self._save()
        return binding

    def unbind(self, source: SessionSource) -> Optional[ThreadTaskBinding]:
        key = thread_binding_key(source)
        if not key:
            return None
        with self._lock:
            self._ensure_loaded()
            raw = self._data.pop(key, None)
            if raw is not None:
                self._save()
        if isinstance(raw, dict):
            return ThreadTaskBinding.from_dict(raw)
        return None

    def get(self, source: SessionSource) -> Optional[ThreadTaskBinding]:
        key = thread_binding_key(source)
        if not key:
            return None
        with self._lock:
            self._ensure_loaded()
            raw = self._data.get(key)
        if isinstance(raw, dict):
            try:
                return ThreadTaskBinding.from_dict(raw)
            except Exception:
                return None
        return None

    def resolve_for_source(self, source: SessionSource) -> Optional[ThreadTaskBinding]:
        binding = self.get(source)
        if binding:
            return binding
        parsed = parse_todoist_task_id(source.chat_name)
        if not parsed:
            parsed = parse_todoist_task_id(source.chat_topic)
        if parsed:
            return ThreadTaskBinding(task_id=parsed, source="thread-title")
