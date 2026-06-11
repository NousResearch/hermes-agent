"""Persistent JSON storage for notification events."""
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .base import ServiceEvent

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "~/.hermes/notification_events.json"
_MAX_EVENTS = 200


class NotificationStore:
    """Simple JSON-file-backed event store."""

    def __init__(self, path: Optional[str] = None, max_events: int = _MAX_EVENTS):
        self._path = Path(os.path.expanduser(path or _DEFAULT_PATH))
        self._max_events = max_events
        self._events: List[dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._events = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(self._events, list):
                    self._events = []
            except Exception as e:
                logger.warning("Failed to load notification store: %s", e)
                self._events = []

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, event: ServiceEvent):
        self._events.append(asdict(event))
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        self._save()

    def query(self, since: Optional[str] = None, app: Optional[str] = None,
              limit: int = 20) -> List[dict]:
        """Query stored events, newest first."""
        results = list(reversed(self._events))
        if app:
            results = [e for e in results if e.get("app") == app]
        if since:
            results = [e for e in results if e.get("timestamp", "") >= since]
        return results[:limit]

    def clear(self):
        self._events = []
        self._save()

    def count(self) -> int:
        return len(self._events)
