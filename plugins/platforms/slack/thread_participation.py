"""Durable, profile-scoped Slack thread participation state."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)


class SlackThreadParticipationStore:
    """Persistent bounded set of threads this bot explicitly left."""

    MAX_ENTRIES = 5000

    def __init__(self, path: Optional[Path] = None, *, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path or get_hermes_home() / "gateway" / "slack_left_threads.json"
        self._max_entries = max(2, max_entries)
        self._lock = threading.RLock()
        self._entries: dict[str, float] = {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
            if isinstance(raw, dict):
                self._entries = {
                    str(key): float(marked_at)
                    for key, marked_at in raw.items()
                    if isinstance(marked_at, (int, float))
                }
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("Slack thread participation state could not be loaded: %s", exc)
        if self._prune_locked():
            self._flush_locked()

    @staticmethod
    def _key(team_id: str, channel_id: str, thread_ts: str) -> str:
        return json.dumps([str(team_id), str(channel_id), str(thread_ts)], separators=(",", ":"))

    def _prune_locked(self) -> bool:
        if len(self._entries) <= self._max_entries:
            return False
        keep = self._max_entries // 2
        newest = sorted(self._entries.items(), key=lambda item: item[1], reverse=True)[:keep]
        self._entries = dict(newest)
        return True

    def _flush_locked(self) -> None:
        try:
            atomic_json_write(self._path, self._entries, indent=None, separators=(",", ":"))
        except OSError as exc:
            logger.debug("Slack thread participation state could not be persisted: %s", exc)

    def is_muted(self, team_id: str, channel_id: str, thread_ts: str) -> bool:
        with self._lock:
            return self._key(team_id, channel_id, thread_ts) in self._entries

    def mute(self, team_id: str, channel_id: str, thread_ts: str) -> None:
        with self._lock:
            self._entries[self._key(team_id, channel_id, thread_ts)] = time.time()
            self._prune_locked()
            self._flush_locked()

    def unmute(self, team_id: str, channel_id: str, thread_ts: str) -> bool:
        with self._lock:
            removed = self._entries.pop(self._key(team_id, channel_id, thread_ts), None) is not None
            if removed:
                self._flush_locked()
            return removed
