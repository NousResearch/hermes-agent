# -*- coding: utf-8 -*-
"""QQBot WebSocket session persistence.

Stores session_id, seq, intents, and last-active timestamp per app_id in
``ws_sessions.json``.  On startup the adapter loads the saved
session and attempts a Resume instead of a fresh Identify — this avoids
losing the sequence number and allows seamless reconnection after a
gateway restart.

File format::

    {
        "<app_id>": {
            "session_id": "...",
            "seq": 42,
            "intents": 1073807360,
            "last_active": "2026-04-23T10:00:00",
            "bot_username": "MyBot"
        }
    }

Zero hermes dependencies — uses only stdlib + pathlib.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "ws_sessions.json"

# Sessions older than this are considered stale and won't be resumed.
_MAX_SESSION_AGE_SECONDS = 3600  # 1 hour


@dataclass
class PersistedSession:
    """Persisted WebSocket session state for one app_id."""

    session_id: str = ""
    seq: Optional[int] = None
    intents: int = 0
    last_active: str = ""
    bot_username: str = ""

    @property
    def is_resumable(self) -> bool:
        """Return True if this session has enough data to attempt Resume."""
        return bool(self.session_id) and self.seq is not None

    @property
    def age_seconds(self) -> float:
        """Seconds since last_active, or inf if not set."""
        if not self.last_active:
            return float("inf")
        try:
            dt = datetime.fromisoformat(self.last_active)
            now = datetime.now(tz=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt).total_seconds()
        except (ValueError, TypeError):
            return float("inf")

    def is_fresh(self, max_age: float = _MAX_SESSION_AGE_SECONDS) -> bool:
        """Return True if the session is recent enough to resume."""
        return self.age_seconds < max_age


class WSSessionStore:
    """Read/write QQBot WebSocket sessions to a JSON file.

    :param base_dir: Directory containing the JSON file (e.g. ``~/.hermes``).
    :param filename: JSON filename (default ``ws_sessions.json``).
    """

    def __init__(self, base_dir: str, filename: str = _DEFAULT_PATH) -> None:
        self._path = Path(base_dir) / filename
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self._path, exc)
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", self._path, exc)

    def get(self, app_id: str) -> PersistedSession:
        """Load the persisted session for *app_id*."""
        raw = self._data.get(app_id)
        if not raw or not isinstance(raw, dict):
            return PersistedSession()
        return PersistedSession(
            session_id=str(raw.get("session_id", "")),
            seq=raw.get("seq"),
            intents=int(raw.get("intents", 0)),
            last_active=str(raw.get("last_active", "")),
            bot_username=str(raw.get("bot_username", "")),
        )

    def save(
        self,
        app_id: str,
        session_id: str,
        seq: Optional[int],
        intents: int,
        bot_username: str = "",
    ) -> None:
        """Persist session state for *app_id*."""
        self._data[app_id] = {
            "session_id": session_id,
            "seq": seq,
            "intents": intents,
            "last_active": datetime.now(tz=timezone.utc).isoformat(),
            "bot_username": bot_username,
        }
        self._save()

    def clear(self, app_id: str) -> None:
        """Remove persisted session for *app_id*."""
        if app_id in self._data:
            del self._data[app_id]
            self._save()

    def touch(self, app_id: str) -> None:
        """Update last_active timestamp without changing session data."""
        if app_id in self._data:
            self._data[app_id]["last_active"] = datetime.now(tz=timezone.utc).isoformat()
            self._save()
