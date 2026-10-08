"""Reloadable per-chat Feishu admission-rule overlay."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HotGroupRules:
    """Keep the last valid JSON payload while a rules file is being replaced."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fingerprint: tuple[int, int] | None = None
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            try:
                stat = self._path.stat()
            except FileNotFoundError:
                self._fingerprint, self._data = None, {}
                return {}
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            if fingerprint != self._fingerprint:
                try:
                    with self._path.open(encoding="utf-8") as handle:
                        loaded = json.load(handle)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    logger.warning("[Feishu] Could not reload %s; retaining last valid rules: %s", self._path, exc)
                    self._fingerprint = fingerprint
                else:
                    if isinstance(loaded, dict):
                        self._fingerprint, self._data = fingerprint, loaded
                    else:
                        logger.warning("[Feishu] Ignoring non-object group-rule file: %s", self._path)
                        self._fingerprint = fingerprint
            raw = self._data.get("group_rules")
            return {str(key): value for key, value in raw.items() if isinstance(value, dict)} if isinstance(raw, dict) else {}
