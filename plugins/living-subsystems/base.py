"""Subsystem ABC: JSON persistence under the active profile's HERMES_HOME with a uniform result shape."""

from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

# One lock per data file (not per instance): `Governance()` is constructed ad hoc from cron, tools
# and hooks, so instance locks would not serialize their read-modify-write cycles.
_FILE_LOCKS: Dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(str(path), threading.RLock())


def result(ok: bool, message: str, **details: Any) -> Dict[str, Any]:
    return {"ok": ok, "message": message, "details": details}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now() -> float:
    return time.time()


class Subsystem(ABC):
    """Base for subsystems. ``home`` defaults to the profile-aware HERMES_HOME resolved per call."""

    def __init__(self, name: Optional[str] = None, home: Optional[Path] = None):
        self.name = name or type(self).__name__
        self._home = Path(home) if home is not None else None

    @property
    def home(self) -> Path:
        return self._home if self._home is not None else get_hermes_home()

    def _path(self, filename: str) -> Path:
        return self.home / filename

    def lock(self, filename: str) -> threading.RLock:
        return _lock_for(self._path(filename))

    def load(self, filename: str, default: Any = None) -> Any:
        """Read JSON; a missing or corrupt file yields *default* (a fresh ``{}`` when omitted)."""
        try:
            return json.loads(self._path(filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {} if default is None else default

    def save(self, filename: str, data: Any) -> None:
        path = self._path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data)

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Periodic entry point. Returns ``{"ok", "message", "details"}``."""

    def status(self) -> Dict[str, Any]:
        return result(True, f"{self.name} ready")
