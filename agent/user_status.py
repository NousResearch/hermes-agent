"""Cross-bot user-status storage layer.

Persists a small, shared snapshot of user state (device mode, AFK status,
focus project, quiet hours, location) to a single JSON file under
``get_hermes_home()/state/user_status.json`` so multiple gateway profiles
(Telegram, Discord, Slack, etc.) can read/write coherently.

Design notes (issue #21122 / kanban t_315c0bfc):

* **Single file, atomic writes.** Up to ~7 profiles may touch this file
  concurrently. Every write does a read-modify-write under a per-process
  lock and a cross-process file lock (``fcntl.flock`` on POSIX, best-effort
  no-op elsewhere), then ``os.replace()`` to publish atomically. Other
  fields are preserved on partial updates.
* **Per-field timestamps.** ``per_field_updated_at`` records ISO-8601 UTC
  for every field that was last written, plus an ``updated_by`` tag
  identifying the last writer (e.g. ``"telegram"``). ``is_stale()`` lets
  callers decide whether a field is still trustworthy.
* **No third-party deps.** Pure stdlib (json, dataclasses, threading, fcntl).

This module owns *only* storage. Tool registration, prompt injection, and
gateway plumbing live in sibling modules (other workers).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:  # POSIX file locking; best-effort fallback for Windows.
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_DIRNAME = "state"
_STATE_FILENAME = "user_status.json"

# Allowed top-level user-state fields (excluding metadata).
_USER_FIELDS = (
    "device_mode",
    "afk_status",
    "focus_project",
    "quiet_hours_until",
    "location",
)

# Process-local lock — guards the read-modify-write critical section
# against threads in *this* process. Cross-process safety is handled by
# the OS file lock on the JSON file itself.
_LOCAL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class UserStatus:
    """Snapshot of cross-bot user state.

    All user fields default to ``None`` (unknown). ``per_field_updated_at``
    maps field name → ISO-8601 UTC timestamp of last write. ``updated_by``
    is the tag of the most recent writer (e.g. a profile/platform name).
    """

    device_mode: Optional[str] = None
    afk_status: Optional[str] = None
    focus_project: Optional[str] = None
    quiet_hours_until: Optional[str] = None
    location: Optional[str] = None
    per_field_updated_at: dict = field(default_factory=dict)
    updated_by: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserStatus":
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        # Preserve any unknown keys onto the instance dict so future
        # schema additions written by a newer worker survive a load/save
        # by an older one. (We still drop them silently — keep simple.)
        per = kwargs.get("per_field_updated_at")
        if not isinstance(per, dict):
            kwargs["per_field_updated_at"] = {}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    return get_hermes_home() / _STATE_DIRNAME


def _state_path() -> Path:
    return _state_dir() / _STATE_FILENAME


def ensure_state_dir() -> Path:
    """Create the state directory (idempotent) and return its path."""
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# IO primitives
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_raw(path: Path) -> dict:
    """Read JSON file → dict. Returns empty dict if missing/corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Corrupt / unreadable → treat as empty rather than crash. The
        # next save will overwrite it atomically.
        return {}


def _atomic_write(path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically via tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load() -> UserStatus:
    """Load the current user status snapshot.

    Returns a default-empty :class:`UserStatus` if the file does not yet
    exist or cannot be parsed.
    """
    return UserStatus.from_dict(_read_raw(_state_path()))


def save_field(field_name: str, value: Any, writer: str) -> UserStatus:
    """Atomically update a single field, preserving all others.

    Args:
        field_name: One of the known user fields (see ``_USER_FIELDS``).
        value: New value (any JSON-serializable type, typically str/None).
        writer: Tag of the writer (e.g. ``"telegram"``, ``"discord"``);
            stored as ``updated_by`` and used for audit/debug.

    Returns:
        The :class:`UserStatus` post-write.

    Raises:
        ValueError: If ``field_name`` is not a known user field.
    """
    if field_name not in _USER_FIELDS:
        raise ValueError(
            f"unknown user_status field {field_name!r}; "
            f"expected one of {_USER_FIELDS}"
        )

    ensure_state_dir()
    path = _state_path()

    with _LOCAL_LOCK:
        # Cross-process lock via a sidecar lock file so the JSON itself
        # is never held open while we publish via os.replace().
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_fh = open(lock_path, "a+")
        try:
            if _HAS_FCNTL:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                raw = _read_raw(path)
                status = UserStatus.from_dict(raw)
                setattr(status, field_name, value)
                status.per_field_updated_at = dict(status.per_field_updated_at)
                status.per_field_updated_at[field_name] = _now_iso()
                status.updated_by = writer
                _atomic_write(path, status.to_dict())
                return status
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()


def is_stale(field_name: str, threshold_seconds: float,
             status: Optional[UserStatus] = None) -> bool:
    """Return ``True`` if ``field_name`` was last updated more than
    ``threshold_seconds`` ago (or has no recorded timestamp).

    A missing/never-set field is considered stale.
    """
    if status is None:
        status = load()
    ts = status.per_field_updated_at.get(field_name) if status.per_field_updated_at else None
    if not ts:
        return True
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - when).total_seconds()
    return age > threshold_seconds


__all__ = [
    "UserStatus",
    "ensure_state_dir",
    "load",
    "save_field",
    "is_stale",
]
