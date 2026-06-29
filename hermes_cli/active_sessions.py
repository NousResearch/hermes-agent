"""Cross-process active chat session leases.

The session database records persisted conversations.  This module records
currently open chat surfaces, including idle CLI/TUI sessions that have not
written a transcript row yet.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


_STATUS_METADATA_STRING_KEYS = {"activity_kind", "current_tool"}
_STATUS_METADATA_NUMBER_KEYS = {
    "last_activity_age_seconds",
    "last_activity_ts",
    "pending_steer_count",
    "queued_steer_count",
    "runtime_last_activity_ts",
    "seconds_since_activity",
}
_STATUS_METADATA_BOOL_KEYS = {
    "has_queued_steer",
    "pending_steer_queued",
    "queued_steer",
}
_STATUS_METADATA_REMOVABLE_KEYS = (
    _STATUS_METADATA_STRING_KEYS
    | _STATUS_METADATA_NUMBER_KEYS
    | _STATUS_METADATA_BOOL_KEYS
)


def coerce_max_concurrent_sessions(value: Any, key: str = "max_concurrent_sessions") -> Optional[int]:
    """Return a positive integer cap, or None when disabled/invalid."""
    if value is None:
        return None
    if isinstance(value, dict) and not value:
        return None
    if isinstance(value, bool):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(value)
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip(), 10)
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)",
            key,
            value,
        )
        return None
    if parsed <= 0:
        return None
    return parsed


def resolve_max_concurrent_sessions(config: Any) -> Optional[int]:
    """Resolve top-level max_concurrent_sessions with gateway.* fallback."""
    raw: Any = None
    key = "max_concurrent_sessions"
    if isinstance(config, dict):
        if "max_concurrent_sessions" in config:
            raw = config.get("max_concurrent_sessions")
        else:
            gateway_cfg = config.get("gateway")
            if isinstance(gateway_cfg, dict):
                raw = gateway_cfg.get("max_concurrent_sessions")
                key = "gateway.max_concurrent_sessions"
    else:
        raw = getattr(config, "max_concurrent_sessions", None)
    return coerce_max_concurrent_sessions(raw, key=key)


def active_session_limit_message(active_count: int, max_sessions: int) -> str:
    return (
        f"Hermes is at the active session limit ({active_count}/{max_sessions}). "
        "Try again when another session finishes."
    )


def _state_dir() -> Path:
    return Path(get_hermes_home()) / "runtime"


def _state_path() -> Path:
    return _state_dir() / "active_sessions.json"


def _lock_path() -> Path:
    return _state_dir() / "active_sessions.lock"


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._lock_dir: Optional[Path] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            # Use an atomic directory create as the Windows mutex.  msvcrt
            # byte-range locks proved unreliable for the highly concurrent
            # active-session last-slot claim path on this machine.
            lock_dir = self.path.with_name(f"{self.path.name}.d")
            deadline = time.time() + 30
            while True:
                try:
                    lock_dir.mkdir()
                    self._lock_dir = lock_dir
                    return self
                except FileExistsError:
                    if time.time() > deadline:
                        raise RuntimeError("active session file lock timed out")
                    time.sleep(0.02)

        self._fh = open(self.path, "a+b")
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("active session file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lock_dir is not None:
            try:
                self._lock_dir.rmdir()
            except Exception:
                pass
            finally:
                self._lock_dir = None
        if self._fh is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        finally:
            self._fh = None


def _read_entries(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception:
        logger.warning("Ignoring corrupt active session registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, sort_keys=True)
    os.replace(tmp, path)


def _safe_status_metadata_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 80:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if any(ch not in allowed for ch in value):
        return None
    return value


def _safe_status_metadata_update(metadata: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    safe: dict[str, Any] = {}
    remove: set[str] = set()
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if value is None:
            if key in _STATUS_METADATA_REMOVABLE_KEYS:
                remove.add(key)
            continue
        if key in _STATUS_METADATA_STRING_KEYS:
            safe_value = _safe_status_metadata_string(value)
            if safe_value is not None:
                safe[key] = safe_value
            continue
        if key in _STATUS_METADATA_NUMBER_KEYS:
            parsed = _metadata_positive_number(value)
            if parsed is not None:
                safe[key] = int(parsed) if parsed.is_integer() else parsed
            continue
        if key in _STATUS_METADATA_BOOL_KEYS:
            if isinstance(value, bool):
                safe[key] = value
            elif isinstance(value, str):
                safe[key] = _truthy_metadata(value)
            elif isinstance(value, (int, float)):
                safe[key] = value > 0
    return safe, remove


def _process_start_time(pid: int) -> Optional[float]:
    # Pair pid with process create_time when psutil can read it, so a recycled
    # pid does not keep a stale lease alive indefinitely.
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: Any, process_start_time: Any = None) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        exists = bool(_pid_exists(pid_int))
    except Exception:
        return False
    if not exists:
        return False
    expected_start = _optional_float(process_start_time)
    if expected_start is None:
        return True
    current_start = _process_start_time(pid_int)
    if current_start is None:
        return True
    # Windows process launchers can round reported create_time differently
    # across processes. Keep the guard tight enough for PID reuse, but avoid
    # pruning a live lease because of millisecond-level timestamp jitter.
    return abs(current_start - expected_start) < 1.0


def _prune_dead(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live, _dead = _split_live_dead(entries)
    return live


def _split_live_dead(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    live: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    for entry in entries:
        if _pid_alive(entry.get("pid"), entry.get("process_start_time")):
            live.append(entry)
        else:
            dead.append(entry)
    return live, dead


def _truthy_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _metadata_positive_number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _entry_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _entry_has_queued_steer_evidence(entry: dict[str, Any]) -> bool:
    metadata = _entry_metadata(entry)
    for key in (
        "pending_steer_queued",
        "queued_steer",
        "has_queued_steer",
    ):
        if _truthy_metadata(metadata.get(key)):
            return True
    for key in ("pending_steer_count", "queued_steer_count"):
        count = _metadata_positive_number(metadata.get(key))
        if count and count > 0:
            return True
    return False


def _entry_has_fresh_runtime_activity(
    entry: dict[str, Any],
    *,
    fresh_activity_grace_seconds: float,
) -> bool:
    metadata = _entry_metadata(entry)
    for key in ("last_activity_age_seconds", "seconds_since_activity"):
        age = _metadata_positive_number(metadata.get(key))
        if age is not None and age <= fresh_activity_grace_seconds:
            return True
    now = time.time()
    for key in ("last_activity_ts", "runtime_last_activity_ts"):
        ts = _metadata_positive_number(metadata.get(key))
        if ts is not None and now - ts <= fresh_activity_grace_seconds:
            return True
    return False


def _session_ids(entries: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for entry in entries:
        session_id = str(entry.get("session_id") or "").strip()
        if session_id:
            ids.add(session_id)
    return ids


def prune_dead_active_session_leases(
    *,
    session_db: Any = None,
    end_reason: str = "stale_active_session",
    fresh_activity_grace_seconds: float = 30.0,
) -> int:
    """Prune dead active-session leases and optionally close DB sessions.

    This is the lifecycle bridge for hard-exit/crash cases: the registry has
    the process identity, while ``state.db`` has the persisted transcript.  When
    the process is gone, the session should not remain indefinitely open.
    """
    state_path = _state_path()
    dead: list[dict[str, Any]]
    with _FileLock(_lock_path()):
        live, dead = _split_live_dead(_read_entries(state_path))
        _write_entries(state_path, live)

    finalized = 0
    if session_db is not None:
        end_session = getattr(session_db, "end_session", None)
        if callable(end_session):
            live_session_ids = _session_ids(live)
            for entry in dead:
                session_id = str(entry.get("session_id") or "").strip()
                if not session_id:
                    continue
                if session_id in live_session_ids:
                    continue
                if _entry_has_queued_steer_evidence(entry):
                    continue
                if _entry_has_fresh_runtime_activity(
                    entry,
                    fresh_activity_grace_seconds=fresh_activity_grace_seconds,
                ):
                    continue
                try:
                    end_session(session_id, end_reason)
                    finalized += 1
                except Exception:
                    logger.debug(
                        "Failed to finalize stale active session %s",
                        session_id,
                        exc_info=True,
                    )
    return finalized


@dataclass
class ActiveSessionLease:
    lease_id: str
    session_id: str
    surface: str
    enabled: bool = True
    released: bool = False

    def release(self) -> None:
        if self.released or not self.enabled:
            return
        release_active_session(self)


def try_acquire_active_session(
    *,
    session_id: str,
    surface: str,
    config: Any,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Optional[ActiveSessionLease], Optional[str]]:
    """Acquire an active-session slot.

    Returns ``(lease, None)`` on success.  When the cap is disabled, the lease is
    still tracked so hard exits can be pruned and reflected into state.db.
    """
    max_sessions = resolve_max_concurrent_sessions(config)
    lease_id = uuid.uuid4().hex
    now = time.time()
    entry = {
        "lease_id": lease_id,
        "session_id": str(session_id),
        "surface": str(surface),
        "pid": os.getpid(),
        "process_start_time": _process_start_time(os.getpid()),
        "started_at": now,
        "updated_at": now,
    }
    if metadata:
        entry["metadata"] = {
            str(k): v for k, v in metadata.items() if isinstance(k, str)
        }

    state_path = _state_path()
    with _FileLock(_lock_path()):
        raw_entries = _read_entries(state_path)
        entries = _prune_dead(raw_entries)
        pruned = len(raw_entries) - len(entries)
        if pruned:
            logger.info("Pruned %d stale active session lease(s)", pruned)
        active_count = len(entries)
        if max_sessions is not None and active_count >= max_sessions:
            _write_entries(state_path, entries)
            logger.info(
                "Active session limit reached: active=%d max=%d surface=%s",
                active_count,
                max_sessions,
                surface,
            )
            return None, active_session_limit_message(active_count, max_sessions)
        entries.append(entry)
        _write_entries(state_path, entries)

    return ActiveSessionLease(
        lease_id=lease_id,
        session_id=str(session_id),
        surface=str(surface),
    ), None


def update_active_session_metadata(
    *,
    session_id: str,
    metadata: dict[str, Any],
) -> int:
    """Merge value-free runtime status metadata into live leases for a session."""

    session_key = str(session_id or "").strip()
    if not session_key or not isinstance(metadata, dict):
        return 0
    safe_metadata, remove_keys = _safe_status_metadata_update(metadata)
    if not safe_metadata and not remove_keys:
        return 0

    state_path = _state_path()
    updated = 0
    current_pid = os.getpid()
    with _FileLock(_lock_path()):
        entries = _prune_dead(_read_entries(state_path))
        now = time.time()
        for entry in entries:
            if str(entry.get("session_id") or "") != session_key:
                continue
            try:
                if int(entry.get("pid")) != current_pid:
                    continue
            except (TypeError, ValueError):
                continue
            current_metadata = entry.get("metadata")
            merged = dict(current_metadata) if isinstance(current_metadata, dict) else {}
            for key in remove_keys:
                merged.pop(key, None)
            merged.update(safe_metadata)
            entry["metadata"] = merged
            entry["updated_at"] = now
            updated += 1
        if updated:
            _write_entries(state_path, entries)
    return updated


def release_active_session(lease: ActiveSessionLease) -> None:
    state_path = _state_path()
    try:
        with _FileLock(_lock_path()):
            entries = _prune_dead(_read_entries(state_path))
            kept = [
                entry
                for entry in entries
                if str(entry.get("lease_id") or "") != lease.lease_id
            ]
            if len(kept) != len(entries):
                _write_entries(state_path, kept)
    finally:
        lease.released = True


def transfer_active_session(
    lease: ActiveSessionLease,
    *,
    session_id: str,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Move an existing lease to a new session id without dropping the slot."""
    new_session_id = str(session_id or "")
    if not new_session_id:
        return False
    if lease.released:
        return False
    if not lease.enabled:
        lease.session_id = new_session_id
        return True

    state_path = _state_path()
    with _FileLock(_lock_path()):
        entries = _prune_dead(_read_entries(state_path))
        updated = False
        for entry in entries:
            if str(entry.get("lease_id") or "") != lease.lease_id:
                continue
            entry["session_id"] = new_session_id
            entry["updated_at"] = time.time()
            if metadata:
                entry["metadata"] = {
                    str(k): v for k, v in metadata.items() if isinstance(k, str)
                }
            updated = True
            break
        if updated:
            _write_entries(state_path, entries)
            lease.session_id = new_session_id
        return updated


def active_session_registry_snapshot() -> list[dict[str, Any]]:
    """Return the pruned active-session registry for diagnostics/tests."""
    state_path = _state_path()
    with _FileLock(_lock_path()):
        entries = _prune_dead(_read_entries(state_path))
        _write_entries(state_path, entries)
        return entries
