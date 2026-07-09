"""Cross-process active chat session leases.

The session database records persisted conversations.  This module records
currently open chat surfaces, including idle CLI/TUI sessions that have not
written a transcript row yet.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


_STATUS_METADATA_STRING_KEYS = {
    "activity_kind",
    "current_tool",
    "model_request_id_fingerprint",
    "model_request_model",
    "model_policy_recommended_action",
    "model_request_provider",
    "model_request_status",
    "model_request_status_message",
    "required_model",
}
_STATUS_METADATA_NUMBER_KEYS = {
    "last_activity_age_seconds",
    "last_activity_ts",
    "model_request_api_call_count",
    "model_request_estimated_context_tokens",
    "model_request_last_byte_at",
    "model_request_last_event_at",
    "model_request_queued_steer_count",
    "model_request_seconds_since_event",
    "model_request_started_at",
    "pending_steer_count",
    "queued_steer_count",
    "runtime_last_activity_ts",
    "seconds_since_activity",
}
_STATUS_METADATA_BOOL_KEYS = {
    "has_queued_steer",
    "model_policy_violation",
    "model_request_high_context",
    "model_request_steer_queued",
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


def active_session_acquire_error_message() -> str:
    return (
        "Hermes could not claim an active session slot. "
        "Try again shortly or run `hermes runtime active-sessions status`."
    )


def active_session_lock_busy_message(owner: Optional[dict[str, Any]]) -> str:
    owner_text = _format_lock_owner_summary(_safe_lock_owner_summary(owner or {}))
    return (
        f"Hermes active session registry is busy; owner {owner_text}. "
        "Try again shortly or run `hermes runtime active-sessions status`."
    )


def _state_dir() -> Path:
    return Path(get_hermes_home()) / "runtime"


def _state_path() -> Path:
    return _state_dir() / "active_sessions.json"


def _lock_path() -> Path:
    return _state_dir() / "active_sessions.lock"


_LOCK_OWNER_FILENAME = "owner.json"
_METADATA_UPDATE_LOCK_TIMEOUT_SECONDS = 0.25
_WINDOWS_LOCK_RELEASE_CLEANUP_TIMEOUT_SECONDS = 5.0
_WINDOWS_LOCK_RELEASE_RETRY_INTERVAL_SECONDS = 0.02
_WINDOWS_OWNERLESS_LOCK_RECLAIM_GRACE_SECONDS = 2.0
_LOCK_BUSY_ALERT_WINDOW_SECONDS = 120.0
_LOCK_BUSY_ALERT_THRESHOLD = 3
_LOCK_BUSY_ALERT_STATE: dict[str, Any] = {
    "window_start": 0.0,
    "count": 0,
    "alerted": False,
}
_METADATA_UPDATE_BUSY_LOG_WINDOW_SECONDS = 120.0
_METADATA_UPDATE_BUSY_LOG_STATE: dict[str, Any] = {
    "window_start": 0.0,
    "messages": {},
}


def _lock_owner_dir() -> Path:
    return _lock_path().with_name(f"{_lock_path().name}.d")


def _read_lock_owner_metadata() -> Optional[dict[str, Any]]:
    try:
        with open(_lock_owner_dir() / _LOCK_OWNER_FILENAME, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        logger.debug("Could not read active session lock owner metadata", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def active_session_lock_owner_summary() -> Optional[dict[str, Any]]:
    owner = _read_lock_owner_metadata()
    return _active_session_owner_summary(owner) if owner else None


class ActiveSessionLockBusyError(RuntimeError):
    def __init__(self, owner: Optional[dict[str, Any]] = None):
        super().__init__("active session file lock busy")
        self.owner = owner if isinstance(owner, dict) else {}


class _FileLock:
    def __init__(
        self,
        path: Path,
        *,
        owner_metadata: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.path = path
        self._fh = None
        self._lock_dir: Optional[Path] = None
        self._owner_metadata = dict(owner_metadata or {})
        self._timeout_seconds = timeout_seconds

    def _owner_path(self, lock_dir: Path) -> Path:
        return lock_dir / _LOCK_OWNER_FILENAME

    def _owner_payload(self) -> dict[str, Any]:
        now = time.time()
        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "process_start_time": _process_start_time(os.getpid()),
            "created_at": now,
            "updated_at": now,
            "cwd_fingerprint": _cwd_fingerprint(),
            "command_line_fingerprint": _command_line_fingerprint(),
        }
        for key in (
            "session_id",
            "session_key",
            "surface",
            "owner_kind",
            "cwd_fingerprint",
            "command_line_fingerprint",
        ):
            value = self._owner_metadata.get(key)
            safe_value = _safe_status_metadata_string(value)
            if safe_value is not None:
                payload[key] = safe_value
        return payload

    def _write_owner(self, lock_dir: Path) -> None:
        owner_path = self._owner_path(lock_dir)
        tmp = owner_path.with_name(f"{owner_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._owner_payload(), fh, sort_keys=True)
        os.replace(tmp, owner_path)

    def _read_owner(self, lock_dir: Path) -> Optional[dict[str, Any]]:
        try:
            with open(self._owner_path(lock_dir), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug(
                "Could not read active session lock owner metadata from %s",
                lock_dir,
                exc_info=True,
            )
            return None
        return data if isinstance(data, dict) else None

    def _try_reclaim_ownerless_lock_dir(self, lock_dir: Path) -> bool:
        """Remove an ownerless Windows lock dir only after a safety grace."""
        try:
            age = time.time() - lock_dir.stat().st_mtime
        except FileNotFoundError:
            return True
        except Exception:
            logger.debug(
                "Could not stat ownerless active session file lock at %s",
                lock_dir,
                exc_info=True,
            )
            return False
        if age < _WINDOWS_OWNERLESS_LOCK_RECLAIM_GRACE_SECONDS:
            return False
        try:
            lock_dir.rmdir()
            logger.warning(
                "Reclaimed ownerless active session file lock at %s age_seconds=%.3f",
                lock_dir,
                max(0.0, age),
            )
            return True
        except FileNotFoundError:
            return True
        except Exception:
            logger.debug(
                "Failed to reclaim ownerless active session file lock at %s",
                lock_dir,
                exc_info=True,
            )
            return False

    def _try_reclaim_stale_lock_dir(self, lock_dir: Path) -> bool:
        owner = self._read_owner(lock_dir)
        if not owner:
            return self._try_reclaim_ownerless_lock_dir(lock_dir)
        if _pid_alive(owner.get("pid"), owner.get("process_start_time")):
            return False
        try:
            self._owner_path(lock_dir).unlink(missing_ok=True)
            lock_dir.rmdir()
            logger.warning(
                "Reclaimed stale active session file lock owned by pid=%r session=%r",
                owner.get("pid"),
                owner.get("session_id"),
            )
            return True
        except FileNotFoundError:
            return True
        except Exception:
            logger.debug(
                "Failed to reclaim stale active session file lock at %s",
                lock_dir,
                exc_info=True,
            )
            return False

    def _cleanup_windows_lock_dir(self, lock_dir: Path) -> bool:
        deadline = time.monotonic() + _WINDOWS_LOCK_RELEASE_CLEANUP_TIMEOUT_SECONDS
        while True:
            try:
                self._owner_path(lock_dir).unlink(missing_ok=True)
                lock_dir.rmdir()
                return True
            except FileNotFoundError:
                return True
            except Exception:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Failed to remove active session file lock at %s after %.1fs",
                        lock_dir,
                        _WINDOWS_LOCK_RELEASE_CLEANUP_TIMEOUT_SECONDS,
                        exc_info=True,
                    )
                    return False
                time.sleep(_WINDOWS_LOCK_RELEASE_RETRY_INTERVAL_SECONDS)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            # Use an atomic directory create as the Windows mutex.  msvcrt
            # byte-range locks proved unreliable for the highly concurrent
            # active-session last-slot claim path on this machine.
            lock_dir = self.path.with_name(f"{self.path.name}.d")
            timeout = 30.0 if self._timeout_seconds is None else max(0.0, float(self._timeout_seconds))
            deadline = time.time() + timeout
            while True:
                try:
                    lock_dir.mkdir()
                    self._lock_dir = lock_dir
                    try:
                        self._write_owner(lock_dir)
                    except Exception:
                        try:
                            lock_dir.rmdir()
                        except Exception:
                            pass
                        self._lock_dir = None
                        raise
                    return self
                except FileExistsError:
                    if self._try_reclaim_stale_lock_dir(lock_dir):
                        continue
                    if time.time() > deadline:
                        raise ActiveSessionLockBusyError(self._read_owner(lock_dir))
                    time.sleep(0.02)

        self._fh = open(self.path, "a+b")
        try:
            import fcntl

            if self._timeout_seconds is None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            else:
                deadline = time.time() + max(0.0, float(self._timeout_seconds))
                while True:
                    try:
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() > deadline:
                            raise ActiveSessionLockBusyError({})
                        time.sleep(0.02)
        except Exception as exc:
            self._fh.close()
            self._fh = None
            if isinstance(exc, ActiveSessionLockBusyError):
                raise
            raise RuntimeError("active session file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._lock_dir is not None:
            lock_dir = self._lock_dir
            try:
                self._cleanup_windows_lock_dir(lock_dir)
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


def _safe_model_request_status_message(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text or len(text) > 240:
        return None
    if not text.startswith("active model request "):
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:()_-")
    if any(ch not in allowed for ch in text):
        return None
    return text


def _fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]


def _cwd_fingerprint() -> str:
    try:
        return _fingerprint_text(os.getcwd())
    except Exception:
        return _fingerprint_text("")


def _command_line_fingerprint() -> str:
    try:
        argv = getattr(sys, "argv", []) or []
        return _fingerprint_text("\0".join(str(part) for part in argv))
    except Exception:
        return _fingerprint_text("")


def _owner_kind_for_surface(surface: Any) -> str:
    prefix = str(surface or "").split(":", 1)[0].strip().lower()
    if prefix in {"cli", "tui", "gateway", "slash-worker", "kanban-worker", "repair"}:
        return prefix
    return "runtime"


def _safe_lock_owner_summary(owner: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    try:
        pid = int(owner.get("pid"))
        if pid > 0:
            safe["pid"] = pid
    except (TypeError, ValueError):
        pass
    for key in ("session_id", "session_key"):
        raw_value = owner.get(key)
        safe_value = _safe_status_metadata_string(raw_value)
        if safe_value is None:
            continue
        if safe_value.startswith("agent:"):
            safe[f"{key}_fingerprint"] = _fingerprint_text(safe_value)
        else:
            safe[key] = safe_value
    for key in ("surface", "owner_kind", "cwd_fingerprint", "command_line_fingerprint"):
        safe_value = _safe_status_metadata_string(owner.get(key))
        if safe_value is not None:
            safe[key] = safe_value
    return safe


def _format_lock_owner_summary(summary: dict[str, Any]) -> str:
    parts = []
    for key in (
        "pid",
        "session_id",
        "session_id_fingerprint",
        "session_key",
        "session_key_fingerprint",
        "surface",
        "owner_kind",
        "cwd_fingerprint",
        "command_line_fingerprint",
    ):
        if key in summary:
            parts.append(f"{key}={summary[key]}")
    return " ".join(parts) if parts else "unknown"


def _active_session_owner_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return _safe_lock_owner_summary(entry)


def _entry_with_runtime_status(entry: dict[str, Any]) -> dict[str, Any]:
    item = dict(entry)
    item["owner_summary"] = _active_session_owner_summary(item)
    item["runtime_status"] = (
        "live" if _pid_alive(item.get("pid"), item.get("process_start_time")) else "stale"
    )
    return item


def _record_lock_busy_alert(owner: dict[str, Any]) -> None:
    now = time.time()
    state = _LOCK_BUSY_ALERT_STATE
    window_start = float(state.get("window_start") or 0.0)
    if window_start <= 0.0 or now - window_start > _LOCK_BUSY_ALERT_WINDOW_SECONDS:
        state["window_start"] = now
        state["count"] = 0
        state["alerted"] = False
    state["count"] = int(state.get("count") or 0) + 1
    if state["count"] < _LOCK_BUSY_ALERT_THRESHOLD or state.get("alerted"):
        return
    state["alerted"] = True
    logger.warning(
        "Repeated active session registry lock timeouts: count=%d "
        "window_seconds=%d owner=%s",
        state["count"],
        int(_LOCK_BUSY_ALERT_WINDOW_SECONDS),
        _format_lock_owner_summary(_safe_lock_owner_summary(owner)),
    )


def _should_log_metadata_update_busy(owner: dict[str, Any]) -> bool:
    now = time.time()
    state = _METADATA_UPDATE_BUSY_LOG_STATE
    window_start = float(state.get("window_start") or 0.0)
    if window_start <= 0.0 or now - window_start > _METADATA_UPDATE_BUSY_LOG_WINDOW_SECONDS:
        state["window_start"] = now
        state["messages"] = {}
    messages = state.setdefault("messages", {})
    if not isinstance(messages, dict):
        messages = {}
        state["messages"] = messages
    key = _format_lock_owner_summary(_safe_lock_owner_summary(owner))
    if key in messages:
        return False
    messages[key] = now
    return True


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
        if key == "model_request_status_message":
            safe_value = _safe_model_request_status_message(value)
            if safe_value is not None:
                safe[key] = safe_value
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
    with _FileLock(_lock_path(), owner_metadata={"owner_kind": "prune"}):
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
    session_key = str(session_id)
    surface_name = str(surface)
    entry = {
        "lease_id": lease_id,
        "session_id": session_key,
        "session_key": session_key,
        "surface": surface_name,
        "owner_kind": _owner_kind_for_surface(surface_name),
        "pid": os.getpid(),
        "process_start_time": _process_start_time(os.getpid()),
        "created_at": now,
        "started_at": now,
        "updated_at": now,
        "cwd_fingerprint": _cwd_fingerprint(),
        "command_line_fingerprint": _command_line_fingerprint(),
    }
    if metadata:
        entry["metadata"] = {
            str(k): v for k, v in metadata.items() if isinstance(k, str)
        }

    state_path = _state_path()
    try:
        with _FileLock(
            _lock_path(),
            owner_metadata={
                "session_id": session_key,
                "session_key": session_key,
                "surface": surface_name,
                "owner_kind": "try_acquire",
            },
        ):
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
    except ActiveSessionLockBusyError as exc:
        _record_lock_busy_alert(exc.owner)
        return None, active_session_lock_busy_message(exc.owner)

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
    try:
        with _FileLock(
            _lock_path(),
            owner_metadata={
                "session_id": session_key,
                "owner_kind": "metadata_update",
            },
            timeout_seconds=_METADATA_UPDATE_LOCK_TIMEOUT_SECONDS,
        ):
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
    except ActiveSessionLockBusyError as exc:
        _record_lock_busy_alert(exc.owner)
        message = active_session_lock_busy_message(exc.owner)
        if _should_log_metadata_update_busy(exc.owner):
            logger.warning(
                "Skipped active session metadata update because registry lock is busy: %s",
                message,
            )
        else:
            logger.debug(
                "Skipped active session metadata update because registry lock is busy: %s",
                message,
            )
        return 0
    return updated


def release_active_session(lease: ActiveSessionLease) -> None:
    state_path = _state_path()
    try:
        with _FileLock(
            _lock_path(),
            owner_metadata={
                "session_id": lease.session_id,
                "surface": lease.surface,
                "owner_kind": "release",
            },
        ):
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


def release_active_sessions_for_current_process(
    *,
    session_id: str,
    surface: Optional[str] = None,
) -> int:
    """Release live leases for ``session_id`` owned by the current process.

    This is for terminal worker protocols that only have environment context
    rather than an in-memory :class:`ActiveSessionLease`. It deliberately
    refuses to touch leases owned by another PID/start-time pair.
    """
    session_key = str(session_id or "").strip()
    if not session_key:
        return 0
    current_pid = os.getpid()
    current_start = _process_start_time(current_pid)

    def _owned_by_current_process(entry: dict[str, Any]) -> bool:
        if str(entry.get("session_id") or "") != session_key:
            return False
        try:
            if int(entry.get("pid")) != current_pid:
                return False
        except (TypeError, ValueError):
            return False
        expected_start = _optional_float(entry.get("process_start_time"))
        if expected_start is None or current_start is None:
            return True
        return abs(current_start - expected_start) < 1.0

    state_path = _state_path()
    with _FileLock(
        _lock_path(),
        owner_metadata={
            "session_id": session_key,
            "surface": surface or "",
            "owner_kind": "release_current_process",
        },
    ):
        entries = _prune_dead(_read_entries(state_path))
        kept = [entry for entry in entries if not _owned_by_current_process(entry)]
        released = len(entries) - len(kept)
        if released:
            _write_entries(state_path, kept)
        return released


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
    with _FileLock(
        _lock_path(),
        owner_metadata={
            "session_id": lease.session_id,
            "surface": lease.surface,
            "owner_kind": "transfer",
        },
    ):
        entries = _prune_dead(_read_entries(state_path))
        updated = False
        for entry in entries:
            if str(entry.get("lease_id") or "") != lease.lease_id:
                continue
            entry["previous_owner_summary"] = _active_session_owner_summary(entry)
            entry["session_id"] = new_session_id
            entry["session_key"] = new_session_id
            entry["owner_kind"] = _owner_kind_for_surface(lease.surface)
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
    with _FileLock(_lock_path(), owner_metadata={"owner_kind": "snapshot"}):
        entries = _prune_dead(_read_entries(state_path))
        _write_entries(state_path, entries)
        snapshot: list[dict[str, Any]] = []
        for entry in entries:
            item = dict(entry)
            item["owner_summary"] = _active_session_owner_summary(item)
            snapshot.append(item)
        return snapshot


def _registry_status_from_entries(
    entries: list[dict[str, Any]],
    *,
    lock_status: str = "ok",
    lock_owner_summary: Optional[dict[str, Any]] = None,
    read_mode: str = "locked",
) -> dict[str, Any]:
    live = []
    stale = []
    for entry in entries:
        item = _entry_with_runtime_status(entry)
        if item["runtime_status"] == "live":
            live.append(item)
        else:
            stale.append(item)
    report: dict[str, Any] = {
        "checked": len(entries),
        "live": len(live),
        "stale": len(stale),
        "entries": live + stale,
        "lock_status": lock_status,
        "read_mode": read_mode,
    }
    if lock_owner_summary:
        report["lock_owner_summary"] = lock_owner_summary
    return report


def active_session_registry_status(*, no_lock: bool = False) -> dict[str, Any]:
    """Return live/stale active-session registry diagnostics without repair."""
    state_path = _state_path()
    if no_lock:
        owner = active_session_lock_owner_summary()
        return _registry_status_from_entries(
            _read_entries(state_path),
            lock_status="degraded" if owner else "lock_free",
            lock_owner_summary=owner,
            read_mode="lock_free",
        )
    try:
        with _FileLock(_lock_path(), owner_metadata={"owner_kind": "status"}):
            return _registry_status_from_entries(_read_entries(state_path))
    except ActiveSessionLockBusyError as exc:
        _record_lock_busy_alert(exc.owner)
        return _registry_status_from_entries(
            _read_entries(state_path),
            lock_status="degraded",
            lock_owner_summary=_active_session_owner_summary(exc.owner),
            read_mode="lock_free",
        )


def repair_stale_active_session_leases(
    *,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Remove only active-session leases with concrete dead-owner evidence."""
    target = str(session_id or "").strip() or None
    state_path = _state_path()
    checked = 0
    stale = 0
    repaired = 0
    skipped_live = 0
    repaired_session_ids: list[str] = []
    skipped_live_session_ids: list[str] = []
    with _FileLock(_lock_path(), owner_metadata={"owner_kind": "repair"}):
        entries = _read_entries(state_path)
        kept: list[dict[str, Any]] = []
        for entry in entries:
            entry_session_id = str(entry.get("session_id") or "")
            matches_target = target is None or entry_session_id == target
            if not matches_target:
                kept.append(entry)
                continue
            checked += 1
            if _pid_alive(entry.get("pid"), entry.get("process_start_time")):
                skipped_live += 1
                if entry_session_id:
                    skipped_live_session_ids.append(entry_session_id)
                kept.append(entry)
                continue
            stale += 1
            repaired += 1
            if entry_session_id:
                repaired_session_ids.append(entry_session_id)
        if repaired:
            _write_entries(state_path, kept)
    return {
        "checked": checked,
        "stale": stale,
        "repaired": repaired,
        "skipped_live": skipped_live,
        "repaired_session_ids": repaired_session_ids,
        "skipped_live_session_ids": skipped_live_session_ids,
    }
