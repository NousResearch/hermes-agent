"""Cross-process active chat session leases.

The session database records persisted conversations.  This module records
currently open chat surfaces, including idle CLI/TUI sessions that have not
written a transcript row yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class ActiveSessionRegistryError(RuntimeError):
    """The liveness registry could not prove a safe ownership decision."""


def coerce_max_concurrent_sessions(value: Any, key: str = "max_concurrent_sessions") -> Optional[int]:
    """Return a positive integer cap, or None when disabled/invalid."""
    if value is None:
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


def format_age(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" if not minutes else f"{hours}h{minutes}m"


def summarize_holders(entries: list[dict[str, Any]]) -> str:
    """Compact "who is holding the slots" phrase, e.g. ``desktop x4, cli``."""
    if not entries:
        return ""
    counts: dict[str, int] = {}
    for entry in entries:
        surface = str(entry.get("surface") or "unknown")
        counts[surface] = counts.get(surface, 0) + 1
    held = ", ".join(
        f"{surface} x{n}" if n > 1 else surface
        for surface, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    started = [t for t in (_optional_float(e.get("started_at")) for e in entries) if t]
    if started:
        held += f", oldest {format_age(time.time() - min(started))} ago"
    return held


def active_session_limit_message(
    active_count: int,
    max_sessions: int,
    entries: Optional[list[dict[str, Any]]] = None,
) -> str:
    # Name the holders: the slots are shared across CLI, desktop/TUI and the
    # messaging gateway, so the surface that gets rejected is usually NOT the
    # one squatting on them (idle desktop chats starving a Discord bot, say).
    # Without this the message is unactionable and the only way to find out is
    # reading runtime/active_sessions.json by hand.
    held = summarize_holders(entries or [])
    detail = f" Held by: {held}." if held else ""
    return (
        f"Hermes is at the active session limit ({active_count}/{max_sessions})."
        f"{detail} Try again when another session finishes."
    )


def _registry_home(registry_home: str | Path | None = None) -> Path:
    return Path(registry_home) if registry_home is not None else Path(get_hermes_home())


def _state_dir(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime"


def _state_path(registry_home: str | Path | None = None) -> Path:
    return _state_dir(registry_home) / "active_sessions.json"


def _lock_path(registry_home: str | Path | None = None) -> Path:
    return _state_dir(registry_home) / "active_sessions.lock"


def _lease_paths(
    lease: Optional["ActiveSessionLease"] = None,
    registry_home: str | Path | None = None,
) -> tuple[Path, Path]:
    if lease is not None and lease.state_path is not None and lease.lock_path is not None:
        return lease.state_path, lease.lock_path
    home = _registry_home(registry_home)
    return _state_path(home), _lock_path(home)


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("active session file lock unavailable") from exc
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("active session file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            self._fh.close()
        finally:
            self._fh = None


def _read_entries(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception as exc:
        if strict:
            raise ActiveSessionRegistryError(
                f"active session registry unreadable: {path}"
            ) from exc
        logger.warning("Ignoring corrupt active session registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        if strict:
            raise ActiveSessionRegistryError(
                f"active session registry has invalid shape: {path}"
            )
        return []
    valid = [entry for entry in entries if isinstance(entry, dict)]
    if not strict:
        return valid
    if len(valid) != len(entries):
        raise ActiveSessionRegistryError(
            f"active session registry contains invalid entries: {path}"
        )
    seen_leases: set[str] = set()
    for entry in valid:
        lease_id = entry.get("lease_id")
        session_id = entry.get("session_id")
        pid = entry.get("pid")
        if not isinstance(lease_id, str) or not lease_id.strip():
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid lease id: {path}"
            )
        if lease_id in seen_leases:
            raise ActiveSessionRegistryError(
                f"active session registry contains a duplicate lease id: {path}"
            )
        seen_leases.add(lease_id)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid session id: {path}"
            )
        if isinstance(pid, bool) or not isinstance(pid, (int, str)):
            pid_int = 0
        else:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = 0
        if pid_int <= 0:
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid pid: {path}"
            )
        surface = entry.get("surface")
        if surface is not None and not isinstance(surface, str):
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid surface: {path}"
            )
        tracked = entry.get("track_liveness")
        if tracked is not None and not isinstance(tracked, bool):
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid liveness marker: {path}"
            )
        metadata = entry.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ActiveSessionRegistryError(
                f"active session registry contains invalid metadata: {path}"
            )
        process_start = entry.get("process_start_time")
        parsed_process_start = _optional_float(process_start)
        if process_start not in (None, "") and (
            parsed_process_start is None or not math.isfinite(parsed_process_start)
        ):
            raise ActiveSessionRegistryError(
                f"active session registry contains an invalid process start time: {path}"
            )
    return valid


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": entries}, fh, sort_keys=True)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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


def _pid_liveness(pid: Any, process_start_time: Any = None) -> Optional[bool]:
    """Return True/False for live/dead, or None when liveness is unknowable."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    if pid_int <= 0:
        return None
    try:
        from gateway.status import _pid_exists

        exists = bool(_pid_exists(pid_int))
    except Exception:
        return None
    if not exists:
        return False
    expected_start = _optional_float(process_start_time)
    if expected_start is None:
        return True
    current_start = _process_start_time(pid_int)
    if current_start is None:
        return None
    return abs(current_start - expected_start) < 0.001


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
    return abs(current_start - expected_start) < 0.001


def _prune_dead(
    entries: list[dict[str, Any]], *, strict: bool = False
) -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    for entry in entries:
        tracked = bool(entry.get("track_liveness"))
        if strict or tracked:
            state = _pid_liveness(entry.get("pid"), entry.get("process_start_time"))
            if state is None:
                raise ActiveSessionRegistryError(
                    "active session owner liveness is unknown"
                )
            if state:
                live.append(entry)
            continue
        if _pid_alive(entry.get("pid"), entry.get("process_start_time")):
            live.append(entry)
    return live


@dataclass
class ActiveSessionLease:
    lease_id: str
    session_id: str
    surface: str
    enabled: bool = True
    released: bool = False
    # Registry paths pinned at acquisition time. A lease acquired under the
    # root ``HERMES_HOME`` must release against the same registry even when
    # ``release()`` runs inside a profile home override (native multiplex
    # routes turns under ``_profile_runtime_scope``), otherwise the root
    # entry survives until process exit and the session cap fills with
    # phantom leases (#85431).
    state_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    track_liveness: bool = False

    def release(self) -> None:
        if self.released or not self.enabled:
            return
        release_active_session(self)


def _lease_entry(
    *,
    lease_id: str,
    session_id: str,
    surface: str,
    metadata: Optional[dict[str, Any]] = None,
    track_liveness: bool = False,
) -> dict[str, Any]:
    now = time.time()
    entry: dict[str, Any] = {
        "lease_id": lease_id,
        "session_id": str(session_id),
        "surface": str(surface),
        "pid": os.getpid(),
        "process_start_time": _process_start_time(os.getpid()),
        "started_at": now,
        "updated_at": now,
    }
    if track_liveness:
        entry["track_liveness"] = True
    if metadata:
        entry["metadata"] = {
            str(k): v for k, v in metadata.items() if isinstance(k, str)
        }
    return entry


def try_acquire_active_session(
    *,
    session_id: str,
    surface: str,
    config: Any,
    metadata: Optional[dict[str, Any]] = None,
    registry_home: str | Path | None = None,
    track_liveness: bool = False,
) -> tuple[Optional[ActiveSessionLease], Optional[str]]:
    """Acquire an active-session slot.

    Returns ``(lease, None)`` on success.  When the cap is disabled, the lease is
    a no-op object so callers can unconditionally call ``release()`` unless
    ``track_liveness`` is true.  Liveness tracking keeps a real lease without
    imposing a concurrency cap; ``registry_home`` lets profile-scoped backends
    share the owning profile's registry even when launched from another home.
    """
    max_sessions = resolve_max_concurrent_sessions(config)
    lease_id = uuid.uuid4().hex
    if max_sessions is None and not track_liveness:
        return ActiveSessionLease(
            lease_id=lease_id,
            session_id=session_id,
            surface=surface,
            enabled=False,
        ), None

    entry = _lease_entry(
        lease_id=lease_id,
        session_id=str(session_id),
        surface=str(surface),
        metadata=metadata,
        track_liveness=track_liveness,
    )

    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        try:
            raw_entries = _read_entries(state_path, strict=True)
            entries = _prune_dead(raw_entries, strict=track_liveness)
        except ActiveSessionRegistryError:
            if track_liveness:
                raise
            logger.warning(
                "Active-session registry is unavailable; allowing an "
                "untracked session without overwriting it"
            )
            return ActiveSessionLease(
                lease_id=lease_id,
                session_id=session_id,
                surface=surface,
                enabled=False,
                state_path=state_path,
                lock_path=lock_path,
            ), None
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
            return None, active_session_limit_message(
                active_count, max_sessions, entries
            )
        entries.append(entry)
        _write_entries(state_path, entries)

    return ActiveSessionLease(
        lease_id=lease_id,
        session_id=str(session_id),
        surface=str(surface),
        state_path=state_path,
        lock_path=lock_path,
        track_liveness=track_liveness,
    ), None


def release_active_session(lease: ActiveSessionLease) -> None:
    # Prefer the registry the lease was acquired against: the caller may be
    # running under a profile HERMES_HOME override (#85431).
    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        if lease.released:
            return
        try:
            raw_entries = _read_entries(state_path, strict=True)
            entries = _prune_dead(raw_entries, strict=lease.track_liveness)
        except ActiveSessionRegistryError:
            if lease.track_liveness:
                raise
            logger.warning(
                "Active-session registry is unavailable; preserving it while "
                "releasing an untracked lease"
            )
            lease.released = True
            return
        kept = [
            entry
            for entry in entries
            if str(entry.get("lease_id") or "") != lease.lease_id
        ]
        if len(kept) != len(entries):
            _write_entries(state_path, kept)
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

    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        # release() may have won after the optimistic precheck but before this
        # thread acquired the file lock. Never resurrect a durably removed lease.
        if lease.released:
            return False
        try:
            raw_entries = _read_entries(state_path, strict=True)
            entries = _prune_dead(raw_entries, strict=lease.track_liveness)
        except ActiveSessionRegistryError:
            if lease.track_liveness:
                raise
            logger.warning(
                "Active-session registry is unavailable; refusing to overwrite "
                "it during lease transfer"
            )
            return False
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
        if not updated and lease.track_liveness:
            entries.append(
                _lease_entry(
                    lease_id=lease.lease_id,
                    session_id=new_session_id,
                    surface=lease.surface,
                    metadata=metadata,
                    track_liveness=True,
                )
            )
            updated = True
        if updated:
            _write_entries(state_path, entries)
            lease.session_id = new_session_id
        return updated


def release_orphaned_leases(live_lease_ids: set[str]) -> int:
    """Drop this process's registry entries that no live session owns.

    ``_prune_dead`` only reclaims leases whose owning process died. A server
    that runs for days (``hermes dashboard`` / ``serve``) never trips that
    check, so a lease whose session skipped teardown is held until restart.
    The owning process is the only authority on which of its own leases are
    real, so it drops the rest itself — exact, with no heartbeat write on the
    turn path and no staleness threshold to tune.
    """
    pid = os.getpid()
    state_path = _state_path()
    # With the cap disabled the registry is never written, so don't take a lock
    # (or create its file) on the idle-reaper tick for the majority of installs.
    if not state_path.exists():
        return 0
    with _FileLock(_lock_path()):
        try:
            raw_entries = _read_entries(state_path, strict=True)
            entries = _prune_dead(raw_entries)
        except ActiveSessionRegistryError:
            logger.warning(
                "Active-session registry is unavailable; skipping orphaned-lease sweep"
            )
            return 0
        kept = [
            entry
            for entry in entries
            if entry.get("pid") != pid
            or str(entry.get("lease_id") or "") in live_lease_ids
        ]
        dropped = len(entries) - len(kept)
        if dropped:
            _write_entries(state_path, kept)
    return dropped


def active_session_registry_snapshot(
    registry_home: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the pruned active-session registry for diagnostics/tests."""
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        raw_entries = _read_entries(state_path, strict=True)
        entries = _prune_dead(raw_entries)
        if entries != raw_entries:
            _write_entries(state_path, entries)
        return entries


@contextmanager
def active_session_liveness_guard(
    session_id: str,
    *,
    registry_home: str | Path | None = None,
) -> Iterator[bool]:
    """Hold the registry lock while reporting whether ``session_id`` is leased.

    Keeping the lock across the caller's lifecycle mutation prevents a new
    backend from acquiring a lease and reopening the row between the liveness
    check and the corresponding ``end_session`` write.
    """
    target = str(session_id or "")
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        entries = _prune_dead(_read_entries(state_path, strict=True), strict=True)
        _write_entries(state_path, entries)
        yield bool(target) and any(
            str(entry.get("session_id") or "") == target for entry in entries
        )


@contextmanager
def release_active_session_liveness_guard(
    lease: ActiveSessionLease,
    session_id: str,
) -> Iterator[bool]:
    """Remove ``lease`` and hold its registry lock through a lifecycle write.

    This makes automatic cleanup one atomic ownership decision: the local
    runtime disappears, sibling liveness is checked, and the caller may end the
    durable row before any new backend can acquire/reopen it.
    """
    if not lease.enabled or lease.released:
        with active_session_liveness_guard(
            session_id, registry_home=_registry_home_for_lease(lease)
        ) as active:
            yield active
        return

    target = str(session_id or "")
    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        raw_entries = _read_entries(state_path, strict=True)
        entries = _prune_dead(raw_entries, strict=True)
        kept = [
            entry
            for entry in entries
            if str(entry.get("lease_id") or "") != lease.lease_id
        ]
        if len(kept) != len(entries):
            _write_entries(state_path, kept)
        lease.released = True
        yield bool(target) and any(
            str(entry.get("session_id") or "") == target for entry in kept
        )


def _registry_home_for_lease(lease: ActiveSessionLease) -> Path | None:
    if lease.state_path is None:
        return None
    return lease.state_path.parent.parent
