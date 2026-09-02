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

from hermes_constants import get_default_hermes_root, get_hermes_home

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


# WHY A REFUSAL IS REFUSED, in a form a caller can branch on.
#
# The two refusals mean opposite things to an automated client. Capacity is
# "the machine is busy, come back later". Ownership is "this specific session
# has a live owner, and writing to it would interleave with theirs".
#
# Callers used to have only the human-readable message, so anything that needed
# to DECIDE had to match prose -- which silently changes meaning whenever the
# wording is improved. The reason is the contract; the message is for people.
SESSION_NOT_OWNED = "SESSION_NOT_OWNED"
MAX_CONCURRENT_SESSIONS = "MAX_CONCURRENT_SESSIONS"
# Ownership could not be PROVEN either way: the registry was unreadable or
# corrupt. Distinct from SESSION_NOT_OWNED on purpose -- "someone else owns
# this" and "I cannot tell who owns this" call for different operator action,
# and collapsing the second into a silent go-ahead is exactly the fail-open
# hole that let two writers share one session (#94595 review, blocker 2).
SESSION_COORDINATION_UNAVAILABLE = "SESSION_COORDINATION_UNAVAILABLE"

# Advertised through the gateway so a client can tell a build that enforces
# per-session exclusivity from one that does not.
#
# A module constant rather than a config flag, deliberately: it is true because
# try_acquire_active_session below performs the check atomically, so it cannot
# be turned on by an operator who has not got the enforcement, and cannot drift
# out of step with it without this file changing.
PER_SESSION_EXCLUSIVE_SUBMIT = True


class ActiveSessionRefusal(str):
    """A refusal message that also carries a machine-readable ``reason``.

    A ``str`` subclass so every existing caller keeps working untouched -- they
    format it, hand it back as a JSON-RPC message, or just test it for None --
    while a caller that must act on WHICH refusal happened reads ``.reason``
    instead of matching the wording.
    """

    reason: str

    def __new__(cls, message: str, reason: str) -> "ActiveSessionRefusal":
        obj = super().__new__(cls, message)
        obj.reason = reason
        return obj


def _is_same_writer(entry: dict[str, Any], metadata: Optional[dict[str, Any]]) -> bool:
    """True when an existing lease belongs to the very caller now re-acquiring it.

    Both halves are required. A pid alone would let two live sessions in one
    process steal each other's lease -- which is a real hazard, since each holds
    its own snapshot of the transcript. A live session id alone would let another
    process with a coincidentally equal id do the same.
    """
    try:
        if int(entry.get("pid") or -1) != os.getpid():
            return False
    except (TypeError, ValueError):
        return False
    existing_live = str((entry.get("metadata") or {}).get("live_session_id") or "")
    incoming_live = str((metadata or {}).get("live_session_id") or "")
    if not existing_live or not incoming_live:
        return False
    return existing_live == incoming_live


def session_already_owned_message(session_id: str, entry: dict[str, Any]) -> str:
    surface = str(entry.get("surface") or "another surface")
    pid = entry.get("pid")
    started = _optional_float(entry.get("started_at"))
    age = f", running {format_age(time.time() - started)}" if started else ""
    return (
        f"Session {session_id} already has a live owner ({surface}, pid {pid}{age}). "
        "Only one surface at a time may run a session, because a second one would "
        "reason from a transcript that does not include the first one's work."
    )


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

    Per-session exclusivity is CORRECTNESS and is enforced unconditionally:
    at most one live owner may run a given stored session, whether or not an
    operator configured ``max_concurrent_sessions`` (#94595). The concurrency
    cap remains a resource POLICY and applies only when configured. Liveness
    tracking keeps richer desktop lifecycle semantics; ``registry_home`` lets
    profile-scoped backends share the owning profile's registry even when
    launched from another home.

    Returns ``(lease, None)`` on success and ``(None, refusal)`` otherwise,
    where ``refusal`` is an :class:`ActiveSessionRefusal` carrying a
    machine-readable ``reason``. Ownership uncertainty fails CLOSED: when the
    registry cannot be read, the caller gets ``SESSION_COORDINATION_UNAVAILABLE``
    rather than a silent go-ahead that could reopen the double-writer state.
    """
    max_sessions = resolve_max_concurrent_sessions(config)
    lease_id = uuid.uuid4().hex
    key = str(session_id or "")

    # A session with no stored id yet cannot collide with another writer, and
    # the strict registry schema (rightly) refuses entries with empty session
    # ids. Nothing to fence, nothing to record: hand back a no-op lease.
    if not key and not track_liveness:
        return ActiveSessionLease(
            lease_id=lease_id,
            session_id=key,
            surface=str(surface),
            enabled=False,
        ), None

    entry = _lease_entry(
        lease_id=lease_id,
        session_id=key,
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
            # A capacity cap could afford to degrade open -- worst case, more
            # sessions than the operator wanted. Exclusivity cannot: "could not
            # prove ownership" must never be collapsed into "no owner exists",
            # because that silently reopens the exact double-writer state this
            # fence guarantees against. Refuse, and say which file to fix.
            logger.warning(
                "Active-session registry is unavailable; refusing the session "
                "rather than risking a concurrent writer"
            )
            return None, ActiveSessionRefusal(
                (
                    "Hermes could not read the active-session registry at "
                    f"{state_path}, so it cannot prove this session has no other "
                    "live owner. Fix or remove that file and try again."
                ),
                SESSION_COORDINATION_UNAVAILABLE,
            )
        pruned = len(raw_entries) - len(entries)
        if pruned:
            logger.info("Pruned %d stale active session lease(s)", pruned)

        # Correctness first, and under the same lock that just pruned the dead
        # owners -- so an owner that died is never mistaken for one that is
        # running, and a live one is never overlooked.
        #
        # An empty key is exempt: a session with no stored id yet cannot collide
        # with another, and treating "" as an identity would make every unsaved
        # draft exclude every other one.
        if key:
            for index, existing in enumerate(entries):
                if str(existing.get("session_id") or "") != key:
                    continue

                # THE SAME WRITER IS NOT A SECOND WRITER.
                #
                # A live session that lost its lease reference -- its record was
                # rebuilt in place, so the object holding the lease is unreachable
                # while the session itself is still the one being driven -- would
                # otherwise be fenced out of its own session by its own leak, and
                # permanently: pruning only removes entries whose PROCESS is dead,
                # and this process is very much alive.
                #
                # Identity here is (pid, live session id). Two processes never
                # match, because their pids differ. Two live sessions inside one
                # process never match, because their live ids differ. Only the
                # exact same writer re-acquiring its own session matches, and that
                # is re-entrancy rather than a concurrent writer.
                if _is_same_writer(existing, metadata):
                    entries[index] = entry
                    _write_entries(state_path, entries)
                    return ActiveSessionLease(
                        lease_id=lease_id,
                        session_id=key,
                        surface=str(surface),
                        state_path=state_path,
                        lock_path=lock_path,
                        track_liveness=track_liveness,
                    ), None

                _write_entries(state_path, entries)
                logger.info(
                    "Refused active session %s: already held by pid=%s surface=%s",
                    key,
                    existing.get("pid"),
                    existing.get("surface"),
                )
                return None, ActiveSessionRefusal(
                    session_already_owned_message(key, existing),
                    SESSION_NOT_OWNED,
                )

        # Capacity second, and only when an operator asked for one.
        if max_sessions is not None:
            active_count = len(entries)
            if active_count >= max_sessions:
                _write_entries(state_path, entries)
                logger.info(
                    "Active session limit reached: active=%d max=%d surface=%s",
                    active_count,
                    max_sessions,
                    surface,
                )
                return None, ActiveSessionRefusal(
                    active_session_limit_message(active_count, max_sessions, entries),
                    MAX_CONCURRENT_SESSIONS,
                )
        entries.append(entry)
        _write_entries(state_path, entries)

    return ActiveSessionLease(
        lease_id=lease_id,
        session_id=key,
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


def _drop_self_orphans(
    entries: list[dict[str, Any]], own_live_lease_ids: set[str] | None
) -> list[dict[str, Any]]:
    """Drop this process's leases only when its caller can vouch for owners."""
    if own_live_lease_ids is None:
        return entries
    pid = os.getpid()
    return [
        entry
        for entry in entries
        if entry.get("pid") != pid
        or str(entry.get("lease_id") or "") in own_live_lease_ids
    ]


def _release_orphaned_leases_in_home(
    registry_home: Path, live_lease_ids: set[str]
) -> int:
    state_path = _state_path(registry_home)
    if not state_path.exists():
        return 0
    with _FileLock(_lock_path(registry_home)):
        try:
            entries = _prune_dead(_read_entries(state_path, strict=True))
        except ActiveSessionRegistryError:
            logger.warning(
                "Active-session registry is unavailable; skipping orphaned-lease sweep"
            )
            return 0
        kept = _drop_self_orphans(entries, live_lease_ids)
        dropped = len(entries) - len(kept)
        if dropped:
            _write_entries(state_path, kept)
        return dropped


def release_orphaned_leases(live_lease_ids: set[str]) -> int:
    """Drop this process's registry entries that no live session owns.

    ``_prune_dead`` only reclaims leases whose owning process died. A server
    that runs for days (``hermes dashboard`` / ``serve``) never trips that
    check, so a lease whose session skipped teardown is held until restart.
    The owning process is the only authority on which of its own leases are
    real, so it drops the rest itself — exact, with no heartbeat write on the
    turn path and no staleness threshold to tune.
    """
    root = get_default_hermes_root()
    homes = [root]
    profiles_root = root / "profiles"
    try:
        homes.extend(
            profile
            for profile in profiles_root.iterdir()
            if profile.is_dir() and not profile.name.startswith(".")
        )
    except OSError:
        pass

    dropped = 0
    for home in homes:
        try:
            dropped += _release_orphaned_leases_in_home(home, live_lease_ids)
        except OSError as exc:
            logger.debug(
                "orphaned-lease sweep failed for %s: %s", home, exc
            )
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
    own_live_lease_ids: set[str] | None = None,
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
        entries = _drop_self_orphans(entries, own_live_lease_ids)
        _write_entries(state_path, entries)
        yield bool(target) and any(
            str(entry.get("session_id") or "") == target for entry in entries
        )


@contextmanager
def release_active_session_liveness_guard(
    lease: ActiveSessionLease,
    session_id: str,
    *,
    own_live_lease_ids: set[str] | None = None,
) -> Iterator[bool]:
    """Remove ``lease`` and hold its registry lock through a lifecycle write.

    This makes automatic cleanup one atomic ownership decision: the local
    runtime disappears, sibling liveness is checked, and the caller may end the
    durable row before any new backend can acquire/reopen it.
    """
    if not lease.enabled or lease.released:
        with active_session_liveness_guard(
            session_id,
            registry_home=_registry_home_for_lease(lease),
            own_live_lease_ids=own_live_lease_ids,
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
        kept = _drop_self_orphans(kept, own_live_lease_ids)
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
