"""Cross-process active chat session leases.

The session database records persisted conversations; this module records
currently open chat surfaces, including idle CLI/TUI sessions that have not
written a transcript row yet.
"""

from __future__ import annotations

import json
import logging
import collections
import math
import os
import time
import uuid
from contextlib import contextmanager, suppress
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
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(value)
        parsed = int(value.strip(), 10) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)", key, value
        )
        return None
    return parsed if parsed > 0 else None


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
    counts = collections.Counter(str(e.get("surface") or "unknown") for e in entries)
    held = ", ".join(
        f"{surface} x{n}" if n > 1 else surface
        for surface, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    started = [t for t in (_optional_float(e.get("started_at")) for e in entries) if t]
    if started:
        held += f", oldest {format_age(time.time() - min(started))} ago"
    return held


def active_session_limit_message(
    active_count: int, max_sessions: int, entries: Optional[list[dict[str, Any]]] = None
) -> str:
    # Name the holders: slots are shared across CLI, desktop/TUI and gateway,
    # so the rejected surface is usually NOT the one squatting on them.
    held = summarize_holders(entries or [])
    detail = f" Held by: {held}." if held else ""
    return (
        f"Hermes is at the active session limit ({active_count}/{max_sessions})."
        f"{detail} Try again when another session finishes."
    )


# Machine-readable refusal reasons (the reason is the contract, the message is for
# people). Capacity = "busy, come back later"; ownership = "a live owner exists and
# writing would interleave with theirs".
SESSION_NOT_OWNED = "SESSION_NOT_OWNED"
MAX_CONCURRENT_SESSIONS = "MAX_CONCURRENT_SESSIONS"
# Ownership could not be PROVEN (registry unreadable/corrupt). Deliberately distinct
# from SESSION_NOT_OWNED: treating "can't tell" as a go-ahead is the fail-open hole
# that let two writers share one session.
# Distinct from SESSION_NOT_OWNED on purpose -- "someone else owns this" and "I cannot tell who owns this"
# call for different operator action, and collapsing the second into a silent go-ahead is exactly the
# fail-open hole that let two writers share one session (#94595 review, blocker 2).
SESSION_COORDINATION_UNAVAILABLE = "SESSION_COORDINATION_UNAVAILABLE"

# Advertised through the gateway. A module constant, not a config flag: it holds
# because try_acquire_active_session checks atomically, so it cannot drift from the
# enforcement without this file changing.
PER_SESSION_EXCLUSIVE_SUBMIT = True


class ActiveSessionRefusal(str):
    """Refusal message (a ``str``, so callers are untouched) with a machine-readable ``reason``."""

    reason: str

    def __new__(cls, message: str, reason: str) -> "ActiveSessionRefusal":
        obj = super().__new__(cls, message)
        obj.reason = reason
        return obj


def _is_same_writer(entry: dict[str, Any], metadata: Optional[dict[str, Any]]) -> bool:
    """True when an existing lease belongs to the very caller re-acquiring it.
    Identity is (pid, live_session_id): pid alone lets two live sessions in one process
    steal each other's lease; the live id alone lets another process with an equal id."""
    try:
        if int(entry.get("pid") or -1) != os.getpid():
            return False
    except (TypeError, ValueError):
        return False
    existing_live = str((entry.get("metadata") or {}).get("live_session_id") or "")
    incoming_live = str((metadata or {}).get("live_session_id") or "")
    return bool(existing_live and incoming_live) and existing_live == incoming_live


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


def _registry_home(registry_home: str | Path | None = None) -> Path:
    return Path(registry_home) if registry_home is not None else Path(get_hermes_home())


def _state_path(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime" / "active_sessions.json"


def _lock_path(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime" / "active_sessions.lock"


def _lease_paths(
    lease: Optional["ActiveSessionLease"] = None, registry_home: str | Path | None = None
) -> tuple[Path, Path]:
    if lease is not None and lease.state_path is not None and lease.lock_path is not None:
        return lease.state_path, lease.lock_path
    home = _registry_home(registry_home)
    return home / "runtime" / "active_sessions.json", home / "runtime" / "active_sessions.lock"


def _flock(fh, *, lock: bool) -> None:
    """Exclusive whole-file lock/unlock on ``fh`` (fcntl on POSIX, msvcrt on Windows)."""
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if lock else fcntl.LOCK_UN)


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            _flock(self._fh, lock=True)
        except Exception as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("active session file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        fh, self._fh = self._fh, None
        if fh is not None:
            with suppress(Exception):
                _flock(fh, lock=False)
            fh.close()


def _read_entries(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    def invalid(what: str) -> ActiveSessionRegistryError:
        return ActiveSessionRegistryError(f"active session registry {what}: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception as exc:
        if strict:
            raise invalid("unreadable") from exc
        logger.warning("Ignoring corrupt active session registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        if strict:
            raise invalid("has invalid shape")
        return []
    valid = [entry for entry in entries if isinstance(entry, dict)]
    if not strict:
        return valid
    if len(valid) != len(entries):
        raise invalid("contains invalid entries")
    seen_leases: set[str] = set()
    for entry in valid:
        lease_id = entry.get("lease_id")
        # (problem-if-True predicate, message fragment) — checked lazily, in
        # this order, so an unhashable lease id is reported before the dup check.
        for bad, what in (
            (lambda: not _nonblank_str(lease_id), "an invalid lease id"),
            (lambda: lease_id in seen_leases, "a duplicate lease id"),
            (lambda: not _nonblank_str(entry.get("session_id")), "an invalid session id"),
            (lambda: _registry_pid(entry.get("pid")) <= 0, "an invalid pid"),
            (lambda: not _optional_isinstance(entry.get("surface"), str), "an invalid surface"),
            (lambda: not _optional_isinstance(entry.get("track_liveness"), bool), "an invalid liveness marker"),
            (lambda: not _optional_isinstance(entry.get("metadata"), dict), "invalid metadata"),
            (lambda: not _valid_process_start(entry.get("process_start_time")), "an invalid process start time"),
        ):
            if bad():
                raise invalid(f"contains {what}")
        seen_leases.add(lease_id)
    return valid


def _nonblank_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _optional_isinstance(v: Any, typ) -> bool:
    return v is None or isinstance(v, typ)


def _registry_pid(pid: Any) -> int:
    """Registry pid as int; 0 for bools, non-int/str, or unparseable values."""
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return 0
    try:
        return int(pid)
    except (TypeError, ValueError):
        return 0


def _valid_process_start(v: Any) -> bool:
    if v in (None, ""):
        return True
    parsed = _optional_float(v)
    return parsed is not None and math.isfinite(parsed)


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
    # Pair pid with create_time when psutil can read it, so a recycled pid does not
    # keep a stale lease alive indefinitely.
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


def _pid_liveness(pid: Any, process_start_time: Any = None, *, lenient: bool = False) -> Optional[bool]:
    """True/False for live/dead, or None when unknowable. ``lenient`` never returns None:
    an unparseable pid or failed existence probe counts as dead, an unreadable start as alive."""
    unknown_dead = False if lenient else None
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        pid_int = 0
    if pid_int <= 0:
        return unknown_dead
    try:
        from gateway.status import _pid_exists
        exists = bool(_pid_exists(pid_int))
    except Exception:
        return unknown_dead
    if not exists:
        return False
    expected_start = _optional_float(process_start_time)
    if expected_start is None:
        return True
    current_start = _process_start_time(pid_int)
    if current_start is None:
        return True if lenient else None
    return abs(current_start - expected_start) < 0.001


def _prune_dead(entries: list[dict[str, Any]], *, strict: bool = False) -> list[dict[str, Any]]:
    """Keep entries whose owner is alive; tracked/strict entries must be provably so."""
    live: list[dict[str, Any]] = []
    for entry in entries:
        tracked = strict or bool(entry.get("track_liveness"))
        state = _pid_liveness(
            entry.get("pid"), entry.get("process_start_time"), lenient=not tracked
        )
        if state is None:
            raise ActiveSessionRegistryError("active session owner liveness is unknown")
        if state:
            live.append(entry)
    return live


@dataclass
class ActiveSessionLease:
    lease_id: str
    session_id: str
    surface: str
    enabled: bool = True
    released: bool = False
    # Pinned at acquisition: a lease taken under the root HERMES_HOME must release
    # against the same registry even inside a profile-home override, or phantom
    # leases fill the session cap.
    # See #85431.
    state_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    track_liveness: bool = False

    def release(self) -> None:
        if self.released or not self.enabled:
            return
        release_active_session(self)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in metadata.items() if isinstance(k, str)}


def _drop_lease(
    state_path: Path, entries: list[dict[str, Any]], lease_id: str
) -> list[dict[str, Any]]:
    """Remove ``lease_id`` from ``entries``, writing the registry only if it was present."""
    kept = [e for e in entries if str(e.get("lease_id") or "") != lease_id]
    if len(kept) != len(entries):
        _write_entries(state_path, kept)
    return kept


def _holds_session(entries: list[dict[str, Any]], session_id: str) -> bool:
    target = str(session_id or "")
    return bool(target) and any(str(e.get("session_id") or "") == target for e in entries)


def _read_live_entries(
    state_path: Path, *, track_liveness: bool, warn: str,
) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """``(raw, pruned)`` from the registry, or None when it is unreadable.

    Liveness-tracked callers re-raise instead (they must not proceed on an unprovable
    registry); untracked callers get ``warn`` logged and decide how to degrade.
    """
    try:
        raw_entries = _read_entries(state_path, strict=True)
        return raw_entries, _prune_dead(raw_entries, strict=track_liveness)
    except ActiveSessionRegistryError:
        if track_liveness:
            raise
        logger.warning(warn)
        return None


def _lease_entry(
    *, lease_id: str, session_id: str, surface: str,
    metadata: Optional[dict[str, Any]] = None, track_liveness: bool = False,
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
        entry["metadata"] = _clean_metadata(metadata)
    return entry


def try_acquire_active_session(
    *, session_id: str, surface: str, config: Any, metadata: Optional[dict[str, Any]] = None,
    registry_home: str | Path | None = None, track_liveness: bool = False,
) -> tuple[Optional[ActiveSessionLease], Optional[str]]:
    """Acquire an active-session slot: ``(lease, None)`` or ``(None, ActiveSessionRefusal)``.

    Per-session exclusivity is CORRECTNESS, enforced unconditionally (at most one live
    owner per stored session); ``max_concurrent_sessions`` is resource POLICY, applied
    only when configured. ``registry_home`` lets profile-scoped backends share the owning
    profile's registry. Ownership uncertainty fails CLOSED (SESSION_COORDINATION_UNAVAILABLE).

    Liveness tracking keeps richer desktop lifecycle semantics; ``registry_home`` lets profile-scoped
    backends share the owning profile's registry even when launched from another home. See #94595.
    """
    max_sessions = resolve_max_concurrent_sessions(config)
    lease_id = uuid.uuid4().hex
    key = str(session_id or "")

    # No stored id yet => nothing to fence or record (and the strict schema
    # refuses empty session ids): hand back a no-op lease.
    if not key and not track_liveness:
        return ActiveSessionLease(
            lease_id=lease_id, session_id=key, surface=str(surface), enabled=False
        ), None

    entry = _lease_entry(
        lease_id=lease_id, session_id=key, surface=str(surface), metadata=metadata,
        track_liveness=track_liveness,
    )
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    lease = ActiveSessionLease(
        lease_id=lease_id, session_id=key, surface=str(surface), state_path=state_path,
        lock_path=lock_path, track_liveness=track_liveness,
    )
    with _FileLock(lock_path):
        # A capacity cap could degrade open; exclusivity cannot: "could not
        # prove ownership" must never become "no owner exists".
        loaded = _read_live_entries(
            state_path, track_liveness=track_liveness,
            warn="Active-session registry is unavailable; refusing the session "
                 "rather than risking a concurrent writer",
        )
        if loaded is None:
            return None, ActiveSessionRefusal(
                "Hermes could not read the active-session registry at "
                f"{state_path}, so it cannot prove this session has no other "
                "live owner. Fix or remove that file and try again.",
                SESSION_COORDINATION_UNAVAILABLE,
            )
        raw_entries, entries = loaded
        pruned = len(raw_entries) - len(entries)
        if pruned:
            logger.info("Pruned %d stale active session lease(s)", pruned)

        def refuse(message: str, reason: str, log: str, *args) -> tuple[None, ActiveSessionRefusal]:
            _write_entries(state_path, entries)  # persist the prune even when refusing
            logger.info(log, *args)
            return None, ActiveSessionRefusal(message, reason)

        # Correctness first, under the same lock that just pruned dead owners.
        # An empty key is exempt: treating "" as an identity would make every
        # unsaved draft exclude every other one.
        if key:
            for index, existing in enumerate(entries):
                if str(existing.get("session_id") or "") != key:
                    continue
                # The same writer is not a second writer: a live session that
                # leaked its lease reference would otherwise be fenced out of
                # its own session permanently (pruning only removes entries
                # whose PROCESS is dead). Re-entrancy, not concurrency.
                if _is_same_writer(existing, metadata):
                    entries[index] = entry
                    _write_entries(state_path, entries)
                    return lease, None
                return refuse(
                    session_already_owned_message(key, existing), SESSION_NOT_OWNED,
                    "Refused active session %s: already held by pid=%s surface=%s",
                    key, existing.get("pid"), existing.get("surface"),
                )

        # Capacity second, and only when an operator asked for one.
        if max_sessions is not None and len(entries) >= max_sessions:
            return refuse(
                active_session_limit_message(len(entries), max_sessions, entries),
                MAX_CONCURRENT_SESSIONS,
                "Active session limit reached: active=%d max=%d surface=%s",
                len(entries), max_sessions, surface,
            )
        entries.append(entry)
        _write_entries(state_path, entries)

    return lease, None


def release_active_session(lease: ActiveSessionLease) -> None:
    # Prefer the registry the lease was acquired against: the caller may be
    # running under a profile HERMES_HOME override.
    # See #85431.
    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        if lease.released:
            return
        loaded = _read_live_entries(
            state_path, track_liveness=lease.track_liveness,
            warn="Active-session registry is unavailable; preserving it while "
                 "releasing an untracked lease",
        )
        if loaded is not None:
            _drop_lease(state_path, loaded[1], lease.lease_id)
        lease.released = True


def drop_lease_for_session(
    session_id: str,
    *, registry_home: str | Path | None = None,
    profile_home: str | Path | None = None,
) -> bool:
    """Remove any active-session lease(s) for ``session_id`` from the on-disk registry.

    Used when a cold resume re-attaches to a session whose in-memory entry was evicted:
    the old lease (held by the now-evicted runtime's live_session_id) is stale and must
    be cleared so the new surface can claim its own lease on the next submit.
    """
    target = str(session_id or "")
    if not target:
        return False
    # Prefer the profile_home override if given (profile-scoped backend), else root.
    home = profile_home if profile_home is not None else registry_home
    state_path, lock_path = _lease_paths(registry_home=home)
    with _FileLock(lock_path):
        raw_entries = _read_entries(state_path, strict=True)
        kept = [e for e in raw_entries if str(e.get("session_id") or "") != target]
        if len(kept) != len(raw_entries):
            _write_entries(state_path, kept)
            return True
    return False


def transfer_active_session(
    lease: ActiveSessionLease, *, session_id: str, metadata: Optional[dict[str, Any]] = None
) -> bool:
    """Move an existing lease to a new session id without dropping the slot."""
    new_session_id = str(session_id or "")
    if not new_session_id or lease.released:
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
        loaded = _read_live_entries(
            state_path, track_liveness=lease.track_liveness,
            warn="Active-session registry is unavailable; refusing to overwrite "
                 "it during lease transfer",
        )
        if loaded is None:
            return False
        entries = loaded[1]
        own = next((e for e in entries if str(e.get("lease_id") or "") == lease.lease_id), None)
        if own is not None:
            own["session_id"] = new_session_id
            own["updated_at"] = time.time()
            if metadata:
                own["metadata"] = _clean_metadata(metadata)
        elif lease.track_liveness:
            entries.append(_lease_entry(
                lease_id=lease.lease_id, session_id=new_session_id, surface=lease.surface,
                metadata=metadata, track_liveness=True,
            ))
        else:
            return False
        _write_entries(state_path, entries)
        lease.session_id = new_session_id
        return True


# A lease this process wrote in the last few seconds may not be in the caller's
# ``own_live_lease_ids`` yet: ``try_acquire_active_session`` writes the registry entry under
# the file lock and the server attaches the lease to its session record only after that
# returns. A concurrent finalize that snapshotted its live ids in between would otherwise
# read the brand-new lease as an orphan and drop it. Real orphans are minutes old.
# See #101415.
_SELF_ORPHAN_GRACE_SECONDS = 30.0


def _drop_self_orphans(
    entries: list[dict[str, Any]], own_live_lease_ids: set[str] | None
) -> list[dict[str, Any]]:
    """Drop this process's leases only when its caller can vouch for owners."""
    if own_live_lease_ids is None:
        return entries
    pid = os.getpid()
    cutoff = time.time() - _SELF_ORPHAN_GRACE_SECONDS
    return [
        entry for entry in entries
        if entry.get("pid") != pid
        or str(entry.get("lease_id") or "") in own_live_lease_ids
        or (_optional_float(entry.get("started_at")) or 0.0) > cutoff
    ]


def _release_orphaned_leases_in_home(registry_home: Path, live_lease_ids: set[str]) -> int:
    state_path = _state_path(registry_home)
    # No registry file yet means no leases have ever been written under this
    # home — don't take a lock (or create its file) on the idle-reaper tick.
    if not state_path.exists():
        return 0
    with _FileLock(_lock_path(registry_home)):
        loaded = _read_live_entries(
            state_path, track_liveness=False,
            warn="Active-session registry is unavailable; skipping orphaned-lease sweep",
        )
        if loaded is None:
            return 0
        entries = loaded[1]
        kept = _drop_self_orphans(entries, live_lease_ids)
        dropped = len(entries) - len(kept)
        if dropped:
            _write_entries(state_path, kept)
        return dropped


def release_orphaned_leases(live_lease_ids: set[str]) -> int:
    """Drop this process's registry entries that no live session owns.

    ``_prune_dead`` only reclaims leases of dead processes, so on a days-long server a
    lease whose session skipped teardown is held until restart. The owning process is the
    only authority on its own leases — exact, no heartbeat on the turn path, no threshold.
    Sweeps the root home and every profile home (a multiplexed server leases across them).
    """
    root = get_default_hermes_root()
    homes = [root]
    try:
        homes.extend(p for p in (root / "profiles").iterdir()
                     if p.is_dir() and not p.name.startswith("."))
    except OSError:
        pass

    dropped = 0
    for home in homes:
        try:
            dropped += _release_orphaned_leases_in_home(home, live_lease_ids)
        except OSError as exc:
            logger.debug("orphaned-lease sweep failed for %s: %s", home, exc)
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
    session_id: str, *, registry_home: str | Path | None = None,
    own_live_lease_ids: set[str] | None = None,
) -> Iterator[bool]:
    """Hold the registry lock while reporting whether ``session_id`` is leased, so no
    new backend can acquire a lease between the check and the caller's ``end_session``."""
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        entries = _prune_dead(_read_entries(state_path, strict=True), strict=True)
        entries = _drop_self_orphans(entries, own_live_lease_ids)
        _write_entries(state_path, entries)
        yield _holds_session(entries, session_id)


@contextmanager
def release_active_session_liveness_guard(
    lease: ActiveSessionLease, session_id: str, *, own_live_lease_ids: set[str] | None = None,
) -> Iterator[bool]:
    """Remove ``lease`` and hold its registry lock through a lifecycle write, making
    cleanup one atomic decision (release, check siblings, end the durable row)."""
    if not lease.enabled or lease.released:
        home = lease.state_path.parent.parent if lease.state_path is not None else None
        with active_session_liveness_guard(
            session_id, registry_home=home, own_live_lease_ids=own_live_lease_ids,
        ) as active:
            yield active
        return

    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        entries = _prune_dead(_read_entries(state_path, strict=True), strict=True)
        kept = [e for e in entries if str(e.get("lease_id") or "") != lease.lease_id]
        kept = _drop_self_orphans(kept, own_live_lease_ids)
        if len(kept) != len(entries):
            _write_entries(state_path, kept)
        lease.released = True
        yield _holds_session(kept, session_id)
