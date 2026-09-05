"""Cross-process writer coordination for ``state.db`` (see #103339).

Design (v3): SQLite owns *row* concurrency; this gate owns *file-structure*
concurrency. Ordinary row writes (appends, claims, handoffs, leases) from any
number of processes proceed under SQLite's own WAL locking — exactly as on
base, where a pinned conformance cell proves 8 concurrent claimants stay
exactly-once. What this gate forbids is *structural* work under a live
writer — schema surgery, checkpoints from a second connection, VACUUM-class
operations — the evidenced corruption class: the second connection's WAL
handling unlinks/replaces the WAL inode the owning gateway holds.

Mechanism, two files beside the database:

- Presence: ``<state.db>.writer.<pid>.lock``. Each writing process holds its
  OWN file exclusive for as long as it may write (first write → close).
  Per-pid files never contend with each other, so any number of writers
  coexist; a probe enumerates them to learn that a writer lives. Crashed
  writers leave litter, which the probe reaps (see below) — no wedged state.
- Structural exclusion: ``<state.db>.writer.lock`` (global). A structural
  operation (today: schema repair) holds it exclusive for the whole surgery.
  Row announces refuse while it is held, and it is only grantable when no
  writer presence is live — closing the race in both directions.

Fail-closed everywhere: contention, an unopenable file, or an unreadable
holder record refuses the structural op, never allows it. Row announces
refuse only while a surgery holds the global lock.

Probe liveness rules for a presence file (all best-effort, never raising):

- Our own pid's file, registered locally → ours, skip.
- Exclusive flock still acquirable → holder gone. Unlink + ignore, UNLESS the
  file is brand new with no readable record (a racing announcer mid-create)
  or the record names a live pid (a mid-close race) — both read as LIVE.
- Exclusive flock contended → live holder; role comes from its record
  (``writer``; unreadable → ``unknown``, still a refusal).

Roles (``writer`` / ``repair``) live in the global lock's record so repair
can tell a fellow repairer (serialize via the repair lock — queuing is
correct) from a live writer (refuse).

Ownership is per SessionDB instance: ``SessionDB.close()`` releases its
share, and presence drops when the last in-process owner closes. A live
gateway (registry-owned handle, never closed mid-life) therefore announces
exactly while it can write; a CLI that wrote and closed does not pin
presence against a later repair. (``close()`` on a registry-shared instance
only decrements the registry refcount; the share releases on final registry
release, which runs the real ``close()``.)

``fork()`` needs no pid guard by construction: presence files are keyed by
pid, so the child naturally announces under its own pid, and release only
ever unlinks a file whose pid matches the releaser — a child can never
delete its parent's presence. A child that never touches the gate leaves no
trace; one that writes announces like any other second process.

Read-only paths never touch the gate (``sessions list/export``, backups,
health probes, ``SessionDB(read_only=True)``).

Windows tier: ``msvcrt.locking`` is exclusive-only but that is all this
design needs (per-pid files are single-owner; the global lock is exclusive).
Same semantics as POSIX.

Known limitation: open-time writes that bypass ``_execute_write`` (the
``state_meta`` generation stamp, schema init DDL) announce nothing — gating
the open would refuse speculative writer handles that never write
(``sessions list`` while the gateway runs).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_state_common import (
    _proc_start_ticks,
    _read_lock_holder_record,
    _rewrite_lock_file,
    is_advisory_lock_contention,
)
from hermes_state_errors import StateDbWriterHeldError

logger = logging.getLogger("hermes_state")

_IS_WINDOWS = sys.platform == "win32"

#: Global structural lock: ``<state.db>.writer.lock`` sits beside the database.
_WRITER_LOCK_SUFFIX = ".writer.lock"
#: Presence-file infixes: ``<state.db>.writer.<pid>.lock``.
_WRITER_PRESENCE_INFIX = ".writer."
_PRESENCE_SUFFIX = ".lock"

#: Fresh-file window (seconds): a flock-free presence file younger than this
#: with no readable record is a racing announcer, not litter — read as live.
_PRESENCE_SETTLE_SECONDS = 60.0

#: Holder roles recorded in lock files.
_WRITER_ROLE = "writer"
_REPAIR_ROLE = "repair"
_UNKNOWN_ROLE = "unknown"


class _GateHold:
    """One held lock file plus its in-process owners and acquirer pid."""

    __slots__ = ("handle", "owners", "role", "acquired_pid")

    def __init__(self, handle, role: str):
        self.handle = handle
        self.role = role
        self.owners = weakref.WeakSet()
        self.acquired_pid = os.getpid()


class OwnerToken:
    """Weakref-able owner token for :func:`acquire_writer_gate` (a bare
    ``object()`` cannot be weak-referenced). SessionDB instances pass
    themselves; one-shot holders (repair surgery) mint one of these."""

    __slots__ = ("label", "__weakref__")

    def __init__(self, label: str = ""):
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"OwnerToken({self.label!r})"


#: Lock path -> live hold. Guarded by ``_held_lock``.
_held: Dict[str, _GateHold] = {}
_held_lock = threading.Lock()


def _writer_lock_path(db_path: Path) -> Path:
    """Global structural lock file for *db_path*."""
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    return resolved.with_name(resolved.name + _WRITER_LOCK_SUFFIX)


def _presence_path(db_path: Path, pid: int) -> Path:
    """This process's presence file for *db_path* (never shared across pids,
    so a symlink and its target resolve to one gate like the registry)."""
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    return resolved.with_name(
        f"{resolved.name}{_WRITER_PRESENCE_INFIX}{pid}{_PRESENCE_SUFFIX}"
    )


def _presence_pid(lock_path: Path, db_name: str) -> Optional[int]:
    """PID encoded in a presence filename, or None when it is not one."""
    name = lock_path.name
    prefix = db_name + _WRITER_PRESENCE_INFIX
    if not name.startswith(prefix) or not name.endswith(_PRESENCE_SUFFIX):
        return None
    try:
        return int(name[len(prefix):-len(_PRESENCE_SUFFIX)])
    except ValueError:
        return None


def _cmdline_of(pid: int) -> Optional[str]:
    """Best-effort ``pid -> cmdline`` for refusal messages; None when unknowable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            return cmd[:200] if cmd else None
    except (OSError, ValueError):
        return None


def _read_gate_record(lock_path: Path) -> Optional[dict]:
    """Best-effort parse of the holder record; None when unreadable."""
    try:
        with lock_path.open("rb") as fh:
            record = _read_lock_holder_record(fh)
    except OSError:
        return None
    return record if isinstance(record, dict) else None


def _holder_role(record: Optional[dict]) -> str:
    if not record:
        return _UNKNOWN_ROLE
    role = record.get("role")
    return role if role in (_WRITER_ROLE, _REPAIR_ROLE) else _UNKNOWN_ROLE


def _describe_holder(lock_path: Path, record: Optional[dict] = None) -> str:
    """Human-readable holder for refusal messages; never raises."""
    if record is None:
        record = _read_gate_record(lock_path)
    if not record:
        return f"another process holds {lock_path} (holder record unreadable)"
    pid = record.get("pid")
    try:
        pid = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return f"another process holds {lock_path} (holder record has no pid)"
    cmd = _cmdline_of(pid)
    who = f"pid {pid}" + (f" ({cmd})" if cmd else "")
    return f"another process holds {lock_path} ({who})"


def _open_gate_file(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return lock_path.open("a+b")


def _try_flock_nb(handle) -> Optional[bool]:
    """Non-blocking exclusive flock: True (held), False (contended), None
    (non-contention OSError — fail closed upstream)."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError) as exc:
        if not is_advisory_lock_contention(exc):
            logger.warning(
                "state.db writer gate: unexpected lock error (fail closed): %s", exc
            )
            return None
        return False


def _unlock_handle(handle) -> None:
    """Release a disposable handle's flock (never a registered hold)."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _write_gate_record(handle, role: str) -> None:
    """Record this process as holder (best effort, under the flock)."""
    record = {
        "pid": os.getpid(),
        "start_ticks": _proc_start_ticks(os.getpid()),
        "acquired_at": time.time(),
        "role": role,
    }
    _rewrite_lock_file(handle, json.dumps(record, sort_keys=True).encode("utf-8"))


def _file_age_seconds(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 0.0


def _live_writer_presences(db_path: Path) -> List[str]:
    """Descriptions of live foreign writer presences; reaps crash litter.

    Never raises. A flock-free file is litter ONLY when it is old or names a
    provably-absent record; a fresh file with no record is a racing announcer
    (fail closed), as is a file whose record names a live pid (mid-close).
    """
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    parent, db_name = resolved.parent, resolved.name
    try:
        candidates = list(parent.glob(db_name + _WRITER_PRESENCE_INFIX + "*" + _PRESENCE_SUFFIX))
    except OSError:
        return [f"cannot scan {parent} for writer presence (fail closed)"]
    live: List[str] = []
    me = os.getpid()
    for lock_path in candidates:
        pid = _presence_pid(lock_path, db_name)
        if pid is None:
            continue
        # One mutex around open + flock + unlock + close: sibling threads'
        # identical sequences can never overlap, so probes never mistake each
        # other for a foreign holder (same-process flock descriptions
        # contend). Record reads and litter reaps stay outside (fail-closed).
        # Own files skip via registry AND pid: a forked child inherits the
        # registry copy but has a new pid, so it must still see (never skip)
        # its parent's presence file.
        with _held_lock:
            if pid == me and str(lock_path) in _held:
                continue  # ours
            try:
                handle = lock_path.open("a+b")
            except OSError:
                live.append(f"unopenable presence file {lock_path} (fail closed)")
                continue
            try:
                held = _try_flock_nb(handle)
                if held is True:
                    _unlock_handle(handle)
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            if held is None:
                live.append(f"unprovable presence file {lock_path} (fail closed)")
                continue
            if held is False:
                # Contended: live holder. Role from its record (unknown refuses).
                live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                continue
        # Flock-free here: litter only when provably not a racer. A free
        # flock plus a record naming a dead pid cannot be a racing announcer
        # (announcers hold the flock while writing their record), so it is
        # reaped regardless of file age.
        record = _read_gate_record(lock_path)
        if record is None and _file_age_seconds(lock_path) < _PRESENCE_SETTLE_SECONDS:
            live.append(_describe_holder(lock_path, record))  # racing announcer
            continue
        if record is not None and _holder_role(record) == _UNKNOWN_ROLE:
            live.append(_describe_holder(lock_path, record))  # conservative
            continue
        if record is not None:
            rp = record.get("pid")
            try:
                rp = int(rp) if rp is not None else 0
            except (TypeError, ValueError):
                rp = 0
            if rp > 0:
                try:
                    os.kill(rp, 0)
                    live.append(_describe_holder(lock_path, record))  # mid-close race
                    continue
                except ProcessLookupError:
                    pass  # dead holder: reap below, not a racer (see above)
                except OSError:
                    live.append(_describe_holder(lock_path, record))  # unknowable: closed
                    continue
                # Dead pid with a record, flock already proven free above:
                # crash litter (a racing announcer holds the flock while
                # writing, so this cannot be one). Re-verify under the mutex
                # before reaping regardless of file age.
                with _held_lock:
                    try:
                        handle = lock_path.open("a+b")
                    except OSError:
                        continue
                    try:
                        if _try_flock_nb(handle) is not True:
                            live.append(_describe_holder(
                                lock_path, _read_gate_record(lock_path)))
                            continue
                        _unlock_handle(handle)
                    finally:
                        try:
                            handle.close()
                        except OSError:
                            pass
                    with contextlib.suppress(OSError):
                        lock_path.unlink()
                continue
        # Re-verify under the mutex before reaping (covers the residual
        # no-record old-litter path; dead-pid records were reaped above).
        with _held_lock:
            try:
                handle = lock_path.open("a+b")
            except OSError:
                continue
            try:
                if _try_flock_nb(handle) is not True:
                    live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                    continue
                _unlock_handle(handle)
                if _file_age_seconds(lock_path) < _PRESENCE_SETTLE_SECONDS:
                    live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                    continue
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            with contextlib.suppress(OSError):
                lock_path.unlink()
    return live


def _probe_global(db_path: Path, *, settle_role: bool = False) -> Tuple[bool, str, str]:
    """Non-acquiring probe of the global structural lock:
    ``(foreign_held, role, description)``. Never raises (fail closed)."""
    lock_path = _writer_lock_path(db_path)
    key = str(lock_path)
    attempts = 6 if settle_role else 1
    last: Tuple[bool, str, str] = (True, _UNKNOWN_ROLE, _describe_holder(lock_path))
    for _ in range(attempts):
        # Mutex Ward (see _live_writer_presences): open + flock + unlock +
        # close never overlap a sibling thread, so no false contention.
        with _held_lock:
            if key in _held:
                return False, _held[key].role, ""
            try:
                handle = _open_gate_file(lock_path)
            except OSError as exc:
                return True, _UNKNOWN_ROLE, f"cannot open {lock_path} ({exc})"
            try:
                held = _try_flock_nb(handle)
                if held is True:
                    _unlock_handle(handle)
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            if held is not True:
                if key in _held:
                    return False, _held[key].role, ""  # our own thread won the race
        if held is True:
            return False, _UNKNOWN_ROLE, ""
        record = _read_gate_record(lock_path)
        role = _holder_role(record)
        if role != _UNKNOWN_ROLE or not settle_role:
            return True, role, _describe_holder(lock_path, record)
        last = (True, role, _describe_holder(lock_path, record))
        time.sleep(0.2)
    return last


def _refusal(db_path: Path, what: str) -> StateDbWriterHeldError:
    return StateDbWriterHeldError(
        f"state.db writer gate: {what}; refusing structural work on {db_path} so it cannot "
        "corrupt the live WAL (see #103339). Stop that process's gateway "
        "(`hermes gateway stop`) and retry."
    )


def acquire_writer_gate(db_path: Path, *, role: str = _WRITER_ROLE, owner=None,
                        exclusive: bool = False) -> None:
    """Take this process's share of *db_path*'s writer gate.

    - Row announces (``exclusive=False``, the ``_execute_write`` path):
      idempotent per process; records presence in this process's own
      presence file and refuses ONLY while a surgery holds the global
      structural lock (fail-closed in both directions). Any number of
      processes may announce concurrently — SQLite arbitrates their row
      writes, as on base.
    - Structural takes (``exclusive=True``, repair surgery): ATOMIC —
      takes the global lock, then scans foreign writer presences, and
      releases + refuses when any are live. Only an actually-quiet database
      is ever returned held. Same-process re-entry just adds *owner*
      (per-pid presence and the global lock are separate files, so no
      self-deadlock and no upgrade protocol). This primitive is the single
      authority carrier: every structural caller (repair, checkpoints,
      future VACUUM/optimize/migration gates) takes through here instead of
      reimplementing probe-then-take.
    """
    if role not in (_WRITER_ROLE, _REPAIR_ROLE):
        role = _WRITER_ROLE
    if not exclusive:
        _announce_presence(db_path, owner=owner)
        foreign, _role, description = _probe_global(db_path)
        if foreign:
            raise StateDbWriterHeldError(
                f"state.db writer gate: {description}; refusing to write {db_path} while a "
                "structural operation owns it (see #103339). Retry once it finishes."
            )
        return
    lock_path = _writer_lock_path(db_path)
    key = str(lock_path)
    # Mutex Ward: the whole check + open + flock + register sequence runs
    # under one hold so sibling threads can never interleave two takes of
    # the same file (same-process flock descriptions contend).
    with _held_lock:
        hold = _held.get(key)
        if hold is not None:
            if owner is not None:
                hold.owners.add(owner)
            return
        try:
            handle = _open_gate_file(lock_path)
        except OSError as exc:
            raise StateDbWriterHeldError(
                f"state.db writer gate: cannot open {lock_path} ({exc}); refusing structural work on "
                f"{db_path} rather than risk a second-writer corruption (see #103339)."
            ) from exc
        # The mutex is held from the registry check through registration, so
        # no sibling thread can interleave a competing take of this file.
        # The holder record goes down BEFORE registration: a sibling probe
        # that lands mid-announce then reads a complete role instead of a
        # transient unknown (both fail closed; the former avoids spurious
        # refusals).
        held = _try_flock_nb(handle)
        if held is not True:
            try:
                handle.close()
            except OSError:
                pass
            raise _refusal(db_path, _describe_holder(lock_path))
        _write_gate_record(handle, role)
        hold = _GateHold(handle, role)
        if owner is not None:
            hold.owners.add(owner)
        _held[key] = hold
    # Atomicity: the global flock is held from here on, so verify foreign
    # writer presence BEFORE returning. A writer announcing later sees the
    # held global lock and refuses; a writer already present is seen here.
    # Either order closes fully — only an actually-quiet database is ever
    # returned held. (Fellow repairers never announce per-pid presence, so
    # they cannot trip this; they queue via the repair lock.)
    writers = _live_writer_presences(db_path)
    if writers:
        with _held_lock:
            _held.pop(key, None)
        _unlock_handle(handle)
        try:
            handle.close()
        except OSError:
            pass
        raise _refusal(db_path, "; ".join(writers))
    return


def structural_lock_held_by_other(db_path: Path) -> Optional[str]:
    """Global-structural-lock-only probe for open-time mutation sites.

    Unlike :func:`writer_gate_holder` this ignores mere writer presence:
    ordinary writers coexisting must NOT block opens — only a live surgery
    (repair/checkpoint/VACUUM-class holder) refuses constructor-time DDL
    and generation writes. Returns a holder description or None. Never
    raises (unprovable reads as held).
    """
    foreign, _role, description = _probe_global(db_path)
    return description if foreign else None


def refuse_if_structural_op_holds(db_path: Path, what: str) -> None:
    """Fail-closed entry guard for open-time mutation sites (schema DDL,
    generation stamp): constructor work must not interleave a live surgery.

    Only a held global structural lock refuses — coexisting writers never
    do. Raises :class:`StateDbWriterHeldError` with an actionable message
    (worded to avoid the write path's locked/busy retry match, so opens
    fail fast instead of waiting out a patience window).
    """
    holder = structural_lock_held_by_other(db_path)
    if holder is not None:
        raise StateDbWriterHeldError(
            f"state.db structural gate: {holder}; refusing {what} on {db_path} while a "
            "structural operation owns it (see #103339). Stop that process's gateway "
            "(`hermes gateway stop`) and retry once it finishes."
        )


def _announce_presence(db_path: Path, owner=None) -> None:
    """Record this process as a live writer (best-effort, never raises for
    contention: row writes proceed under SQLite's own locking).

    Warns loudly when presence itself cannot be recorded (exotic filesystem):
    probes may then miss this writer, so structural ops could proceed under
    it — the same exposure as base, but visible in logs.
    """
    lock_path = _presence_path(db_path, os.getpid())
    key = str(lock_path)
    # Mutex Ward (see _live_writer_presences): the whole check + open + flock
    # + register sequence runs under one hold. Without it, two sibling
    # threads announcing at once contend on the same per-pid file and the
    # loser would unlink the winner's live file as "stale".
    with _held_lock:
        if key in _held:
            # Lifetime invariant: every announcing owner registers, so the
            # presence drops only when the LAST in-process owner closes.
            if owner is not None:
                _held[key].owners.add(owner)
            return
        try:
            handle = _open_gate_file(lock_path)
        except OSError as exc:
            logger.warning(
                "state.db writer gate: cannot record presence in %s (%s); proceeding "
                "without presence (structural ops may miss this writer).", lock_path, exc)
            return
        held = _try_flock_nb(handle)
        if held is not True:
            # Same-pid stale file (crashed previous owner with a recycled pid):
            # reclaim once under the same hold, then give up loud.
            try:
                handle.close()
            except OSError:
                pass
            try:
                lock_path.unlink()
                handle = _open_gate_file(lock_path)
                held = _try_flock_nb(handle)
            except OSError as exc:
                logger.warning(
                    "state.db writer gate: cannot record presence in %s (%s); proceeding "
                    "without presence (structural ops may miss this writer).", lock_path, exc)
                return
            if held is not True:
                logger.warning(
                    "state.db writer gate: cannot lock presence file %s; proceeding "
                    "without presence (structural ops may miss this writer).", lock_path)
                try:
                    handle.close()
                except OSError:
                    pass
                return
        hold = _held.get(key)
        if hold is not None:
            if owner is not None:
                hold.owners.add(owner)
            try:
                handle.close()
            except OSError:
                pass
            return
        # Record before registration (see exclusive take above).
        _write_gate_record(handle, _WRITER_ROLE)
        hold = _GateHold(handle, _WRITER_ROLE)
        if owner is not None:
            hold.owners.add(owner)
        _held[key] = hold


def release_writer_gate(db_path: Path, owner) -> None:
    """Drop *owner*'s share; the flock goes when the last owner closes.

    Never raises. ``owner=None`` is a no-op. Presence files are unlinked only
    when their pid matches the releaser (a forked child can never delete its
    parent's presence); the global structural file is never unlinked. A hold
    acquired by another pid (fork-inherited) is dropped locally without
    unlocking — the owner's description keeps the kernel lock.
    """
    if owner is None:
        return
    global_path = _writer_lock_path(db_path)
    try:
        db_name = Path(db_path).expanduser().resolve().name
    except OSError:
        db_name = Path(db_path).name
    for lock_path in (_presence_path(db_path, os.getpid()), global_path):
        key = str(lock_path)
        # Unlock + close + unlink under the same hold (Mutex Ward): a sibling
        # thread's open + flock can never land between our unlock and close
        # and mistake itself for a foreign holder.
        with _held_lock:
            hold = _held.get(key)
            if hold is None:
                continue
            try:
                hold.owners.discard(owner)
            except (KeyError, TypeError):
                pass
            if len(list(hold.owners)) > 0:
                continue
            del _held[key]
            if hold.acquired_pid != os.getpid():
                continue  # fork-inherited: drop locally, never touch the fd
            handle = hold.handle
            _unlock_handle(handle)
            try:
                handle.close()
            except OSError:
                pass
            # Presence files are unlinked only when their pid matches the releaser
            # (a forked child can never delete its parent's presence); the global
            # structural file is never unlinked.
            if lock_path != global_path and _presence_pid(lock_path, db_name) == os.getpid():
                with contextlib.suppress(OSError):
                    lock_path.unlink()


def writer_gate_holder(db_path: Path) -> Optional[str]:
    """Non-acquiring probe: None when no foreign surgery or writer presence
    is live (or only ours is); otherwise a human-readable description.

    Used by repair/doctor/checkpoint paths that must refuse without taking
    the gate.
    """
    foreign, _role, description = _probe_global(db_path)
    if foreign:
        return description
    writers = _live_writer_presences(db_path)
    if writers:
        return "; ".join(writers)
    return None


def writer_gate_holder_role(db_path: Path) -> Optional[str]:
    """Non-acquiring probe: None when this process may do structural work;
    otherwise the live holder's role (``writer`` / ``repair`` / ``unknown``).

    Repair uses this to tell a fellow repairer (serialize via the repair
    lock — queuing is correct) from a live writer (refuse).
    """
    foreign, role, _description = _probe_global(db_path, settle_role=True)
    if foreign:
        if role == _UNKNOWN_ROLE and not _live_writer_presences(db_path):
            # Contended global lock with no readable record and no writer
            # presence: fellow repairer mid-record-write (settle already
            # retried) — report repair so we queue rather than refuse.
            return _REPAIR_ROLE
        return role
    if _live_writer_presences(db_path):
        return _WRITER_ROLE
    return None
