"""Room-epoch fences and one-promise-per-epoch records for Group Chat succession.

A participant gateway keeps one succession record per room: the highest room authority
epoch it has fenced, the latest promise it made, and the latest verified authority it learned.
Each epoch is promised at most once, to one candidate installation, and a promise is never
revoked: a later candidate can only ask for a later epoch, which fences the earlier one too.
Once epoch N is fenced here, this gateway refuses new room-grant work stamped with an epoch at
or below N. Runs it admitted before keep executing and stay readable, and their Status and Stop
pass to the promised successor, or to the verified authority once one is learned for an epoch
at least as late (a candidate that lost to a certified successor never gains control).

The record lives in the durable Runs store beside the owner's participant freezes
(``group_run_freezes``), so a fence and a run admission are ordered by one SQLite writer.
The owner freeze stays a separate, permanent, single-scope record. SQL guards keep the
record monotonic and back the admission check for every writer that shares the store.

A voting computer of a group in automatic mode also grants the group's host a short lease
(``grant_lease``): while it runs, this gateway promises no later epoch to anyone
(``room_lease_active``), so a majority of promises can only form once the host's majority of
leases has run out. Leases are measured on a clock that keeps counting while the computer sleeps
(``gateway/hosted_room_clock.py``) and survive restarts. That clock starts near zero at boot, so
after a reboot a lease counts as running for its full length less the time since the new boot:
never shorter than it really ran, and never measured on the wall clock, which can be stepped.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from gateway.hosted_room_peer import _identifier
from gateway.hosted_rooms_common import bounded_int, clock, table_exists

FENCES = "hosted_room_fences"
LEASES = "hosted_room_leases"
MAX_FENCED_ROOMS = 4096
MAX_LEASE_SECONDS = 60.0
MAX_RESTART_EXTENSION_SECONDS = 300.0
_MAX_EPOCH = 2**63 - 1
_SELECT = f"""SELECT fenced_epoch, promise_epoch, candidate_install_id, issued_at,
    authority_epoch, authority_install_id FROM {FENCES} WHERE room_id=?"""
# The epoch a recorded room scope carries, when its run identity names a fenced room.
_FENCED_IDENTITY = f"""EXISTS (SELECT 1 FROM {FENCES} AS fence
    WHERE fence.room_id=CASE WHEN json_valid({{identity}}) THEN json_extract({{identity}},'$.room_id') END
      AND CASE WHEN json_valid({{identity}}) THEN json_extract({{identity}},'$.authority_epoch') END
          <= fence.fenced_epoch)"""


class RoomFenceError(RuntimeError):
    code = "room_fence_unavailable"
    status = 503
    message = "Durable Group Chat succession storage is unavailable."

    def __init__(self):
        super().__init__(self.message)

    @property
    def reason(self) -> str:
        return self.code


class RoomAuthorityFenced(RoomFenceError):
    code = "room_authority_fenced"
    status = 409
    message = "This Group Chat's authority epoch is fenced on this gateway."


class RoomAuthorityPromised(RoomFenceError):
    code = "room_authority_promised"
    status = 409
    message = "This gateway already promised that Group Chat epoch to another installation."


class RoomAuthorityConflict(RoomFenceError):
    code = "room_authority_conflict"
    status = 409
    message = "This gateway already follows another authority at that Group Chat epoch."


class RoomLeaseActive(RoomFenceError):
    code = "room_lease_active"
    status = 409
    message = "This gateway's lease to the Group Chat's current host is still running."


class RoomFenceCapacity(RoomFenceError):
    code = "room_fence_capacity"
    status = 507
    message = "Durable Group Chat succession capacity is full."


def _exact(value: Any, field: str) -> str:
    if type(value) is not str or value != value.strip() or "\0" in value:
        raise ValueError(f"{field} must be an exact identifier string")
    return _identifier(value, field=field)


def _epoch(value: Any, field: str, low: int = 1) -> int:
    return bounded_int(value, error=ValueError, message=f"{field} must be a positive integer", low=low,
                       high=_MAX_EPOCH)


def initialize_fence_schema(conn: sqlite3.Connection) -> None:
    """Create the record and its guards; the caller holds the write transaction."""
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {FENCES} (
        room_id TEXT PRIMARY KEY,
        fenced_epoch INTEGER NOT NULL CHECK (fenced_epoch >= 1),
        promise_epoch INTEGER CHECK (promise_epoch IS NULL OR promise_epoch >= 2),
        candidate_install_id TEXT,
        issued_at REAL,
        authority_epoch INTEGER CHECK (authority_epoch IS NULL OR authority_epoch >= 2),
        authority_install_id TEXT,
        fenced_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        CHECK ((promise_epoch IS NULL) = (candidate_install_id IS NULL)
           AND (promise_epoch IS NULL) = (issued_at IS NULL)
           AND (authority_epoch IS NULL) = (authority_install_id IS NULL)))""")
    # A fence only rises; an epoch keeps its one promise and its one authority; nothing is forgotten.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_room_fence_monotonic
        BEFORE UPDATE ON {FENCES}
        WHEN NEW.room_id IS NOT OLD.room_id OR NEW.fenced_epoch < OLD.fenced_epoch
          OR NEW.fenced_at IS NOT OLD.fenced_at
          OR (OLD.promise_epoch IS NOT NULL AND (NEW.promise_epoch IS NULL
              OR NEW.promise_epoch < OLD.promise_epoch
              OR (NEW.promise_epoch = OLD.promise_epoch
                  AND (NEW.candidate_install_id IS NOT OLD.candidate_install_id
                       OR NEW.issued_at IS NOT OLD.issued_at))))
          OR (OLD.authority_epoch IS NOT NULL AND (NEW.authority_epoch IS NULL
              OR NEW.authority_epoch < OLD.authority_epoch
              OR (NEW.authority_epoch = OLD.authority_epoch
                  AND NEW.authority_install_id IS NOT OLD.authority_install_id)))
        BEGIN SELECT RAISE(ABORT, 'room fence is monotonic'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_room_fence_kept
        BEFORE DELETE ON {FENCES}
        BEGIN SELECT RAISE(ABORT, 'room fence is permanent'); END""")
    # ``granted_clock`` and ``boot_id`` are this gateway's sleep-counting clock and boot at the grant;
    # ``host_sent_at`` and ``host_boot`` the host's own clock and boot when it asked for the latest
    # renewal, which a handover statement the host signed earlier can't release.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {LEASES} (
        room_id TEXT PRIMARY KEY,
        epoch INTEGER NOT NULL CHECK (epoch >= 1),
        authority_install_id TEXT NOT NULL,
        granted_clock REAL NOT NULL,
        length_s REAL NOT NULL CHECK (length_s > 0),
        boot_id TEXT NOT NULL,
        granted_at REAL NOT NULL,
        host_sent_at REAL,
        host_boot TEXT)""")
    # Back the admission check for every writer sharing the Runs store: a fenced room epoch
    # records no new scope, and a scope already recorded for it reserves no new run.
    if table_exists(conn, "group_run_scopes"):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_room_fence_scope_insert
            BEFORE INSERT ON group_run_scopes
            WHEN {_FENCED_IDENTITY.format(identity='NEW.identity_json')}
            BEGIN SELECT RAISE(ABORT, 'room authority fenced'); END""")
        if table_exists(conn, "run_idempotency"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_room_fence_run_insert
                BEFORE INSERT ON run_idempotency
                WHEN EXISTS (SELECT 1 FROM group_run_scopes AS recorded WHERE recorded.scope=NEW.scope
                    AND {_FENCED_IDENTITY.format(identity='recorded.identity_json')})
                BEGIN SELECT RAISE(ABORT, 'room authority fenced'); END""")


def _state(row) -> dict[str, Any]:
    if row is None:
        return {"fenced_epoch": 0, "promise": None, "authority": None}
    fenced, promise_epoch, candidate, issued_at, authority_epoch, authority = row
    return {"fenced_epoch": int(fenced), "promise": None if promise_epoch is None else {
        "epoch": int(promise_epoch), "candidate_install_id": candidate, "issued_at": float(issued_at)},
        "authority": None if authority_epoch is None else {"epoch": int(authority_epoch), "install_id": authority}}


def fence_state_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    return _state(conn.execute(_SELECT, (room_id,)).fetchone() if table_exists(conn, FENCES) else None)


def fenced_epoch_locked(conn: sqlite3.Connection, room_id: str) -> int:
    return fence_state_locked(conn, room_id)["fenced_epoch"]


def successor_controls_locked(conn: sqlite3.Connection, room_id: str, *, candidate_install_id: str,
                              epoch: int | None = None) -> bool:
    """Whether control passed here to this candidate, at ``epoch`` when given: the learned authority
    for its epoch and later, else the live promise."""
    state = fence_state_locked(conn, room_id)
    promise, authority = state["promise"], state["authority"]
    if authority is not None and (promise is None or authority["epoch"] >= promise["epoch"]):
        holder, held_epoch = authority["install_id"], authority["epoch"]
    elif promise is not None and promise["epoch"] > state["fenced_epoch"]:
        holder, held_epoch = promise["candidate_install_id"], promise["epoch"]
    else:
        return False
    return holder == candidate_install_id and (epoch is None or held_epoch == epoch)


def lease_remaining(*, granted_clock: float, length_s: float, boot: str, granted_at: float,
                    now: float | None = None) -> float:
    """Seconds a lease granted at ``granted_clock`` (this gateway's sleep-counting clock, in ``boot``)
    still runs, never fewer than it really does.

    In the same boot, the clock tells. After a reboot at least the time since the new boot went by,
    and a clock below the grant's own reading proves a reboot. Only a computer without a
    sleep-counting clock measures on the wall clock (``granted_at``).
    """
    from gateway import hosted_room_clock
    if not hosted_room_clock.EXACT:
        return float(granted_at) + float(length_s) - clock(now)
    current, here = hosted_room_clock.now(), str(hosted_room_clock.boot_id() or "")
    rebooted = (bool(boot) and bool(here) and str(boot) != here) or current < float(granted_clock)
    if rebooted:
        return float(length_s) - current
    return float(granted_clock) + float(length_s) - current


def lease_locked(conn: sqlite3.Connection, room_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The lease this gateway granted the room's host, while it still runs; else ``None``."""
    if not table_exists(conn, LEASES):
        return None
    row = conn.execute(f"""SELECT epoch, authority_install_id, granted_clock, length_s, boot_id, granted_at,
        host_sent_at, host_boot FROM {LEASES} WHERE room_id=?""", (room_id,)).fetchone()
    if row is None:
        return None
    epoch, authority, granted_clock, length, boot, granted_at, host_sent_at, host_boot = row
    remaining = lease_remaining(granted_clock=granted_clock, length_s=length, boot=boot, granted_at=granted_at,
                                now=now)
    if remaining <= 0:
        return None
    return {"epoch": int(epoch), "authority_install_id": authority, "remaining": remaining,
            "host_sent_at": None if host_sent_at is None else float(host_sent_at), "host_boot": host_boot}


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db
    return open_db(db_path, db_label="runs_idempotency.db", busy_timeout_ms=30_000, row_factory=None)


def _write(db_path, operation):
    """One IMMEDIATE transaction; storage failures are typed, decisions are not."""
    try:
        with closing(_connect(db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                initialize_fence_schema(conn)
                result = operation(conn)
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise
    except sqlite3.Error:
        raise RoomFenceError() from None


def fence_and_promise(db_path: Path | str, *, room_id: str, fence_epoch: int, promise_epoch: int,
                      candidate_install_id: str, now: float | None = None) -> dict[str, Any]:
    """Atomically fence ``fence_epoch`` and promise ``fence_epoch + 1`` to one candidate.

    Repeating the same request returns the existing promise. Another candidate for a promised
    epoch, or any epoch at or below the fence, is refused; a later epoch supersedes the earlier
    promise by fencing it. Nothing here starts, stops or moves work.
    """
    room_id = _exact(room_id, "room_id")
    candidate_install_id = _exact(candidate_install_id, "candidate_install_id")
    fence_epoch = _epoch(fence_epoch, "fence_epoch")
    if _epoch(promise_epoch, "promise_epoch", low=2) != fence_epoch + 1:
        raise ValueError("promise_epoch must follow fence_epoch")
    timestamp = clock(now)

    def operation(conn):
        row = conn.execute(_SELECT, (room_id,)).fetchone()
        state = _state(row)
        promise = state["promise"]
        if promise is not None and promise["epoch"] == promise_epoch:
            if promise["candidate_install_id"] != candidate_install_id:
                raise RoomAuthorityPromised()
            return {**state, "idempotent": True}
        if promise_epoch <= state["fenced_epoch"]:
            raise RoomAuthorityFenced()
        if promise is not None and promise_epoch < promise["epoch"]:
            raise RoomAuthorityPromised()
        lease = lease_locked(conn, room_id, now=timestamp)
        if lease is not None and lease["epoch"] <= fence_epoch:
            raise RoomLeaseActive()
        if row is None:
            if conn.execute(f"SELECT COUNT(*) FROM {FENCES}").fetchone()[0] >= MAX_FENCED_ROOMS:
                raise RoomFenceCapacity()
            conn.execute(f"""INSERT INTO {FENCES}(room_id, fenced_epoch, promise_epoch, candidate_install_id,
                issued_at, fenced_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (room_id, fence_epoch, promise_epoch, candidate_install_id, timestamp, timestamp, timestamp))
        else:
            conn.execute(f"""UPDATE {FENCES} SET fenced_epoch=MAX(fenced_epoch, ?), promise_epoch=?,
                candidate_install_id=?, issued_at=?, updated_at=? WHERE room_id=?""",
                (fence_epoch, promise_epoch, candidate_install_id, timestamp, timestamp, room_id))
        return {**_state(conn.execute(_SELECT, (room_id,)).fetchone()), "idempotent": False}

    return _write(db_path, operation)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == value and abs(value) < 1e300


def grant_lease(db_path: Path | str, *, room_id: str, epoch: int, authority_install_id: str, duration_s: float,
                until: float | None = None, host_sent_at: float | None = None, host_boot: str | None = None,
                now: float | None = None) -> dict[str, Any]:
    """Grant the room's host at ``epoch`` a lease of ``duration_s`` seconds, atomically with the fence.

    Only the authority of an epoch above the fenced one, never against a promise or a learned
    authority of this or a later epoch for another computer. ``until`` (wall clock) extends the
    lease over a planned restart, by at most five minutes. A renewal never shortens a lease.
    ``host_sent_at`` and ``host_boot`` are the host's own clock and boot when it asked (its request):
    the latest one asked is kept, for ``release_lease``.
    """
    from gateway import hosted_room_clock
    room_id = _exact(room_id, "room_id")
    authority_install_id = _exact(authority_install_id, "authority_install_id")
    epoch = _epoch(epoch, "epoch")
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not (
            0 < float(duration_s) <= MAX_LEASE_SECONDS):
        raise ValueError("duration_s must be a positive number of seconds, at most a minute")
    timestamp = clock(now)
    length = float(duration_s)
    if until is not None:
        if isinstance(until, bool) or not isinstance(until, (int, float)) or until != until:
            raise ValueError("until must be a unix time")
        length = max(length, min(float(until) - timestamp, MAX_RESTART_EXTENSION_SECONDS))
    sent_at = float(host_sent_at) if _finite(host_sent_at) else None
    sent_boot = host_boot if isinstance(host_boot, str) and host_boot and len(host_boot) <= 128 else None

    def operation(conn):
        state = _state(conn.execute(_SELECT, (room_id,)).fetchone())
        promise, authority = state["promise"], state["authority"]
        if epoch <= state["fenced_epoch"]:
            raise RoomAuthorityFenced()
        if promise is not None and (promise["epoch"] > epoch or (
                promise["epoch"] == epoch and promise["candidate_install_id"] != authority_install_id)):
            raise RoomAuthorityPromised()
        if authority is not None and (authority["epoch"] > epoch or (
                authority["epoch"] == epoch and authority["install_id"] != authority_install_id)):
            raise RoomAuthorityConflict()
        current = lease_locked(conn, room_id, now=timestamp)
        if current is not None and (current["epoch"] > epoch or (
                current["epoch"] == epoch and current["authority_install_id"] != authority_install_id)):
            raise RoomAuthorityConflict()
        same = current is not None and current["epoch"] == epoch
        kept = current["remaining"] if same else 0.0
        granted, asked = max(length, kept), sent_at
        if same and current["host_boot"] == sent_boot and current["host_sent_at"] is not None and (
                asked is None or current["host_sent_at"] > asked):
            asked = current["host_sent_at"]  # a request delayed in transit never moves this back
        conn.execute(f"""INSERT INTO {LEASES}(room_id, epoch, authority_install_id, granted_clock, length_s,
            boot_id, granted_at, host_sent_at, host_boot) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(room_id) DO UPDATE
            SET epoch=excluded.epoch, authority_install_id=excluded.authority_install_id,
            granted_clock=excluded.granted_clock, length_s=excluded.length_s, boot_id=excluded.boot_id,
            granted_at=excluded.granted_at, host_sent_at=excluded.host_sent_at, host_boot=excluded.host_boot""",
                     (room_id, epoch, authority_install_id, hosted_room_clock.now(), granted,
                      str(hosted_room_clock.boot_id() or ""), timestamp, asked, sent_boot))
        return {"room_id": room_id, "epoch": epoch, "authority_install_id": authority_install_id,
                "duration_s": granted}

    return _write(db_path, operation)


def release_lease(db_path: Path | str, *, room_id: str, epoch: int, authority_install_id: str,
                  signed_at: float | None = None, host_boot: str | None = None) -> bool:
    """The host itself gives its lease back: from now on this gateway may promise the next epoch.
    A lease of another epoch or host stays.

    With ``signed_at`` and ``host_boot`` (a handover statement the host signed, checked by the
    caller) only a lease the host last asked for before it signed is given back: once the host asks
    again it serves on that lease, and an older statement can't take it away. Without a recorded
    request, or across the host's reboot, the lease runs out by itself instead.
    """
    room_id = _exact(room_id, "room_id")
    authority_install_id = _exact(authority_install_id, "authority_install_id")
    epoch = _epoch(epoch, "epoch")
    bound = signed_at is not None or host_boot is not None
    if bound and not (_finite(signed_at) and isinstance(host_boot, str) and host_boot):
        return False

    def operation(conn):
        if bound:
            current = lease_locked(conn, room_id)
            if current is not None and (current["host_boot"] != host_boot or current["host_sent_at"] is None
                                        or current["host_sent_at"] >= float(signed_at)):
                return False
        return conn.execute(f"DELETE FROM {LEASES} WHERE room_id=? AND epoch=? AND authority_install_id=?",
                            (room_id, epoch, authority_install_id)).rowcount == 1

    return _write(db_path, operation)


def room_lease_state(db_path: Path | str, room_id: str) -> dict[str, Any] | None:
    """The lease this gateway granted the room's host while it runs, else ``None``."""
    room_id = _exact(room_id, "room_id")
    return _read(db_path, lambda conn: lease_locked(conn, room_id))


def fence_room(db_path: Path | str, *, room_id: str, fence_epoch: int, now: float | None = None) -> dict[str, Any]:
    """Fence epochs up to ``fence_epoch`` without promising anything, as an old authority stepping
    down does; a lower fence than the recorded one changes nothing."""
    room_id = _exact(room_id, "room_id")
    fence_epoch = _epoch(fence_epoch, "fence_epoch")
    timestamp = clock(now)

    def operation(conn):
        row = conn.execute(_SELECT, (room_id,)).fetchone()
        if row is None:
            if conn.execute(f"SELECT COUNT(*) FROM {FENCES}").fetchone()[0] >= MAX_FENCED_ROOMS:
                raise RoomFenceCapacity()
            conn.execute(f"INSERT INTO {FENCES}(room_id, fenced_epoch, fenced_at, updated_at) VALUES (?, ?, ?, ?)",
                         (room_id, fence_epoch, timestamp, timestamp))
        elif fence_epoch > int(row[0]):
            conn.execute(f"UPDATE {FENCES} SET fenced_epoch=?, updated_at=? WHERE room_id=?",
                         (fence_epoch, timestamp, room_id))
        return _state(conn.execute(_SELECT, (room_id,)).fetchone())

    return _write(db_path, operation)


def learn_authority(db_path: Path | str, *, room_id: str, epoch: int, install_id: str,
                    now: float | None = None) -> dict[str, Any]:
    """Record a verified authority (certificate or attestation checked by the caller) and fence every
    earlier epoch. A different authority for the same epoch is refused; an older one changes nothing."""
    room_id = _exact(room_id, "room_id")
    install_id = _exact(install_id, "install_id")
    epoch = _epoch(epoch, "epoch", low=2)
    timestamp = clock(now)

    def operation(conn):
        row = conn.execute(_SELECT, (room_id,)).fetchone()
        authority = _state(row)["authority"]
        if authority is not None and authority["epoch"] == epoch and authority["install_id"] != install_id:
            raise RoomAuthorityConflict()
        if row is None:
            if conn.execute(f"SELECT COUNT(*) FROM {FENCES}").fetchone()[0] >= MAX_FENCED_ROOMS:
                raise RoomFenceCapacity()
            conn.execute(f"""INSERT INTO {FENCES}(room_id, fenced_epoch, authority_epoch, authority_install_id,
                fenced_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (room_id, epoch - 1, epoch, install_id, timestamp, timestamp))
        elif authority is None or epoch > authority["epoch"]:
            conn.execute(f"""UPDATE {FENCES} SET fenced_epoch=MAX(fenced_epoch, ?), authority_epoch=?,
                authority_install_id=?, updated_at=? WHERE room_id=?""",
                (epoch - 1, epoch, install_id, timestamp, room_id))
        return _state(conn.execute(_SELECT, (room_id,)).fetchone())

    return _write(db_path, operation)


def _read(db_path, operation):
    try:
        with closing(_connect(db_path)) as conn:
            return operation(conn)
    except sqlite3.Error:
        raise RoomFenceError() from None


def room_fence_state(db_path: Path | str, room_id: str) -> dict[str, Any]:
    """``{fenced_epoch, promise, authority}``: 0 and ``None`` for a room this gateway never fenced."""
    room_id = _exact(room_id, "room_id")
    return _read(db_path, lambda conn: fence_state_locked(conn, room_id))


def successor_may_control(db_path: Path | str, room_id: str, candidate_install_id: str) -> bool:
    """Whether control of the room's existing runs here passed to ``candidate_install_id``."""
    room_id = _exact(room_id, "room_id")
    candidate_install_id = _exact(candidate_install_id, "candidate_install_id")
    return _read(db_path, lambda conn: successor_controls_locked(
        conn, room_id, candidate_install_id=candidate_install_id))
