"""The subagent-record verbs of the SQLite ``CoordinationRepository`` (``SubagentLedgerVerbs``): the
``resumable_subagents`` rows of the bound profile's ``state.db``, each verb one transaction on the store's writer with
the ledgers' lock wait, or one read.

The record columns are typed; ``record_json`` holds the run record the caller decodes as ``document``. A write whose
record was computed by the caller from a row it read applies only while the stored row still equals that row, compared
inside the write. The parent-supersession check walks the compression chain on the verb's own connection.

A verb for a profile whose store was removed creates nothing: the store's opener refuses the missing home and the verb
answers as a write that did nothing (a read answers no row). The worktree sweep's pinned read and the owed probe never
open the store for writing: each reads through its own read-only connection, and a missing store answers empty.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, Optional, TypeVar, cast

from hermes_cli.sqlite_safe_read import connect_tracked
from hermes_state_holders import read_only_db_uri
from persistence import DelegationId, JsonValue, SessionId
from persistence.authority.subagents import (
    AUTO_RESUME_CANCELLED, AUTO_RESUME_MAX_ATTEMPTS, GATEWAY_SHUTDOWN, SUBAGENT_EXPIRED_RETENTION_SECONDS, AutoRefusal,
    AutoResumeClaim, AutoStart, DelegationRoute, GaveUpCompletion, GiveUpOutcome, SubagentLimits, SubagentRecord,
    SubagentRecovery, SubagentState, SubagentWrite, parent_superseded_note,
)

from .._backend import translated
from ..conversation._store.compression import _CHAIN_STEP_SQL, _walk_compression_chain

__all__ = ["KEEP_WAITING_UNITS_SQL", "SubagentLedgerVerbs", "recover_subagents"]

T = TypeVar("T")

# The log records keep main's logger name: the delegation tool's.
logger = logging.getLogger("tools.delegate_tool")

_COLUMNS = ("subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at, record_json, created_at, updated_at, "
            "expires_at")
# A written run record never carries the columns or the marks of a claim no run has started yet.
_RUN_ONLY = ("state", "stop_cause", "auto_claim", "auto_cancel")
_CANCEL_STATES = (SubagentState.SUPERSEDED, SubagentState.STOPPED)

# A promised record no give-up failure was delivered for: its parent is owed exactly one report.
OWED_SQL = "json_extract(record_json, '$.auto_promised') AND json_extract(record_json, '$.auto_gave_up') IS NULL"
# The one marker of a debt the parent cleared: ``auto_promised`` false, set in the same write as the cancel.
CANCEL_DEBT = "'$.auto_promised', json('false')"
# The ledger prune keeps the unit row a waiting child reports on: a silent unit is 'delivered', the first kind the
# count cap deletes, and the child's give-up failure is routed through that row.
KEEP_WAITING_UNITS_SQL = (
    " AND delegation_id NOT IN (SELECT json_extract(record_json, '$.unit_id') FROM resumable_subagents WHERE "
    f"state IN ('resumable','expired') AND {OWED_SQL} AND json_extract(record_json, '$.unit_id') IS NOT NULL)")


def _record(row) -> SubagentRecord:
    sid, state, pinned, cause, pid, started, raw, created, updated, expires = row
    return SubagentRecord(subagent_id=sid, state=SubagentState(state), pinned=bool(pinned), stop_cause=cause,
                          owner_pid=pid, owner_started_at=started, document=json.loads(raw or "{}"),
                          created_at=created, updated_at=updated, expires_at=expires)


def _row(conn, sid: str) -> Optional[SubagentRecord]:
    got = conn.execute(f"SELECT {_COLUMNS} FROM resumable_subagents WHERE subagent_id=?", (sid,)).fetchone()
    return None if got is None else _record(tuple(got))


def _rows(conn, sql: str, params: Sequence[Any] = ()) -> tuple[SubagentRecord, ...]:
    return tuple(_record(tuple(row)) for row in conn.execute(sql, params).fetchall())


def _upsert(conn, sid: str, write: SubagentWrite, now: float) -> None:
    record = {k: v for k, v in write.document.items() if k not in _RUN_ONLY}
    conn.execute(
        """INSERT INTO resumable_subagents (subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at,
               record_json, created_at, updated_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(subagent_id) DO UPDATE SET state=excluded.state, pinned=excluded.pinned,
               stop_cause=excluded.stop_cause, owner_pid=excluded.owner_pid,
               owner_started_at=excluded.owner_started_at, record_json=excluded.record_json,
               updated_at=excluded.updated_at, expires_at=excluded.expires_at""",
        (sid, SubagentState(write.state).value, int(write.pinned), write.stop_cause, write.owner_pid,
         write.owner_started_at, json.dumps(record, ensure_ascii=False, default=str), now, now, write.expires_at))


def _apply_limits(conn, limits: SubagentLimits, now: float) -> None:
    """TTL un-pins and retires a record; beyond the cap the oldest pins drop (the row stays continuable)."""
    conn.execute("UPDATE resumable_subagents SET pinned=0, state='expired' WHERE expires_at IS NOT NULL "
                 "AND expires_at <= ? AND state IN ('resumable','completed','stopped','superseded')", (now,))
    conn.execute(f"DELETE FROM resumable_subagents WHERE state='expired' AND updated_at < ? AND NOT COALESCE(({OWED_SQL}), 0)",
                 (now - SUBAGENT_EXPIRED_RETENTION_SECONDS - limits.ttl_hours * 3600,))
    conn.execute("""UPDATE resumable_subagents SET pinned=0 WHERE pinned=1 AND subagent_id NOT IN (
                        SELECT subagent_id FROM resumable_subagents WHERE pinned=1
                        ORDER BY updated_at DESC LIMIT ?)""", (max(limits.max_resumable, 0),))


def _note(conn, sid: str, *, state: Optional[SubagentState], note: str, now: float,
          from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False,
          unstarted_claim: bool = False) -> bool:
    """Record why the scheduler will not (or no longer) run this record; *state* moves it when given (a cancel
    state also clears its debt). *unstarted_claim*: also a record an automatic continue claimed whose run has not
    started (``$.auto_claim``)."""
    marks = ",".join("?" * len(from_states))
    debt = f", {CANCEL_DEBT}" if state in _CANCEL_STATES else ""
    sets = f"record_json=json_remove(json_set(record_json, '$.auto_note', ?{debt}), '$.auto_claim'), updated_at=?"
    if state:
        sets += ", state=?" + (", pinned=0" if unpin else "")
    claimed = (" OR (state='claimed' AND json_extract(record_json, '$.auto_claim') IS NOT NULL)"
               if unstarted_claim else "")
    params = [note, now, *([SubagentState(state).value] if state else []), sid,
              *(SubagentState(s).value for s in from_states)]
    return conn.execute(f"UPDATE resumable_subagents SET {sets} WHERE subagent_id=? AND (state IN ({marks}){claimed})",
                        params).rowcount == 1


def _cancel_expired(conn, sid: str, note: str, now: float) -> bool:
    """A cancel that lands after the TTL moved an owed record to ``expired`` but before a tick reported it."""
    return conn.execute(f"UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_note', ?, "
                        f"{CANCEL_DEBT}), updated_at=? WHERE subagent_id=? AND state='expired' AND {OWED_SQL}",
                        (note, now, sid)).rowcount == 1


def _cancel_manual_claim(conn, sid: str, *, state: SubagentState, note: str, now: float) -> bool:
    """A cancel that lands while the parent's own continue holds the row (``claimed``, no ``$.auto_claim``) before it
    ran: the decision rides on the row (``$.auto_cancel``) for the hand-back."""
    return conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_cancel', "
                        "json_object('state', ?, 'note', ?)), updated_at=? WHERE subagent_id=? AND state='claimed' "
                        "AND json_extract(record_json, '$.auto_claim') IS NULL",
                        (SubagentState(state).value, note, now, sid)).rowcount == 1


def _superseded_reason(conn, parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants its child resumed, from the live tip of its compression chain, read on *conn*
    (inside a write transaction nothing the parent does lands between this answer and the write acting on it)."""
    def step(current):
        row = conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()
        return {"id": row[0]} if row else None
    chain = _walk_compression_chain(parent_sid, step) or [parent_sid]
    tip = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id=?", (chain[-1],)).fetchone()
    if tip is None:
        return parent_superseded_note(None, tip_exists=False)
    return parent_superseded_note(tip[1] if tip[0] is not None else None, tip_exists=True)


def _unstarted_auto_claim(record: SubagentRecord) -> bool:
    return record.state == SubagentState.CLAIMED and bool(record.document.get("auto_claim"))


def _attempts(record: SubagentRecord) -> int:
    return int(cast(Any, record.document.get("auto_attempts")) or 0)


def _refund(conn, sid: str) -> None:
    conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_attempts', "
                 "MAX(COALESCE(json_extract(record_json, '$.auto_attempts'), 1) - 1, 0)) WHERE subagent_id=?", (sid,))


def _insert_completion(conn, completion: GaveUpCompletion, now: float) -> None:
    """The give-up's failed completion on its own ledger row, pending, never replacing a row."""
    evt, result = completion.event, completion.result
    task = {k: evt.get(k) for k in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")}
    conn.execute(
        "INSERT OR IGNORE INTO async_delegations (delegation_id, origin_session, origin_ui_session_id, "
        "parent_session_id, state, dispatched_at, completed_at, updated_at, event_json, result_json, delivery_state, "
        "delivery_attempts, owner_pid, owner_started_at, task_json, origin_session_id) "
        "VALUES (?, ?, ?, ?, 'failed', ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)",
        (evt["delegation_id"], evt.get("session_key") or "", evt.get("origin_ui_session_id") or "",
         evt.get("parent_session_id"), evt.get("dispatched_at") or now, evt.get("completed_at") or now, now,
         json.dumps(evt), json.dumps(result), completion.owner_pid, completion.owner_started_at, json.dumps(task),
         evt.get("origin_session_id") or ""))


def recover_subagents(conn, recoveries: Sequence[SubagentRecovery], limits: Optional[SubagentLimits],
                      now: float) -> None:
    """Boot recovery's subagent half, on the recovery's own transaction: each record still held by the same dead owner
    (``running`` or ``claimed``, read inside the write lock) is written as given, the limits applied after each."""
    for recovery in recoveries:
        got = conn.execute("SELECT state, owner_pid, owner_started_at FROM resumable_subagents WHERE subagent_id=?",
                           (recovery.subagent_id,)).fetchone()
        if got is None or got[0] not in ("running", "claimed") or (got[1], got[2]) != (recovery.owner_pid,
                                                                                       recovery.owner_started_at):
            continue
        _upsert(conn, recovery.subagent_id, recovery.write, now)
        if limits is not None:
            _apply_limits(conn, limits, now)


class _RouteMoved(Exception):
    """Rolls a give-up back when its unit row changed since the caller built the completion."""


def _at(now: Optional[float]) -> float:
    return time.time() if now is None else now


class SubagentLedgerVerbs:
    """The subagent-record verbs; the host class provides ``_store``, ``_ledger_write`` and ``_ledger_read``."""

    _store: Any

    def _ledger_write(self, fn: Callable[[Any], T]) -> T:
        raise NotImplementedError

    def _ledger_read(self, read: Callable[[Any], T]) -> T:
        raise NotImplementedError

    def _subagent_write(self, fn: Callable[[Any], T], gone: Any = None) -> T:
        """``fn(conn)`` in one ledger write; ``gone`` when the profile's store was removed (nothing is created)."""
        try:
            with translated():
                return self._ledger_write(fn)
        except FileNotFoundError:
            logger.debug("resumable subagent record skipped: %s is gone", self._store.store_path.parent)
            return gone

    def _subagent_read(self, read: Callable[[Any], T], gone: Any = None) -> T:
        try:
            with translated():
                return self._ledger_read(read)
        except FileNotFoundError:
            return gone

    # ── reads ────────────────────────────────────────────────────────────

    def get_subagent(self, *, subagent_id: str) -> Optional[SubagentRecord]:
        return self._subagent_read(lambda conn: _row(conn, subagent_id))

    def list_inflight_subagents(self) -> tuple[SubagentRecord, ...]:
        return self._subagent_read(lambda conn: _rows(
            conn, f"SELECT {_COLUMNS} FROM resumable_subagents WHERE state IN ('running','claimed')"), ())

    def list_continuable_subagents(self, *, unit_id: str) -> tuple[str, ...]:
        return self._subagent_read(lambda conn: tuple(sid for (sid,) in conn.execute(
            "SELECT subagent_id FROM resumable_subagents WHERE state='resumable' AND COALESCE("
            "json_extract(record_json, '$.unit_id'), json_extract(record_json, '$.delegation_id'))=? "
            "ORDER BY created_at", (unit_id,)).fetchall()), ())

    def list_pinned_subagents(self, *, max_resumable: int, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        self._store.require_open()
        db = self._store.store_path
        try:
            conn = connect_tracked(read_only_db_uri(db), tracking_path=db, uri=True, timeout=2.0)
        except sqlite3.Error:
            return ()
        try:
            return _rows(conn, f"SELECT {_COLUMNS} FROM resumable_subagents "
                               "WHERE pinned=1 AND state IN ('resumable','superseded','running','claimed') "
                               "AND (expires_at IS NULL OR expires_at > ?) ORDER BY updated_at DESC LIMIT ?",
                         (_at(now), max(max_resumable, 0)))
        except sqlite3.Error:
            return ()  # no table yet: nothing pinned
        finally:
            conn.close()

    def list_auto_resume_candidates(self, *, owed_only: bool = False) -> tuple[SubagentRecord, ...]:
        return self._subagent_read(lambda conn: _rows(
            conn, f"SELECT {_COLUMNS} FROM resumable_subagents WHERE "
                  f"(state='resumable' OR (state='expired' AND {OWED_SQL}))" + (f" AND {OWED_SQL}" if owed_only else "")
                  + " ORDER BY updated_at"), ())

    def has_owed_subagents(self) -> bool:
        self._store.require_open()
        db = self._store.store_path
        if not db.exists():
            return False
        try:
            conn = connect_tracked(read_only_db_uri(db), uri=True, timeout=5.0, tracking_path=db)
            try:
                return conn.execute(f"SELECT 1 FROM resumable_subagents WHERE state IN ('resumable','expired') "
                                    f"AND {OWED_SQL} LIMIT 1").fetchone() is not None
            finally:
                conn.close()
        except sqlite3.Error:  # unreadable here: the caller's writer pass decides
            return True

    def list_waiting_subagents(self, *, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        return self._subagent_read(lambda conn: _rows(
            conn, f"SELECT {_COLUMNS} FROM resumable_subagents WHERE state='resumable' AND {OWED_SQL} "
                  "AND (expires_at IS NULL OR expires_at > ?)", (_at(now),)), ())

    def subagent_parent_superseded(self, *, parent_session_id: SessionId) -> Optional[str]:
        return self._subagent_read(lambda conn: _superseded_reason(conn, str(parent_session_id)))

    def get_delegation_route(self, *, delegation_id: DelegationId) -> Optional[DelegationRoute]:
        def read(conn):
            return conn.execute("SELECT origin_session, origin_ui_session_id, parent_session_id, origin_session_id, "
                                "task_json, result_json, dispatched_at, updated_at FROM async_delegations "
                                "WHERE delegation_id=?", (str(delegation_id),)).fetchone()
        got = self._subagent_read(read)
        if got is None:
            return None
        origin, origin_ui, parent, origin_sid, task_json, result_json, dispatched_at, updated_at = tuple(got)
        return DelegationRoute(delegation_id=DelegationId(str(delegation_id)), origin_session=origin,
                               origin_ui_session_id=origin_ui, parent_session_id=parent,
                               origin_session_id=origin_sid or "", task=json.loads(task_json) if task_json else None,
                               result=json.loads(result_json) if result_json else None,
                               dispatched_at=dispatched_at, updated_at=updated_at)

    # ── run records ──────────────────────────────────────────────────────

    def put_subagent(self, *, subagent_id: str, write: SubagentWrite, expected: Optional[SubagentRecord],
                     limits: Optional[SubagentLimits] = None, now: Optional[float] = None) -> bool:
        at = _at(now)
        def _do(conn):
            if _row(conn, subagent_id) != expected:
                return False
            _upsert(conn, subagent_id, write, at)
            if limits is not None:
                _apply_limits(conn, limits, at)
            return True
        return bool(self._subagent_write(_do, False))

    def start_auto_resumed_run(self, *, subagent_id: str, write: SubagentWrite, expected: SubagentRecord,
                               owner_pid: int, now: Optional[float] = None) -> AutoStart:
        """The run-start write of a run the scheduler started gives the parent its last word first: the parent's
        decision stays on the row and the attempt is refunded."""
        at = _at(now)
        def _do(conn):
            row = _row(conn, subagent_id)
            if row is not None and _unstarted_auto_claim(row) and row.owner_pid == owner_pid:
                reason = _superseded_reason(conn, str(row.document["auto_claim"]))
                if not reason:
                    if row != expected:
                        return AutoStart(written=False, cancelled=None)
                    _upsert(conn, subagent_id, write, at)
                    return AutoStart(written=True, cancelled=None)
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=at, from_states=(),
                      unstarted_claim=True)
            elif row is not None and row.state in (SubagentState.STOPPED, SubagentState.SUPERSEDED):
                reason = row.document.get("auto_note") or "its parent stopped it"
            else:
                return AutoStart(written=False, cancelled=(
                    f"{AUTO_RESUME_CANCELLED}: the record moved ({row.state.value if row else 'gone'})"))
            _refund(conn, subagent_id)
            return AutoStart(written=False, cancelled=f"{AUTO_RESUME_CANCELLED}: {reason}")
        return self._subagent_write(_do, AutoStart(written=False, cancelled=None))

    def checkpoint_subagent(self, *, subagent_id: str, cwd: str, repo_baselines: Mapping[str, JsonValue],
                            now: Optional[float] = None) -> bool:
        at = _at(now)
        def _do(conn):
            return conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.cwd', ?, "
                                "'$.repo_baselines', json(?)), updated_at=? WHERE subagent_id=? AND state='running'",
                                (cwd, json.dumps(repo_baselines), at, subagent_id)).rowcount == 1
        return bool(self._subagent_write(_do, False))

    def mark_subagents_gateway_shutdown(self, *, owner_pid: int) -> int:
        return int(self._subagent_write(lambda conn: conn.execute(
            "UPDATE resumable_subagents SET stop_cause=? WHERE state='running' AND owner_pid=?",
            (GATEWAY_SHUTDOWN, owner_pid)).rowcount, 0) or 0)

    def apply_subagent_limits(self, *, limits: SubagentLimits, subagent_id: Optional[str] = None,
                              now: Optional[float] = None) -> Optional[SubagentRecord]:
        at = _at(now)
        def _do(conn):
            _apply_limits(conn, limits, at)
            return None if subagent_id is None else _row(conn, subagent_id)
        return self._subagent_write(_do)

    # ── continue ─────────────────────────────────────────────────────────

    def claim_subagent(self, *, subagent_id: str, expected_state: SubagentState, owner_pid: int,
                       owner_started_at: Optional[int], now: Optional[float] = None) -> bool:
        at = _at(now)
        return self._subagent_write(lambda conn: conn.execute(
            f"UPDATE resumable_subagents SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=?, "
            f"record_json=json_set(record_json, {CANCEL_DEBT}) WHERE subagent_id=? AND state=?",
            (owner_pid, owner_started_at, at, subagent_id, SubagentState(expected_state).value)).rowcount, 0) == 1

    def hand_back_subagent(self, *, subagent_id: str, state: SubagentState, promised: Optional[bool] = None,
                           owner_pid: Optional[int] = None) -> None:
        """A stop that landed during the claim (``$.auto_cancel``) wins over handing a promise back."""
        def _do(conn):
            target = SubagentState(state).value
            hand_promised = promised
            mine, where = (" AND owner_pid=?", (subagent_id, owner_pid)) if owner_pid is not None else ("", (subagent_id,))
            got = conn.execute("SELECT json_extract(record_json, '$.auto_cancel.state'), json_extract(record_json, "
                               f"'$.auto_cancel.note') FROM resumable_subagents WHERE subagent_id=? AND state='claimed'{mine}",
                               where).fetchone()
            if got is None:
                return
            record_json, params, unpin = "record_json", [], ""
            if got[0]:  # the parent cancelled it during the claim: no automatic resume, no report
                hand_promised = False
                if target == "resumable":
                    target, record_json, params = got[0], "json_set(record_json, '$.auto_note', ?)", [got[1]]
                    unpin = ", pinned=0" if target == "stopped" else ""
            if hand_promised is not None:
                record_json, params = (f"json_set({record_json}, '$.auto_promised', json(?))",
                                       [*params, json.dumps(bool(hand_promised))])
            conn.execute(f"UPDATE resumable_subagents SET record_json=json_remove({record_json}, '$.auto_claim', "
                         f"'$.auto_cancel'), state=?{unpin} WHERE subagent_id=? AND state='claimed'{mine}",
                         (*params, target, *where))
        self._subagent_write(_do)

    # ── automatic resume ─────────────────────────────────────────────────

    def note_subagent(self, *, subagent_id: str, note: str, state: Optional[SubagentState] = None,
                      from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False,
                      unstarted_claim: bool = False, now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(lambda conn: _note(
            conn, subagent_id, state=state, note=note, now=at, from_states=from_states, unpin=unpin,
            unstarted_claim=unstarted_claim), False))

    def cancel_expired_subagent(self, *, subagent_id: str, note: str, now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(lambda conn: _cancel_expired(conn, subagent_id, note, at), False))

    def cancel_manual_subagent_claim(self, *, subagent_id: str,
                                     state: Literal[SubagentState.STOPPED, SubagentState.SUPERSEDED], note: str,
                                     now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(
            lambda conn: _cancel_manual_claim(conn, subagent_id, state=state, note=note, now=at), False))

    def claim_auto_resume(self, *, subagent_id: str, expected_attempts: int, parent_session_id: SessionId,
                          owner_pid: int, owner_started_at: Optional[int], now: Optional[float] = None) -> AutoResumeClaim:
        """ONE write re-reads the row, re-runs supersession on that connection and, only while it is still
        ``resumable``, unsuperseded, uncancelled and under the cap, claims it and spends the attempt before it runs."""
        at = _at(now)
        parent_sid = str(parent_session_id)
        def _do(conn):
            row = _row(conn, subagent_id)
            if (row is None or row.state != SubagentState.RESUMABLE or row.document.get("auto_off")
                    or not row.document.get("auto_promised")
                    or _attempts(row) != expected_attempts
                    or expected_attempts >= AUTO_RESUME_MAX_ATTEMPTS):
                return AutoResumeClaim(record=None, superseded=None)
            if reason := _superseded_reason(conn, parent_sid):
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=at)
                return AutoResumeClaim(record=None, superseded=reason)
            conn.execute(
                "UPDATE resumable_subagents SET state='claimed', owner_pid=?, owner_started_at=?, updated_at=?, "
                "record_json=json_set(record_json, '$.auto_attempts', ?, '$.auto_last_at', ?, '$.auto_claim', ?) "
                "WHERE subagent_id=? AND state='resumable'",
                (owner_pid, owner_started_at, at, expected_attempts + 1, at, parent_sid, subagent_id))
            return AutoResumeClaim(record=_row(conn, subagent_id), superseded=None)
        return self._subagent_write(_do, AutoResumeClaim(record=None, superseded=None))

    def confirm_auto_start(self, *, subagent_id: str) -> Optional[str]:
        def _do(conn):
            row = _row(conn, subagent_id)
            if row is None or not _unstarted_auto_claim(row):
                return AUTO_RESUME_CANCELLED
            if reason := _superseded_reason(conn, str(row.document["auto_claim"])):
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=time.time(), from_states=(),
                      unstarted_claim=True)
                return f"{AUTO_RESUME_CANCELLED}: {reason}"
            return None
        return self._subagent_write(_do, AUTO_RESUME_CANCELLED)

    def settle_auto_refusal(self, *, subagent_id: str, kind: AutoRefusal, attempts: int, note: str,
                            owner_pid: int) -> None:
        if kind == AutoRefusal.CANCELLED:  # the parent's decision already wrote the row: only the attempt is refunded
            sql, params = "json_set(record_json, '$.auto_attempts', ?)", (attempts,)
        elif kind == AutoRefusal.TRANSIENT:
            sql, params = "json_set(record_json, '$.auto_attempts', ?, '$.auto_note', ?)", (attempts, note)
        else:
            sql, params = "json_set(record_json, '$.auto_off', ?, '$.auto_note', ?)", (note, note)
        self._subagent_write(lambda conn: conn.execute(
            f"UPDATE resumable_subagents SET record_json=json_remove({sql}, '$.auto_claim'), "
            "state=CASE WHEN state='claimed' AND owner_pid=? THEN 'resumable' ELSE state END WHERE subagent_id=?",
            (*params, owner_pid, subagent_id)))

    def give_up_auto_resume(self, *, subagent_id: str, reason: str, completion: Optional[GaveUpCompletion],
                            now: Optional[float] = None) -> GiveUpOutcome:
        """ONE write marks the record and records its one failed completion as its own pending ledger row, so a crash
        on either side of the commit leaves the report replayable and never doubled."""
        at = _at(now)
        def _do(conn):
            if conn.execute(
                    "UPDATE resumable_subagents SET record_json=json_set(record_json, '$.auto_gave_up', ?, '$.auto_off', ?, "
                    "'$.auto_note', ?), updated_at=? WHERE subagent_id=? AND state IN ('resumable','expired') "
                    f"AND {OWED_SQL}",  # a cancel since the caller read it cleared the debt: nothing to report
                    (reason, reason, reason, at, subagent_id)).rowcount != 1:
                return GiveUpOutcome.NOT_OWED
            if completion is None:
                return GiveUpOutcome.NO_ROUTE
            got = conn.execute("SELECT updated_at FROM async_delegations WHERE delegation_id=?",
                               (str(completion.unit_id),)).fetchone()
            if got is None:  # pruning keeps a waiting child's unit row: only a nested or joined child has none
                return GiveUpOutcome.NO_ROUTE
            if got[0] != completion.unit_updated_at:
                raise _RouteMoved
            _insert_completion(conn, completion, at)
            return GiveUpOutcome.RECORDED
        try:
            return self._subagent_write(_do, GiveUpOutcome.NOT_OWED)
        except _RouteMoved:
            return GiveUpOutcome.ROUTE_MOVED

    def supersede_subagents_for_session(self, *, note: str, session_key: str = "", origin_ui_session_id: str = "",
                                        parent_session_id: str = "", promised_only: bool = False) -> int:
        selectors = [(col, val) for col, val in (("origin_session", session_key), ("origin_ui_session_id",
                     origin_ui_session_id), ("parent_session_id", parent_session_id)) if val]
        if not selectors:
            return 0
        owed_only = f" AND {OWED_SQL}" if promised_only else ""

        def _do(conn):
            now = time.time()
            where = " OR ".join(f"{col}=?" for col, _ in selectors)
            units = {row[0] for row in conn.execute(f"SELECT delegation_id FROM async_delegations WHERE {where}",
                                                    [val for _, val in selectors])}
            rows = conn.execute("SELECT subagent_id, json_extract(record_json, '$.unit_id') FROM resumable_subagents "
                                f"WHERE state='claimed' OR (state='resumable'{owed_only}) "
                                f"OR (state='expired' AND {OWED_SQL})").fetchall()
            return sum(_note(conn, sid, state=SubagentState.SUPERSEDED, note=note, now=now, unstarted_claim=True)
                       or _cancel_expired(conn, sid, note, now)
                       or _cancel_manual_claim(conn, sid, state=SubagentState.SUPERSEDED, note=note, now=now)
                       for sid, unit in rows if unit in units)
        return int(self._subagent_write(_do, 0) or 0)

    def dismiss_subagent(self, *, expected: SubagentRecord, note: str,
                         now: Optional[float] = None) -> Optional[Literal["dismissed", "recorded"]]:
        at = _at(now)
        sid = expected.subagent_id
        def _do(conn):
            record = _row(conn, sid)
            if record is None or record != expected:
                return None
            if record.state == SubagentState.CLAIMED and not record.document.get("auto_claim"):
                # The parent's continue goes on; if it is refused the hand-back ends the row ``stopped``.
                return "recorded" if _cancel_manual_claim(conn, sid, state=SubagentState.STOPPED, note=note, now=at) else None
            done = _note(conn, sid, state=SubagentState.STOPPED, note=note, now=at,
                         from_states=(SubagentState.RESUMABLE, SubagentState.SUPERSEDED), unpin=True,
                         unstarted_claim=True) or _cancel_expired(conn, sid, note, at)
            return "dismissed" if done else None
        return self._subagent_write(_do)
