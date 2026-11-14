"""The subagent-record verbs of the reference coordination repository.

Rows live in the scope store's ``resumable_subagents`` table, keyed by subagent id: the columns of :class:`persistence.authority.subagents.SubagentRecord` with ``document`` the decoded run record, round-tripped through JSON as the store encodes it (``default=str``). The supersession walk is the lineage reference's compression chain step from the parent to its tip. Every verb runs under the backend lock, so each is one atomic step; the composites check everything they refuse on before they write.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

from persistence import DelegationId, JsonValue, SessionId
from persistence.authority.subagents import (
    AUTO_RESUME_CANCELLED,
    AUTO_RESUME_MAX_ATTEMPTS,
    GATEWAY_SHUTDOWN,
    SUBAGENT_EXPIRED_RETENTION_SECONDS,
    AutoRefusal,
    AutoResumeClaim,
    AutoStart,
    DelegationRoute,
    GaveUpCompletion,
    GiveUpOutcome,
    SubagentLimits,
    SubagentRecord,
    SubagentRecovery,
    SubagentState,
    SubagentWrite,
    parent_superseded_note,
)

from ..conversation.lineage import _CHAIN_CAP

__all__ = ["SubagentRecords"]

S = SubagentState
_TABLE = "resumable_subagents"
_RUN_ONLY = ("state", "stop_cause", "auto_claim", "auto_cancel")
_EXPIRABLE = (S.RESUMABLE, S.COMPLETED, S.STOPPED, S.SUPERSEDED)
_PINNED_LIVE = (S.RESUMABLE, S.SUPERSEDED, S.RUNNING, S.CLAIMED)
_CANCEL_STATES = (S.SUPERSEDED, S.STOPPED)


def _stored(document: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(json.dumps(dict(document), ensure_ascii=False, default=str))
    return value


def _owed(row: Mapping[str, Any]) -> bool:
    return bool(row["document"].get("auto_promised")) and row["document"].get("auto_gave_up") is None


def _record(row: Mapping[str, Any]) -> SubagentRecord:
    return SubagentRecord(subagent_id=row["subagent_id"], state=S(row["state"]), pinned=bool(row["pinned"]), stop_cause=row["stop_cause"],
                          owner_pid=row["owner_pid"], owner_started_at=row["owner_started_at"], document=copy.deepcopy(row["document"]),
                          created_at=row["created_at"], updated_at=row["updated_at"], expires_at=row["expires_at"])


def _unstarted_auto_claim(row: Mapping[str, Any]) -> bool:
    return row["state"] == S.CLAIMED and row["document"].get("auto_claim") is not None


def _refund(row: dict[str, Any]) -> None:
    attempts = row["document"].get("auto_attempts")
    row["document"]["auto_attempts"] = max((1 if attempts is None else int(attempts)) - 1, 0)


class SubagentRecords:
    """The subagent-record verbs; the host class provides ``_table`` and ``_backend``."""

    _backend: Any

    def _table(self, name: str) -> dict[Any, Any]:
        raise NotImplementedError

    def _subagents(self) -> dict[str, dict[str, Any]]:
        return self._table(_TABLE)

    def _row(self, subagent_id: str) -> Optional[dict[str, Any]]:
        return self._subagents().get(subagent_id)

    # ── reads ────────────────────────────────────────────────────────────

    def get_subagent(self, *, subagent_id: str) -> Optional[SubagentRecord]:
        row = self._row(subagent_id)
        return None if row is None else _record(row)

    def _listed(self, keep: Any, *, order: str = "updated_at", reverse: bool = False) -> list[dict[str, Any]]:
        rows = [row for row in self._subagents().values() if keep(row)]
        rows.sort(key=lambda row: row[order], reverse=reverse)
        return rows

    def list_inflight_subagents(self) -> tuple[SubagentRecord, ...]:
        return tuple(_record(row) for row in self._listed(lambda row: row["state"] in (S.RUNNING, S.CLAIMED)))

    def list_continuable_subagents(self, *, unit_id: str) -> tuple[str, ...]:
        def keep(row: Mapping[str, Any]) -> bool:
            document = row["document"]
            unit = document.get("unit_id") if document.get("unit_id") is not None else document.get("delegation_id")
            return bool(row["state"] == S.RESUMABLE and unit == unit_id)
        return tuple(row["subagent_id"] for row in self._listed(keep, order="created_at"))

    def list_pinned_subagents(self, *, max_resumable: int, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        at = time.time() if now is None else now
        rows = self._listed(lambda row: row["pinned"] and row["state"] in _PINNED_LIVE and (row["expires_at"] is None or row["expires_at"] > at), reverse=True)
        return tuple(_record(row) for row in rows[:max(max_resumable, 0)])

    def list_auto_resume_candidates(self, *, owed_only: bool = False) -> tuple[SubagentRecord, ...]:
        def keep(row: Mapping[str, Any]) -> bool:
            listed = row["state"] == S.RESUMABLE or (row["state"] == S.EXPIRED and _owed(row))
            return listed and (not owed_only or _owed(row))
        return tuple(_record(row) for row in self._listed(keep))

    def has_owed_subagents(self) -> bool:
        return any(row["state"] in (S.RESUMABLE, S.EXPIRED) and _owed(row) for row in self._subagents().values())

    def list_waiting_subagents(self, *, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        at = time.time() if now is None else now
        return tuple(_record(row) for row in self._listed(
            lambda row: row["state"] == S.RESUMABLE and _owed(row) and (row["expires_at"] is None or row["expires_at"] > at)))

    def subagent_parent_superseded(self, *, parent_session_id: SessionId) -> Optional[str]:
        sessions: dict[str, dict[str, Any]] = self._table("sessions")
        chain = [str(parent_session_id)]
        for _ in range(_CHAIN_CAP):
            child = self._backend.lineage._chain_step(self._backend.scope, sessions, chain[-1])
            if not child or child in chain:
                break
            chain.append(child)
        tip = sessions.get(chain[-1])
        if tip is None:
            return parent_superseded_note(None, tip_exists=False)
        return parent_superseded_note(tip["end_reason"] if tip["ended_at"] is not None else None, tip_exists=True)

    def get_delegation_route(self, *, delegation_id: DelegationId) -> Optional[DelegationRoute]:
        row = self._table("async_delegations").get(str(delegation_id))
        if row is None:
            return None
        return DelegationRoute(delegation_id=DelegationId(row["delegation_id"]), origin_session=row["origin_session"], origin_ui_session_id=row["origin_ui_session_id"],
                               parent_session_id=row["parent_session_id"], origin_session_id=row["origin_session_id"] or "", task=copy.deepcopy(row["task"]),
                               result=copy.deepcopy(row["result"]), dispatched_at=row["dispatched_at"], updated_at=row["updated_at"])

    # ── run records ──────────────────────────────────────────────────────

    def _upsert(self, subagent_id: str, write: SubagentWrite, now: float) -> None:
        document = {k: v for k, v in _stored(write.document).items() if k not in _RUN_ONLY}
        previous = self._row(subagent_id)
        self._subagents()[subagent_id] = {
            "subagent_id": subagent_id, "state": S(write.state).value, "pinned": bool(write.pinned), "stop_cause": write.stop_cause,
            "owner_pid": write.owner_pid, "owner_started_at": write.owner_started_at, "document": document,
            "created_at": previous["created_at"] if previous is not None else now, "updated_at": now, "expires_at": write.expires_at}

    def _unchanged(self, subagent_id: str, expected: Optional[SubagentRecord]) -> bool:
        current = self._row(subagent_id)
        return (current is None) if expected is None else (current is not None and _record(current) == expected)

    def put_subagent(self, *, subagent_id: str, write: SubagentWrite, expected: Optional[SubagentRecord], limits: Optional[SubagentLimits] = None, now: Optional[float] = None) -> bool:
        at = time.time() if now is None else now
        _stored(write.document)
        if not self._unchanged(subagent_id, expected):
            return False
        self._upsert(subagent_id, write, at)
        if limits is not None:
            self._apply_limits(limits, at)
        return True

    def start_auto_resumed_run(self, *, subagent_id: str, write: SubagentWrite, expected: SubagentRecord, owner_pid: int, now: Optional[float] = None) -> AutoStart:
        at = time.time() if now is None else now
        _stored(write.document)
        row = self._row(subagent_id)
        state = row["state"] if row is not None else None
        if row is not None and _unstarted_auto_claim(row) and row["owner_pid"] == owner_pid:
            reason = self.subagent_parent_superseded(parent_session_id=SessionId(str(row["document"]["auto_claim"])))
            if not reason:
                if not self._unchanged(subagent_id, expected):
                    return AutoStart(written=False, cancelled=None)
                self._upsert(subagent_id, write, at)
                return AutoStart(written=True, cancelled=None)
            self._note(row, note=reason, state=S.SUPERSEDED, unpin=False, now=at)
        elif row is not None and state in (S.STOPPED, S.SUPERSEDED):
            reason = row["document"].get("auto_note") or "its parent stopped it"
        else:
            return AutoStart(written=False, cancelled=f"{AUTO_RESUME_CANCELLED}: the record moved ({state or 'gone'})")
        _refund(row)
        return AutoStart(written=False, cancelled=f"{AUTO_RESUME_CANCELLED}: {reason}")

    def checkpoint_subagent(self, *, subagent_id: str, cwd: str, repo_baselines: Mapping[str, JsonValue], now: Optional[float] = None) -> bool:
        row = self._row(subagent_id)
        if row is None or row["state"] != S.RUNNING:
            return False
        row["document"].update(cwd=cwd, repo_baselines=_stored(repo_baselines))
        row["updated_at"] = time.time() if now is None else now
        return True

    def mark_subagents_gateway_shutdown(self, *, owner_pid: int) -> int:
        rows = [row for row in self._subagents().values() if row["state"] == S.RUNNING and row["owner_pid"] == owner_pid]
        for row in rows:
            row["stop_cause"] = GATEWAY_SHUTDOWN
        return len(rows)

    def _apply_limits(self, limits: SubagentLimits, now: float) -> None:
        table = self._subagents()
        for row in table.values():
            if row["expires_at"] is not None and row["expires_at"] <= now and row["state"] in _EXPIRABLE:
                row.update(pinned=False, state=S.EXPIRED.value)
        cutoff = now - SUBAGENT_EXPIRED_RETENTION_SECONDS - limits.ttl_hours * 3600
        for key in [key for key, row in table.items() if row["state"] == S.EXPIRED and row["updated_at"] < cutoff and not _owed(row)]:
            del table[key]
        pinned = sorted((row for row in table.values() if row["pinned"]), key=lambda row: row["updated_at"], reverse=True)
        for row in pinned[max(limits.max_resumable, 0):]:
            row["pinned"] = False

    def apply_subagent_limits(self, *, limits: SubagentLimits, subagent_id: Optional[str] = None, now: Optional[float] = None) -> Optional[SubagentRecord]:
        self._apply_limits(limits, time.time() if now is None else now)
        return None if subagent_id is None else self.get_subagent(subagent_id=subagent_id)

    def _recover_subagents(self, recoveries: Sequence[SubagentRecovery], limits: Optional[SubagentLimits], now: float) -> None:
        """The boot recovery's subagent half: each record still held by its dead owner is written as given."""
        for recovery in recoveries:
            _stored(recovery.write.document)
        for recovery in recoveries:
            row = self._row(recovery.subagent_id)
            if row is None or row["state"] not in (S.RUNNING, S.CLAIMED) or (row["owner_pid"], row["owner_started_at"]) != (recovery.owner_pid, recovery.owner_started_at):
                continue
            self._upsert(recovery.subagent_id, recovery.write, now)
            if limits is not None:
                self._apply_limits(limits, now)

    # ── continue ─────────────────────────────────────────────────────────

    def claim_subagent(self, *, subagent_id: str, expected_state: SubagentState, owner_pid: int, owner_started_at: Optional[int], now: Optional[float] = None) -> bool:
        row = self._row(subagent_id)
        if row is None or row["state"] != expected_state:
            return False
        row.update(state=S.CLAIMED.value, owner_pid=owner_pid, owner_started_at=owner_started_at, updated_at=time.time() if now is None else now)
        row["document"]["auto_promised"] = False
        return True

    def hand_back_subagent(self, *, subagent_id: str, state: SubagentState, promised: Optional[bool] = None, owner_pid: Optional[int] = None) -> None:
        row = self._row(subagent_id)
        if row is None or row["state"] != S.CLAIMED or (owner_pid is not None and row["owner_pid"] != owner_pid):
            return
        document, target = row["document"], S(state)
        cancel = document.get("auto_cancel")
        if isinstance(cancel, dict) and cancel.get("state"):
            promised = False
            if target == S.RESUMABLE:
                target = S(cancel["state"])
                document["auto_note"] = cancel.get("note")
                if target == S.STOPPED:
                    row["pinned"] = False
        if promised is not None:
            document["auto_promised"] = bool(promised)
        document.pop("auto_claim", None)
        document.pop("auto_cancel", None)
        row["state"] = target.value

    # ── automatic resume ─────────────────────────────────────────────────

    def _note(self, row: dict[str, Any], *, note: str, state: Optional[SubagentState], unpin: bool, now: float) -> None:
        document = row["document"]
        document["auto_note"] = note
        if state in _CANCEL_STATES:
            document["auto_promised"] = False
        document.pop("auto_claim", None)
        row["updated_at"] = now
        if state:
            row["state"] = S(state).value
            if unpin:
                row["pinned"] = False

    def note_subagent(self, *, subagent_id: str, note: str, state: Optional[SubagentState] = None, from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False, unstarted_claim: bool = False, now: Optional[float] = None) -> bool:
        row = self._row(subagent_id)
        if row is None or not (row["state"] in {S(s).value for s in from_states} or (unstarted_claim and _unstarted_auto_claim(row))):
            return False
        self._note(row, note=note, state=state, unpin=unpin, now=time.time() if now is None else now)
        return True

    def cancel_expired_subagent(self, *, subagent_id: str, note: str, now: Optional[float] = None) -> bool:
        row = self._row(subagent_id)
        if row is None or row["state"] != S.EXPIRED or not _owed(row):
            return False
        row["document"].update(auto_note=note, auto_promised=False)
        row["updated_at"] = time.time() if now is None else now
        return True

    def cancel_manual_subagent_claim(self, *, subagent_id: str, state: Literal[SubagentState.STOPPED, SubagentState.SUPERSEDED], note: str, now: Optional[float] = None) -> bool:
        row = self._row(subagent_id)
        if row is None or row["state"] != S.CLAIMED or row["document"].get("auto_claim") is not None:
            return False
        row["document"]["auto_cancel"] = {"state": S(state).value, "note": note}
        row["updated_at"] = time.time() if now is None else now
        return True

    def claim_auto_resume(self, *, subagent_id: str, expected_attempts: int, parent_session_id: SessionId, owner_pid: int, owner_started_at: Optional[int], now: Optional[float] = None) -> AutoResumeClaim:
        at = time.time() if now is None else now
        row = self._row(subagent_id)
        if row is None:
            return AutoResumeClaim(record=None, superseded=None)
        document = row["document"]
        if (row["state"] != S.RESUMABLE or document.get("auto_off") or not document.get("auto_promised")
                or int(document.get("auto_attempts") or 0) != expected_attempts or expected_attempts >= AUTO_RESUME_MAX_ATTEMPTS):
            return AutoResumeClaim(record=None, superseded=None)
        if reason := self.subagent_parent_superseded(parent_session_id=parent_session_id):
            self._note(row, note=reason, state=S.SUPERSEDED, unpin=False, now=at)
            return AutoResumeClaim(record=None, superseded=reason)
        row.update(state=S.CLAIMED.value, owner_pid=owner_pid, owner_started_at=owner_started_at, updated_at=at)
        document.update(auto_attempts=expected_attempts + 1, auto_last_at=at, auto_claim=str(parent_session_id))
        return AutoResumeClaim(record=_record(row), superseded=None)

    def confirm_auto_start(self, *, subagent_id: str) -> Optional[str]:
        row = self._row(subagent_id)
        if row is None or not _unstarted_auto_claim(row):
            return AUTO_RESUME_CANCELLED
        if reason := self.subagent_parent_superseded(parent_session_id=SessionId(str(row["document"]["auto_claim"]))):
            self._note(row, note=reason, state=S.SUPERSEDED, unpin=False, now=time.time())
            return f"{AUTO_RESUME_CANCELLED}: {reason}"
        return None

    def settle_auto_refusal(self, *, subagent_id: str, kind: AutoRefusal, attempts: int, note: str, owner_pid: int) -> None:
        row = self._row(subagent_id)
        if row is None:
            return
        document = row["document"]
        if kind == AutoRefusal.CANCELLED:
            document["auto_attempts"] = attempts
        elif kind == AutoRefusal.TRANSIENT:
            document.update(auto_attempts=attempts, auto_note=note)
        else:
            document.update(auto_off=note, auto_note=note)
        document.pop("auto_claim", None)
        if row["state"] == S.CLAIMED and row["owner_pid"] == owner_pid:
            row["state"] = S.RESUMABLE.value

    def give_up_auto_resume(self, *, subagent_id: str, reason: str, completion: Optional[GaveUpCompletion], now: Optional[float] = None) -> GiveUpOutcome:
        at = time.time() if now is None else now
        row = self._row(subagent_id)
        if row is None or row["state"] not in (S.RESUMABLE, S.EXPIRED) or not _owed(row):
            return GiveUpOutcome.NOT_OWED
        ledger = self._table("async_delegations")
        unit = ledger.get(str(completion.unit_id)) if completion is not None else None
        if unit is not None and completion is not None and unit["updated_at"] != completion.unit_updated_at:
            return GiveUpOutcome.ROUTE_MOVED
        event = _stored(completion.event) if completion is not None else None
        result = _stored(completion.result) if completion is not None else None
        row["document"].update(auto_gave_up=reason, auto_off=reason, auto_note=reason)
        row["updated_at"] = at
        if unit is None or completion is None or event is None:
            return GiveUpOutcome.NO_ROUTE
        self._insert_gave_up(completion, event, result, at)
        return GiveUpOutcome.RECORDED

    def _insert_gave_up(self, completion: GaveUpCompletion, event: dict[str, Any], result: Any, now: float) -> None:
        """Insert-or-ignore the failed completion on its own ledger row, in the shape the dispatch writes."""
        from ..workers.delegations import _dispatch_row

        ledger = self._table("async_delegations")
        delegation_id = str(event["delegation_id"])
        if delegation_id in ledger:
            return
        row = _dispatch_row(delegation_id, event.get("session_key") or "", event.get("origin_session_id") or "", event.get("dispatched_at") or now)
        row.update(origin_ui_session_id=event.get("origin_ui_session_id") or "", parent_session_id=event.get("parent_session_id"),
                   state="failed", completed_at=event.get("completed_at") or now, updated_at=now, event=event, result=result,
                   owner_pid=completion.owner_pid, owner_birth=completion.owner_started_at,
                   task={k: event.get(k) for k in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")})
        ledger[delegation_id] = row

    def supersede_subagents_for_session(self, *, note: str, session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "", promised_only: bool = False) -> int:
        selectors = [(column, value) for column, value in (("origin_session", session_key), ("origin_ui_session_id", origin_ui_session_id),
                                                           ("parent_session_id", parent_session_id)) if value]
        if not selectors:
            return 0
        units = {row["delegation_id"] for row in self._table("async_delegations").values() if any(row[column] == value for column, value in selectors)}
        rows = [row for row in self._subagents().values()
                if row["state"] == S.CLAIMED or (row["state"] == S.RESUMABLE and (not promised_only or _owed(row))) or (row["state"] == S.EXPIRED and _owed(row))]
        now, moved = time.time(), 0
        for row in rows:
            if row["document"].get("unit_id") not in units:
                continue
            sid = row["subagent_id"]
            moved += bool(self.note_subagent(subagent_id=sid, note=note, state=S.SUPERSEDED, unstarted_claim=True, now=now)
                          or self.cancel_expired_subagent(subagent_id=sid, note=note, now=now)
                          or self.cancel_manual_subagent_claim(subagent_id=sid, state=S.SUPERSEDED, note=note, now=now))
        return moved

    def dismiss_subagent(self, *, expected: SubagentRecord, note: str, now: Optional[float] = None) -> Optional[Literal["dismissed", "recorded"]]:
        at = time.time() if now is None else now
        sid = expected.subagent_id
        row = self._row(sid)
        if row is None or _record(row) != expected:
            return None
        if row["state"] == S.CLAIMED and row["document"].get("auto_claim") is None:
            return "recorded" if self.cancel_manual_subagent_claim(subagent_id=sid, state=S.STOPPED, note=note, now=at) else None
        done = (self.note_subagent(subagent_id=sid, note=note, state=S.STOPPED, from_states=(S.RESUMABLE, S.SUPERSEDED), unpin=True, unstarted_claim=True, now=at)
                or self.cancel_expired_subagent(subagent_id=sid, note=note, now=at))
        return "dismissed" if done else None
