"""Durable subagent records (``resumable_subagents``): the delegation ledger's record of each delegated child a parent can continue after a restart or crash, and the worktree pin the startup sweep honours.

One row per subagent id. The columns are typed (:class:`SubagentRecord`); the run record itself (goal, route request, owners, cwd, worktree baselines, the automatic-resume bookkeeping) stays the caller's decoded JSON document, and every change main makes inside it is a typed verb here: the checkpoint's cwd, the claim's attempt and parent, a note, a cancel, a hand-back, a refusal settlement, a give-up. Policy a verb needs comes in as parameters the caller reads (:class:`SubagentLimits` from ``delegation.resumable_ttl_hours`` and ``delegation.max_resumable``).

Writes the caller computes from a record it read (a run's start or stop record, a dismissal) are compare-and-swap: they apply only while the stored row still equals the ``expected`` record, so a decision taken on a read never lands over a row that moved; a lost write returns False and the caller re-reads. Decisions that depend on another table (whether the parent conversation still wants the child) are evaluated by the verb inside its own transaction. Composites main runs in one write stay one verb: the boot recovery with the ledger's own recovery (:meth:`persistence.authority.coordination.CoordinationRepository.recover_abandoned_delegations`), the automatic claim, the give-up with its completion row, the supersession of a session's children.

Every write is for an existing profile store: a write whose profile was removed must not create its store again; the store reports it as a write that did nothing.

"Owed" (:attr:`SubagentRecord.owed`) is a record whose stop promised an automatic resume (``auto_promised``) and whose one give-up failure has not been delivered (``auto_gave_up`` unset): its parent is owed exactly one report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Optional, Protocol

from ..core.identity import DelegationId, JsonValue, SessionId

__all__ = [
    "AUTO_RESUME_CANCELLED",
    "AUTO_RESUME_MAX_ATTEMPTS",
    "GATEWAY_SHUTDOWN",
    "SUBAGENT_EXPIRED_RETENTION_SECONDS",
    "AutoRefusal",
    "AutoResumeClaim",
    "AutoStart",
    "DelegationRoute",
    "GaveUpCompletion",
    "GiveUpOutcome",
    "SubagentLedger",
    "SubagentLimits",
    "SubagentRecord",
    "SubagentRecovery",
    "SubagentState",
    "SubagentWrite",
    "parent_superseded_note",
]

#: Automatic attempts a record may spend.
AUTO_RESUME_MAX_ATTEMPTS = 2
#: An expired record not owed a report is deleted this long (plus the TTL) after its last update.
SUBAGENT_EXPIRED_RETENTION_SECONDS = 7 * 24 * 3600.0
#: The stop cause a gateway shutdown marks on its running records.
GATEWAY_SHUTDOWN = "gateway_shutdown"
#: The refusal text of an automatic start the parent cancelled.
AUTO_RESUME_CANCELLED = "Not continued: its parent cancelled the automatic resume"


class SubagentState(StrEnum):
    """A record's ``state``: ``running`` live, ``claimed`` while a continue starts, ``resumable`` (pinned), ``completed``, ``stopped`` (a person or the parent stopped it), ``superseded`` (its parent moved on), ``expired`` (TTL passed)."""

    RUNNING = "running"
    CLAIMED = "claimed"
    RESUMABLE = "resumable"
    COMPLETED = "completed"
    STOPPED = "stopped"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"



@dataclass(frozen=True, slots=True)
class SubagentRecord:
    """One stored row: the columns typed, ``document`` the decoded run record (never carrying ``state``, ``stop_cause``; ``auto_claim`` and ``auto_cancel`` only while a claim holds the row)."""

    subagent_id: str
    state: SubagentState
    pinned: bool
    stop_cause: Optional[str]
    owner_pid: Optional[int]
    owner_started_at: Optional[int]
    document: Mapping[str, JsonValue]
    created_at: float
    updated_at: float
    expires_at: Optional[float]

    @property
    def owed(self) -> bool:
        """Promised an automatic resume and no give-up delivered yet."""
        return bool(self.document.get("auto_promised")) and self.document.get("auto_gave_up") is None


@dataclass(frozen=True, slots=True)
class SubagentWrite:
    """A whole row as a run's start or stop (or a recovery) writes it; the verb drops ``state``, ``stop_cause``, ``auto_claim`` and ``auto_cancel`` from ``document`` and stamps ``updated_at`` (and ``created_at`` on insert)."""

    state: SubagentState
    pinned: bool
    stop_cause: Optional[str]
    owner_pid: Optional[int]
    owner_started_at: Optional[int]
    document: Mapping[str, JsonValue]
    expires_at: Optional[float]


@dataclass(frozen=True, slots=True)
class SubagentLimits:
    """The profile's ``delegation.resumable_ttl_hours`` and ``delegation.max_resumable`` as the caller read them."""

    ttl_hours: float
    max_resumable: int


@dataclass(frozen=True, slots=True)
class SubagentRecovery:
    """A boot recovery of one record whose owner died, built outside the store from the record it read: written only while the row is still ``running`` or ``claimed`` and stamped with ``owner`` (the dead owner)."""

    subagent_id: str
    owner_pid: Optional[int]
    owner_started_at: Optional[int]
    write: SubagentWrite


@dataclass(frozen=True, slots=True)
class AutoResumeClaim:
    """The automatic claim's answer: the claimed row, or the supersession note when the parent moved on (``record`` None), or both None when the row moved."""

    record: Optional[SubagentRecord]
    superseded: Optional[str]


@dataclass(frozen=True, slots=True)
class AutoStart:
    """An automatic run's start record: ``written``, or ``cancelled`` with the reason the parent's last word gives, or neither when the row moved since the caller read it."""

    written: bool
    cancelled: Optional[str]


class AutoRefusal(StrEnum):
    """How a refused automatic continue settles: ``cancelled`` (the parent's decision already wrote the row: only the attempt is refunded), ``transient`` (refunded, noted), ``permanent`` (automatic resumes of the record end)."""

    CANCELLED = "cancelled"
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class GiveUpOutcome(StrEnum):
    """``not_owed`` (a cancel cleared the debt: nothing written), ``no_route`` (marked, no unit row to report on), ``route_moved`` (the unit row changed since the caller read it: nothing written), ``recorded`` (marked and its failed completion recorded pending)."""

    NOT_OWED = "not_owed"
    NO_ROUTE = "no_route"
    ROUTE_MOVED = "route_moved"
    RECORDED = "recorded"


@dataclass(frozen=True, slots=True)
class DelegationRoute:
    """The ledger row a child's unit reported on, as the scheduler reads it: its route columns, the decoded task and result, its dispatch time and ``updated_at``."""

    delegation_id: DelegationId
    origin_session: str
    origin_ui_session_id: str
    parent_session_id: Optional[SessionId]
    origin_session_id: str
    task: Optional[JsonValue]
    result: Optional[JsonValue]
    dispatched_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class GaveUpCompletion:
    """The one failed completion a give-up records, built outside the store from the unit's :class:`DelegationRoute`: inserted (never replacing a row) only while the unit row still has ``unit_updated_at``. ``event`` and ``result`` are caller-redacted; the row's id is ``event["delegation_id"]``."""

    unit_id: DelegationId
    unit_updated_at: float
    event: Mapping[str, JsonValue]
    result: Mapping[str, JsonValue]
    owner_pid: int
    owner_started_at: Optional[int]


def parent_superseded_note(tip_end_reason: Optional[str], *, tip_exists: bool) -> Optional[str]:
    """Why the parent no longer wants its child resumed, from the live tip of its compression chain: gone, or ended for a deliberate (not automatic) reason; None while it still does. ``tip_end_reason`` is the tip's end reason when it ended, else None."""
    from ..conversation.sessions import is_automatic_end_reason

    if not tip_exists:
        return "its parent session no longer exists"
    if not tip_end_reason or is_automatic_end_reason(tip_end_reason):
        return None
    return f"its parent session ended ({tip_end_reason})"


class SubagentLedger(Protocol):
    """The subagent-record verbs of the delegation ledger (module docstring). Times default to now; ``now`` makes a verb testable without a clock."""

    # ── reads ──

    def get_subagent(self, *, subagent_id: str) -> Optional[SubagentRecord]:
        """The row, or None."""
        ...

    def list_inflight_subagents(self) -> tuple[SubagentRecord, ...]:
        """Every ``running`` or ``claimed`` row: the boot recovery's plan reads them to judge each owner."""
        ...

    def list_continuable_subagents(self, *, unit_id: str) -> tuple[str, ...]:
        """The ids of ``resumable`` rows of the async unit (the document's ``unit_id``, else its ``delegation_id``), oldest ``created_at`` first."""
        ...

    def list_pinned_subagents(self, *, max_resumable: int, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        """The worktree sweep's view: pinned rows in ``resumable``, ``superseded``, ``running`` or ``claimed`` not expired (``expires_at`` unset or after now), newest ``updated_at`` first, at most ``max_resumable``. A read only: a missing store or table answers ()."""
        ...

    def list_auto_resume_candidates(self, *, owed_only: bool = False) -> tuple[SubagentRecord, ...]:
        """The scheduler's rows: ``resumable`` ones, and ``expired`` ones still owed; with ``owed_only`` only owed rows; oldest ``updated_at`` first."""
        ...

    def has_owed_subagents(self) -> bool:
        """Whether a ``resumable`` or ``expired`` row is owed its report. A read only: a missing store answers False and is not created."""
        ...

    def list_waiting_subagents(self, *, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        """``resumable`` owed rows not expired (``expires_at`` unset or after now): the children waiting on a promised automatic resume."""
        ...

    def subagent_parent_superseded(self, *, parent_session_id: SessionId) -> Optional[str]:
        """:func:`parent_superseded_note` for the live tip of ``parent_session_id``'s compression chain (the walk the turn lease uses, capped, cycles stopping it)."""
        ...

    def get_delegation_route(self, *, delegation_id: DelegationId) -> Optional[DelegationRoute]:
        """The ledger row's route, task, result, dispatch time and ``updated_at``, or None."""
        ...

    # ── run records ──

    def put_subagent(self, *, subagent_id: str, write: SubagentWrite, expected: Optional[SubagentRecord], limits: Optional[SubagentLimits] = None, now: Optional[float] = None) -> bool:
        """Upsert the row while the stored one equals ``expected`` (None: no row yet); ``created_at`` is kept on update. With ``limits``, :meth:`apply_subagent_limits` runs in the same write (a stopped record's write). False when the row moved, with nothing written."""
        ...

    def start_auto_resumed_run(self, *, subagent_id: str, write: SubagentWrite, expected: SubagentRecord, owner_pid: int, now: Optional[float] = None) -> AutoStart:
        """An automatic run's start record, in one write that first gives the parent its last word: the row ``claimed`` by ``owner_pid`` with ``auto_claim`` set and its parent not superseded proceeds to :meth:`put_subagent`'s compare-and-swap; superseded, it is noted ``superseded`` (debt cleared, claim dropped) and its attempt refunded; a ``stopped`` or ``superseded`` row has its attempt refunded; the cancel reason is ``"<AUTO_RESUME_CANCELLED>: <note>"`` (the row's ``auto_note``, else ``its parent stopped it``); any other row answers ``"<AUTO_RESUME_CANCELLED>: the record moved (<state or gone>)"`` with nothing written. A refund is ``auto_attempts`` minus one (an unset count read as 1), never below 0."""
        ...

    def checkpoint_subagent(self, *, subagent_id: str, cwd: str, repo_baselines: Mapping[str, JsonValue], now: Optional[float] = None) -> bool:
        """A ``running`` row's document takes ``cwd`` and ``repo_baselines``, ``updated_at`` stamped; True when it changed."""
        ...

    def mark_subagents_gateway_shutdown(self, *, owner_pid: int) -> int:
        """Every ``running`` row of ``owner_pid`` takes stop cause :data:`GATEWAY_SHUTDOWN` (nothing else moves); returns how many."""
        ...

    def apply_subagent_limits(self, *, limits: SubagentLimits, subagent_id: Optional[str] = None, now: Optional[float] = None) -> Optional[SubagentRecord]:
        """One write: rows in ``resumable``, ``completed``, ``stopped`` or ``superseded`` whose ``expires_at`` has passed become ``expired`` and unpinned; ``expired`` rows not owed and not updated for :data:`SUBAGENT_EXPIRED_RETENTION_SECONDS` plus the TTL are deleted; pins beyond the newest ``max_resumable`` (by ``updated_at``) drop, the rows staying. Returns ``subagent_id``'s row after that, when given."""
        ...

    # ── continue ──

    def claim_subagent(self, *, subagent_id: str, expected_state: SubagentState, owner_pid: int, owner_started_at: Optional[int], now: Optional[float] = None) -> bool:
        """The parent's continue: while the row is in ``expected_state`` it becomes ``claimed`` by the owner, ``updated_at`` stamped, its debt cleared (``auto_promised`` false); True when this call moved it."""
        ...

    def hand_back_subagent(self, *, subagent_id: str, state: SubagentState, promised: Optional[bool] = None, owner_pid: Optional[int] = None) -> None:
        """Return a ``claimed`` row (of ``owner_pid`` when given) no run started to ``state``, ``auto_claim`` and ``auto_cancel`` removed. A cancel recorded during the claim (``auto_cancel``) wins: no debt, and a hand-back to ``resumable`` takes the cancel's state and note instead (unpinned when that is ``stopped``). ``promised``, when given and not overridden, is stored as ``auto_promised``. ``updated_at`` does not move; another row is left alone."""
        ...

    # ── automatic resume ──

    def note_subagent(self, *, subagent_id: str, note: str, state: Optional[SubagentState] = None, from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False, unstarted_claim: bool = False, now: Optional[float] = None) -> bool:
        """For a row in ``from_states`` (or, with ``unstarted_claim``, a ``claimed`` row an automatic continue holds: ``auto_claim`` set): ``auto_note`` set, ``auto_claim`` removed, ``updated_at`` stamped; with ``state`` the row moves there (unpinned with ``unpin``), and ``superseded`` or ``stopped`` also clears its debt. True when a row changed."""
        ...

    def cancel_expired_subagent(self, *, subagent_id: str, note: str, now: Optional[float] = None) -> bool:
        """An owed ``expired`` row: ``auto_note`` set, debt cleared, ``updated_at`` stamped."""
        ...

    def cancel_manual_subagent_claim(self, *, subagent_id: str, state: Literal[SubagentState.STOPPED, SubagentState.SUPERSEDED], note: str, now: Optional[float] = None) -> bool:
        """A row the parent's own continue holds (``claimed``, no ``auto_claim``) records the cancel as ``auto_cancel`` ``{state, note}`` for its hand-back; ``updated_at`` stamped."""
        ...

    def claim_auto_resume(self, *, subagent_id: str, expected_attempts: int, parent_session_id: SessionId, owner_pid: int, owner_started_at: Optional[int], now: Optional[float] = None) -> AutoResumeClaim:
        """The automatic claim in one write: a row still ``resumable``, not ``auto_off``, promised, with ``expected_attempts`` attempts and under :data:`AUTO_RESUME_MAX_ATTEMPTS` is checked against its parent (:meth:`subagent_parent_superseded`): superseded, it is noted ``superseded`` and the note returned; else it becomes ``claimed`` by the owner, ``updated_at`` and ``auto_last_at`` now, ``auto_attempts`` one more, ``auto_claim`` the parent, and the claimed row is returned. Any other row: both None, nothing written."""
        ...

    def confirm_auto_start(self, *, subagent_id: str) -> Optional[str]:
        """Right before dispatch, in one write: None while the row is ``claimed`` by an automatic continue whose parent still wants it; a superseded parent notes the row ``superseded`` (as an unstarted claim) and answers ``"<AUTO_RESUME_CANCELLED>: <note>"``; any other row answers :data:`AUTO_RESUME_CANCELLED`."""
        ...

    def settle_auto_refusal(self, *, subagent_id: str, kind: AutoRefusal, attempts: int, note: str, owner_pid: int) -> None:
        """A continue refused before anything ran: ``cancelled`` and ``transient`` store ``auto_attempts`` ``attempts`` (``transient`` also ``auto_note``), ``permanent`` stores ``auto_off`` and ``auto_note``; ``auto_claim`` removed; a row ``claimed`` by ``owner_pid`` returns to ``resumable``. ``updated_at`` does not move."""
        ...

    def give_up_auto_resume(self, *, subagent_id: str, reason: str, completion: Optional[GaveUpCompletion], now: Optional[float] = None) -> GiveUpOutcome:
        """A promised record will not be resumed, in one write: an owed ``resumable`` or ``expired`` row takes ``auto_gave_up``, ``auto_off`` and ``auto_note`` ``reason`` and ``updated_at``; then, with ``completion`` and its unit row present, the failed completion is inserted (state ``failed``, delivery ``pending``, never replacing a row). See :class:`GiveUpOutcome`; ``route_moved`` and ``not_owed`` write nothing."""
        ...

    def supersede_subagents_for_session(self, *, note: str, session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "", promised_only: bool = False) -> int:
        """A person stopped, reset or closed a conversation, in one write: every child row of a ledger unit matching any given selector (``origin_session``, ``origin_ui_session_id``, ``parent_session_id``) that is ``claimed``, ``resumable`` (owed only with ``promised_only``) or owed ``expired`` is noted ``superseded`` (an unstarted automatic claim included), or its expired debt cancelled, or a manual claim's cancel recorded. Returns how many; no selector is 0 with no write."""
        ...

    def dismiss_subagent(self, *, expected: SubagentRecord, note: str, now: Optional[float] = None) -> Optional[Literal["dismissed", "recorded"]]:
        """The parent's stop on a waiting child it owns, while the row still equals ``expected`` (the caller judged ownership and eligibility on it): a manual claim records the cancel as ``stopped`` (``recorded``); else the row is noted ``stopped`` and unpinned from ``resumable`` or ``superseded`` or an unstarted automatic claim, or its expired debt is cancelled (``dismissed``). None when nothing changed or the row moved."""
        ...
