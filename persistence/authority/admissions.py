"""The admission ledger: every session input as one durable row, admitted once per (principal, session, request), claimed in FIFO order, settled with terminal evidence.

``AdmissionRepository`` offers the admission operations and the admission reads the gateway makes. Records keep the ``session_admissions`` row keys and these string vocabularies: statuses ``queued``, ``started``, ``unknown``, ``terminal``; outcomes ``completed``, ``interrupted``, ``rejected``, ``failed``, ``cancelled``, ``steering``, ``steered``. Refusals are ``AuthorityRefusal`` with a reason code (``FenceStale``, ``UnknownExecution``, ``ProofRejected`` for the fixed ones).

Owner-memory checks arrive as data: ``gate`` is the ``GateRefusal`` the caller decided just before the write, or ``None``; the verb raises its refusal at a fixed point inside the write (``admit`` after the queue cap, ``claim`` right after the epoch fence), and exact replays bypass it. The SQL-bearing hooks callers pass into ``admit`` are typed parameters (``replaces``, ``supersedes``, ``api_target``, ``rewrite``, ``hosted``) run inside the same write after the gate, before the row is inserted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Literal, NewType, Protocol, runtime_checkable

from ..core.errors import AuthorityRefusal, Conflict
from .vocabulary import STEERING

if TYPE_CHECKING:
    from ..gateway.hosted_rooms import HostedMemberGuard

__all__ = [
    "ANY_HEAD",
    "ActiveReplace",
    "AdmissionId",
    "AdmissionIntent",
    "AdmissionOutcome",
    "AdmissionRecord",
    "AdmissionRepository",
    "AdmissionResult",
    "AdmissionStatus",
    "AnyHead",
    "ApiTargetClaim",
    "CapturedResult",
    "CarrierRewind",
    "ClaimedAdmission",
    "GateRefusal",
    "QueuedFold",
    "QueuedReplacement",
    "RetryRewriteRefused",
    "TranscriptRewrite",
    "TurnBoundary",
    "UnheldMedia",
    "steer_replayed",
]

#: Opaque identity of one admission row (the SQLite backend mints ``uuid4().hex``).
AdmissionId = NewType("AdmissionId", str)


class AdmissionIntent(StrEnum):
    """The stored ``intent`` strings; records carry the plain string."""

    QUEUE = "queue"
    STEER = "steer"
    REDIRECT = "redirect"


class AdmissionStatus(StrEnum):
    """The stored ``status`` strings; records carry the plain string."""

    QUEUED = "queued"
    STARTED = "started"
    UNKNOWN = "unknown"
    TERMINAL = "terminal"


class AdmissionOutcome(StrEnum):
    """The stored ``outcome`` strings; records carry the plain string (``None`` until terminal)."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STEERING = "steering"
    STEERED = "steered"


class AnyHead(Enum):
    """The type of :data:`ANY_HEAD`."""

    ANY_HEAD = "any_head"


#: ``claim(head=ANY_HEAD)`` claims whatever row heads the FIFO.
ANY_HEAD = AnyHead.ANY_HEAD


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
    """One admission row: the ``session_admissions`` columns without ``payload_json``, ``payload_digest`` and ``lineage_json``, with ``payload`` decoded. A retired row is its terminal tombstone, read the same way."""

    seq: int
    admission_id: str
    request_id: str
    principal_id: str
    target_session_id: str
    payload: Mapping[str, Any]
    intent: str
    status: str
    outcome: str | None
    owner_epoch: int | None
    generation: int | None


@dataclass(frozen=True, slots=True)
class ClaimedAdmission(AdmissionRecord):
    """The row ``claim`` started: ``transcript_mark`` is the highest ``messages.id`` at the claim, ``queued_owner_epoch`` the ``owner_epoch`` the row held while queued, which ``return_claim`` restores."""

    transcript_mark: int
    queued_owner_epoch: int | None


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """An admission and the stored result of its settlement, read together (``settled_result``)."""

    record: AdmissionRecord
    result: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class GateRefusal:
    """An owner check the caller decided before the call: the verb that takes it raises its refusal at the point it checks the gate."""

    reason: str
    field: str | None = None
    detail: str | None = None

    def refusal(self) -> AuthorityRefusal:
        """The refusal the verb raises: ``AuthorityRefusal(reason, field=field, detail=detail)``."""
        return AuthorityRefusal(self.reason, field=self.field, detail=self.detail)


@dataclass(frozen=True, slots=True)
class TurnBoundary:
    """The discarded-turn closer ``settle`` runs: ``notice`` is the boundary copy, ``replace_after`` the claim's transcript mark, ``display_metadata`` the error card the boundary carries."""

    notice: str
    replace_after: int | None = None
    display_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CapturedResult:
    """The result this owner captured before its settlement write failed; ``resolve_unknown`` commits it."""

    outcome: Literal["completed", "interrupted", "failed"]
    result: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QueuedReplacement:
    """``admit`` replaces the still-queued ``admission_id``."""

    admission_id: str


@dataclass(frozen=True, slots=True)
class QueuedFold:
    """``admit`` folds the still-queued completion ``admission_id`` into this batch: it must be the FIFO tail but for ``skip_ids`` (the finishing process's ticks, withdrawn right after)."""

    admission_id: str
    skip_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ApiTargetClaim:
    """An API turn's target claim: one target per API request id, and the session's API settings written with the admitted turn. ``client_key`` is whether the request id came from the client, so a retired tombstone also spends it."""

    settings: Mapping[str, Any]
    client_key: bool


@dataclass(frozen=True, slots=True)
class CarrierRewind:
    """The ``/retry`` rewind of a composite compaction carrier: rewind ``target_row_id`` keeping the handoff, when the active ids are still ``expected_active_ids`` and the target content is still ``expected_target_content``."""

    session_id: str
    target_row_id: int
    expected_active_ids: tuple[int, ...]
    expected_target_content: Any


@dataclass(frozen=True, slots=True)
class ActiveReplace:
    """The ``/retry`` replacement of the active transcript: the active rows become ``messages``, when the active ids are still ``expected_active_ids``."""

    session_id: str
    messages: tuple[Mapping[str, Any], ...]
    expected_active_ids: tuple[int, ...]


#: The ``/retry`` transcript rewrite ``admit`` runs inside the admission write.
TranscriptRewrite = CarrierRewind | ActiveReplace


@dataclass(frozen=True, slots=True)
class UnheldMedia:
    """``unheld_media``'s answer: the given references nothing holds (``candidates``) and every held path (``held``)."""

    candidates: tuple[Mapping[str, Any], ...]
    held: frozenset[str]


class RetryRewriteRefused(Conflict, RuntimeError):
    """The planned ``/retry`` rewrite no longer applies (the transcript changed since the read, a turn lease is live, the session closed): the re-send was refused with it and nothing was written."""

    def __init__(self, session_id: str, message: str) -> None:
        super().__init__("transcript", session_id)
        self.args = (message,)

    def __str__(self) -> str:
        return str(self.args[0])


def steer_replayed(row: AdmissionRecord | Mapping[str, Any]) -> bool:
    """``row`` is a committed transfer (still in flight, or settled ``steered``): an exact retry of ``prompt.steer_queued`` replays it."""
    status = row["status"] if isinstance(row, Mapping) else row.status
    outcome = row["outcome"] if isinstance(row, Mapping) else row.outcome
    return status == "terminal" and outcome in (STEERING, "steered")


@runtime_checkable
class AdmissionRepository(Protocol):
    """The admission ledger of the handle's profile; every verb is one transaction."""

    # Writes

    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any], intent: Literal['queue', 'steer', 'redirect'] = 'queue', gate: GateRefusal | None = None, replaces: QueuedReplacement | None = None, supersedes: str | None = None, api_target: ApiTargetClaim | None = None, rewrite: TranscriptRewrite | None = None, hosted: HostedMemberGuard | None = None, fold: QueuedFold | None = None) -> tuple[AdmissionRecord, bool]:
        """Admit one input; the bool is ``created``.

        Order: identity texts, intent and payload encoding (``invalid_params``); the tombstone read on a retired target (epoch fence; an exact retry replays its tombstone with ``created`` False, an unseen identity is ``not_found``, another digest ``admission_conflict`` with :data:`ADMITTED_REQUEST_CONFLICT`); then one write: epoch, session (``not_found``), the (principal, target, request) row (another digest ``admission_conflict`` with :data:`ADMITTED_REQUEST_CONFLICT`; an exact live replay runs only the ``replaces`` check and returns ``created`` False), the queue cap (``queue_full`` at :data:`ADMISSION_QUEUE_MAX` queued rows), ``gate`` (its refusal is raised when given), then the typed extras: ``replaces`` (the replaced row must still be queued, else ``admission_started``; it is cancelled and bound to this identity), ``supersedes`` (that queued row on this target is cancelled, else ``stale_generation``), ``api_target`` (another target holds this API request id, live or, for a client key, tombstoned: ``admission_conflict``; the settings are stored), ``rewrite`` (the ``/retry`` rewrite; a failure other than a refusal is ``RetryRewriteRefused``), ``hosted`` (the hosted member check, ``permission_denied``), ``fold`` (the folded row must still be queued on this target with no later queued row outside ``skip_ids``, else ``stale_generation``; it is cancelled and its payload becomes ``session_rules.automation.folded_payload`` of it); then the queued insert.
        """
        ...

    def claim(self, *, epoch: int, session_id: str, gate: GateRefusal | None = None, head: str | None | AnyHead = ANY_HEAD) -> ClaimedAdmission | Literal[False] | None:
        """Start the FIFO head under the next execution generation.

        Order: epoch, ``gate`` (its refusal is raised when given), session (``not_found``); an ``unknown`` admission or worker on the session is ``unknown_execution``; a started admission or a live worker is ``None``; ``head`` other than :data:`ANY_HEAD` must name the current head (``None``: none queued) else ``False``; an empty queue is ``None``. The claim bumps ``runtime_generation`` and starts the row under ``epoch``.
        """
        ...

    def return_claim(self, *, epoch: int, admission_id: str, generation: int, queued_owner_epoch: int | None) -> AdmissionRecord:
        """Undo a claim whose turn never started (the claim's two updates reversed).

        Order: epoch (``stale_epoch``); the row (``not_found``); a row already ``queued`` with no generation is returned unchanged; otherwise the row must be ``started`` with ``owner_epoch == epoch`` and this ``generation``, the session's ``runtime_generation`` must equal ``generation``, and no worker execution and no steering transfer may be bound to ``(epoch, generation)``, else ``stale_generation``. The row goes back to ``queued`` with ``owner_epoch = queued_owner_epoch`` and no generation (its ``seq`` unchanged) and ``runtime_generation`` back to ``generation - 1``; no revision bump.
        """
        ...

    def settle(self, *, epoch: int, admission_id: str, generation: int, outcome: Literal['completed', 'interrupted', 'rejected', 'failed'], result: Mapping[str, Any] | None = None, transfers: Mapping[str, bool] | None = None, boundary: TurnBoundary | None = None) -> AdmissionRecord:
        """Settle a started admission.

        Order: outcome (``invalid_params``); epoch; the row (``not_found``); its session (``not_found``); the composite fence (started, owned by ``epoch``, at ``generation``, the session at ``generation``; else ``stale_generation``); then: its workers retire, ``boundary`` closes the turn on the current physical transcript, ``result`` is stored compacted and encoded, the row goes terminal, the steering ``transfers`` settle (delivered ones ``steered`` on ``completed`` else ``interrupted``, undelivered ones queued again), the session revision bumps.
        """
        ...

    def cancel(self, *, epoch: int, admission_id: str) -> AdmissionRecord:
        """Cancel a queued admission: ``unknown`` refuses ``unknown_execution``; ``started`` or an in-flight steer refuses ``stale_generation``; a queued row goes terminal ``cancelled``; a terminal row is returned unchanged."""
        ...

    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[AdmissionRecord, bool]:
        """Hand a queued admission to its FIFO's running claim: an already committed transfer replays ``(row, False)``; a row not queued refuses ``admission_started``; no started claim at ``(epoch, generation)`` on its target refuses ``session_idle``; else the row is terminal ``steering`` bound to the claim, ``(row, True)``."""
        ...

    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        """Queue a transfer whose text never reached the claim's agent again at its original ``seq``; False when the claim's settlement already decided it."""
        ...

    def mark_unknown(self, *, epoch: int, admission_id: str, generation: int, transfers: Mapping[str, bool] | None = None) -> Literal['unknown', 'terminal']:
        """Mark this owner's started admission ``unknown`` after its settlement failed: another owner or generation refuses ``stale_generation``; a terminal or unknown row answers its status; a queued row refuses ``stale_generation``; else the row goes ``unknown``, its transfers settle (delivered ones ``interrupted``, others queued again) and the session revision bumps."""
        ...

    def rehome_queued(self, *, epoch: int, old_session_id: str, new_session_id: str, principal_prefix: str) -> list[str]:
        """Move the old owner's queued ``principal_prefix*`` admissions to ``new_session_id``, keeping ids, request ids and ``seq``, re-keying their digests; nothing moves behind an ``unknown`` row; both revisions bump when any moved. Returns the moved ids in FIFO order."""
        ...

    def retarget_local_automation(self, *, epoch: int, session_id: str, principal_id: str, old_targets: Iterable[str], new_target: str) -> list[str]:
        """Point the queued ``principal_id`` automation rows whose ``local_automation_v1.target`` is in ``old_targets`` at ``new_target``, re-keying their digests; nothing moves behind an ``unknown`` row; the revision bumps when any moved."""
        ...

    def recover(self, *, epoch: int) -> int:
        """Boot recovery: stale child workers retire, foreign-epoch live workers go ``unknown``, foreign-epoch in-flight steers settle ``interrupted``, and foreign-epoch started admissions go ``unknown``; returns how many admissions it marked."""
        ...

    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int, notice: str | None = None, captured: CapturedResult | None = None) -> AdmissionRecord:
        """The operator's resolution of an ``unknown`` admission.

        A row already terminal at ``generation`` (``interrupted``, or with a stored result) replays; a row not ``unknown`` at ``generation`` refuses ``stale_generation``. ``captured`` commits (outcome and stored result) when the row's ``owner_epoch`` is ``epoch``; then the discarded-turn closer runs, with ``notice`` unless ``captured`` committed; the workers retire, the row goes terminal and the revision bumps.
        """
        ...

    def compact_api_payloads(self, *, admission_ids: Sequence[str]) -> int:
        """One write: each given row re-read under the candidate predicate and, when ``session_rules.api_payloads.compacted_api_payload`` changes its payload, rewritten with it; ``payload_digest`` is untouched. Returns the rows written."""
        ...

    def unheld_media(self, references: Sequence[Mapping[str, Any]], *, after_id: int, retain_history: bool, media_root: str) -> UnheldMedia:
        """One write transaction that writes nothing: the held paths (live admissions' media, API media of every row, live hosted prompts' documents under ``media_root``, and transcript mentions above ``after_id`` when ``retain_history``) and the given references not held."""
        ...

    def record_retired_media(self, references: Sequence[Mapping[str, Any]]) -> None:
        """Record durable media retirement candidates, keyed by path digest, first write wins."""
        ...

    def drop_retired_candidates(self, keys: Sequence[str]) -> None:
        """Delete collected candidate keys."""
        ...

    # Reads

    def find(self, *, principal_id: str, session_id: str, request_id: str) -> AdmissionRecord | None:
        """The live row at its exact (principal, target, request) identity."""
        ...

    def match(self, *, request_id: str | None = None, admission_id: str | None = None, principal_id: str | None = None, session_id: str | None = None, include_retired: bool = False) -> list[AdmissionRecord]:
        """Live rows matching every given filter, in ``seq`` order; ``include_retired`` appends the API tombstones carrying ``request_id`` that match the other filters."""
        ...

    def list(self, *, session_id: str, pending_only: bool = True) -> list[AdmissionRecord]:
        """The session's rows in ``seq`` order; pending is ``queued``, ``started`` or ``unknown``."""
        ...

    def get(self, *, admission_id: str) -> AdmissionRecord | None:
        """The live row, else its terminal tombstone, else ``None``."""
        ...

    def settled_result(self, *, admission_id: str, epoch: int | None = None, generation: int | None = None) -> AdmissionResult | None:
        """The row and its stored result. Without ``epoch``: the row (``not_found``), ``None`` unless it is terminal with a stored result. With ``epoch`` and ``generation``: epoch, the row (``not_found``), its owner (``stale_generation``); a row not terminal answers with result ``None``; a terminal row without a stored result refuses ``storage_unavailable``."""
        ...

    def retired_replay(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any], intent: Literal['queue', 'steer', 'redirect'] = 'queue') -> AdmissionRecord | None:
        """``None`` for a live target; on a retired target the tombstone of the exact retry, ``not_found`` for an unseen identity, ``admission_conflict`` with :data:`ADMITTED_REQUEST_CONFLICT` for another digest."""
        ...

    def pending_targets(self, *, statuses: frozenset[str], principal_id: str | None = None) -> frozenset[str]:
        """The distinct targets holding an admission in ``statuses`` (and of ``principal_id`` when given)."""
        ...

    def list_unknown(self) -> list[AdmissionRecord]:
        """Every ``unknown`` admission in ``seq`` order."""
        ...

    def has_pending(self, *, session_ids: Sequence[str] = (), statuses: frozenset[str] | None = None) -> bool:
        """Whether a row targets one of ``session_ids`` (any target when empty) with a status in ``statuses`` (any status when ``None``)."""
        ...

    def ledger_busy(self, session_id: str, target: str) -> bool:
        """A queued, started or unknown admission on ``session_id``, or a live worker on ``session_id`` or ``target``."""
        ...

    def webhook_ingress_payloads(self, *, chat_id: str) -> list[Mapping[str, Any]]:
        """Payloads of the signed webhook ingress admissions for ``chat_id``: ``native_text_v1.source`` names the chat on platform ``webhook`` and carries no ``automation``."""
        ...

    def unfinalized_webhook_admissions(self) -> list[str]:
        """Ids of terminal webhook ingress admissions with a generation whose ``webhook`` session has not ended."""
        ...

    def terminal_ids(self, admission_ids: Sequence[str]) -> frozenset[str]:
        """The given ids that are terminal, live or tombstoned."""
        ...

    def api_compaction_candidates(self) -> list[str]:
        """Terminal API admissions carrying an inline image beside committed media, or a client history outside ``responses:``."""
        ...

    def history_mentions(self, references: Sequence[Mapping[str, Any]]) -> tuple[int, frozenset[str]]:
        """One read: the highest ``messages.id`` and the reference paths a transcript row mentions by their ``<sha256>/<name>`` tail, raw or JSON-escaped."""
        ...

    def settled_media(self) -> list[Sequence[Mapping[str, Any]]]:
        """The ``attachments_v1`` and ``native_text_v1`` media lists of terminal admissions, one list per row."""
        ...

    def retired_candidates(self, kind: Literal['media', 'spillover']) -> list[tuple[str, Any]]:
        """``(key, decoded value)`` of the durable retirement candidates of ``kind`` in key order."""
        ...
