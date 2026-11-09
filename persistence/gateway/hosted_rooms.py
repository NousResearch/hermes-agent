"""Hosted Bot Mode rooms: owners, transport bindings, rooms and their event log, peer links and grants, replicas, room drivers, policy checkpoints and attachments.

The hosted-room records live in two stores of one install. ``placement`` (:data:`HostedPlacement`) chooses between them on every verb that acts on those records: ``'profile'`` is the bound profile's own store, where the canonical hosted service keeps them; ``'shared'`` is the install-wide store, found from the bound profile's folder (the install root for a profile under ``<root>/profiles/``, else the profile's own folder), so every profile of one install reaches the same shared records. The owner claims, transport bindings, the two member-session verbs and the two member-home verbs act on the bound profile's store and take no placement; so do the hand-off verbs, which act on the install-wide store (``import_fenced_rooms`` copies from it into the bound profile's store).

The pure validators (identifiers, names, actors, payload bounds, page shapes) run in the gateway before each call; a verb's transaction starts at its first read. ``now=None`` reads the wall clock. Writes are one transaction each (10 s patience); a read is one read context. Errors are the hosted-room error classes below, with their messages and ``reason`` attributes; each joins the core class its condition belongs to, so ``except NotFound`` and ``except ValueError`` both catch a missing room.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

from ..authority.vocabulary import OwnerRule
from ..core.errors import Conflict, NotFound, PersistenceError, StoreUnavailable
from ..core.identity import JsonValue

__all__ = [
    "MAX_OWED_REVOKES",
    "MAX_OWED_REVOKES_PER_MEMBER",
    "MAX_OWED_REVOKE_SECONDS",
    "TASK_STATUSES",
    "TERMINAL_STATUSES",
    "AttachmentConflictError",
    "AttachmentData",
    "AttachmentError",
    "AttachmentIntegrityError",
    "AttachmentNotFoundError",
    "AttachmentQuotaError",
    "AuthorityConflictError",
    "AuthorityHistoryError",
    "AuthoritySupersededError",
    "DriverLease",
    "DriverStateError",
    "DriverValidationError",
    "EventAttachmentConflictError",
    "EventConflictError",
    "EventCursorConflictError",
    "HostedMemberGuard",
    "HostedOwnerClaim",
    "HostedPlacement",
    "HostedRoomAttachmentRepository",
    "HostedRoomDriverRepository",
    "HostedRoomError",
    "HostedRoomPeerRepository",
    "HostedRoomPolicyRepository",
    "HostedRoomRefusal",
    "HostedRoomReplicaRepository",
    "HostedRoomRepository",
    "HostedTaskIdentity",
    "InvalidTaskTransitionError",
    "LeaseHeldError",
    "LeaseRelease",
    "OwedRevokesFullError",
    "PolicyProjectionError",
    "PolicySnapshot",
    "RemoteRunIdentity",
    "RemoteRunReceipt",
    "ReplicaEpochRegressionError",
    "ReplicaError",
    "ReplicaGapError",
    "ReplicaPage",
    "RoomConflictError",
    "RoomGrantClaims",
    "RoomHeldGrantsPendingError",
    "RoomHistoryExpiredError",
    "RoomLinkRecord",
    "RoomNotFoundError",
    "RoomProbeUnavailableError",
    "RoomRecovery",
    "RoomStopPendingError",
    "RoomUnavailableError",
    "StaleLeaseError",
    "StaleTaskError",
    "TaskAttempt",
    "TaskConflictError",
    "TaskStatus",
    "TerminalStatus",
    "grant_digest",
    "room_grant_hard_expiry",
]

# The store holding the hosted-room records a verb acts on: the bound profile's own store or the install-wide store.
HostedPlacement = Literal["profile", "shared"]

# Room-driver task statuses.
TaskStatus = Literal["queued", "running", "settled", "failed", "cancelled", "indeterminate", "deferred", "stopping"]
TerminalStatus = Literal["settled", "failed"]
TASK_STATUSES: frozenset[str] = frozenset(get_args(TaskStatus))
TERMINAL_STATUSES: frozenset[str] = frozenset({"settled", "failed", "cancelled"})

# The shape and length bound of a room-driver identifier.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MAX_IDENTIFIER_CHARS = 128

# Owed revokes are bounded however many grants a caller hands in: an expiry is read from the grant's unsigned payload, so it is capped at the longest status horizon a member gateway signs; a member keeps at most a few rows; and the table has a ceiling, where expired rows make room first. A grant that finds no room is revoked by the caller at once: only the member's gateway can check a grant's signature, so an owed row's unsigned expiry never counts as covering another grant, nor is a row evicted for one. A revoke that never lands is retried, with capped backoff, until its grant expires: the grant passes at its member's gateway until then.
MAX_OWED_REVOKE_SECONDS = 30 * 24 * 60 * 60
MAX_OWED_REVOKES_PER_MEMBER = 4
MAX_OWED_REVOKES = 512


def grant_digest(grant: str) -> str:
    """The non-secret key an owed revoke is stored and logged under: the grant's SHA-256 hex digest."""
    return hashlib.sha256(grant.encode("utf-8")).hexdigest()


def room_grant_hard_expiry(grant: str, *, now: float | None = None) -> float:
    """When a room grant and every renewal of it are dead (its status horizon), read from its unsigned payload without the secret: the payload's ``status_expires_at``, else its ``expires_at``. An unreadable grant counts as living the longest horizon, ``now`` (the wall clock when None) plus :data:`MAX_OWED_REVOKE_SECONDS`."""
    try:
        encoded, separator, signature = grant.partition(".")
        if not separator:
            raise ValueError("room grant is invalid")
        payload, _ = (base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)) for part in (encoded, signature))
        claims = json.loads(payload.decode("ascii"))
        return float(claims.get("status_expires_at", claims["expires_at"]))
    except (ValueError, KeyError, TypeError, AttributeError):  # base64 and JSON errors are ValueErrors
        return (time.time() if now is None else now) + MAX_OWED_REVOKE_SECONDS


class _OneMessage:
    """A one-message constructor for an error whose core base takes structured arguments (``NotFound``, ``Conflict``)."""

    def __init__(self, message: str = "") -> None:
        Exception.__init__(self, message)


class HostedRoomError(PersistenceError, ValueError):
    """Base class for invalid or conflicting hosted-room operations."""


class HostedRoomRefusal(PersistenceError):
    """A hosted-room write the stored rows refuse: a room member session whose existing row or title belongs to something else."""


class RoomNotFoundError(_OneMessage, HostedRoomError, NotFound):
    """Raised when a room does not exist or has been disbanded."""


class RoomHistoryExpiredError(RoomNotFoundError):
    """Raised when a retired room remains reserved after history compaction."""

    reason = "room_history_expired"


class RoomConflictError(_OneMessage, HostedRoomError, Conflict):
    """Raised when an idempotency key is reused for different room state."""


class RoomHeldGrantsPendingError(HostedRoomError):
    """Retry Disband after revoking grants received since its last sweep."""

    reason = "room_disbanding_retry"


class RoomStopPendingError(HostedRoomError, RuntimeError):
    """Expected Stop refusal; also a ``RuntimeError`` for direct service callers."""

    reason = "room_stop_pending"


class OwedRevokesFullError(HostedRoomError):
    """No room is left to record an owed revoke, while the grant it names may still pass."""


class RoomProbeUnavailableError(HostedRoomError, StoreUnavailable):
    """Raised when a non-blocking ownership probe cannot read the room store."""


class EventConflictError(_OneMessage, HostedRoomError, Conflict):
    """Raised when an event id is reused with different immutable content."""


class EventCursorConflictError(_OneMessage, HostedRoomError, Conflict):
    """Raised when new room events invalidate an uncommitted publication plan."""


class EventAttachmentConflictError(EventCursorConflictError):
    """A publication lost its file commitments; use the same fresh-plan retry."""


class AuthorityConflictError(_OneMessage, HostedRoomError, Conflict):
    """Raised when a stale room authority attempts to mutate hosted state."""

    reason = "authority_conflict"


class AuthoritySupersededError(AuthorityConflictError):
    """Raised when a successful authority claim was later superseded."""


class ReplicaError(HostedRoomError):
    """Base class for invalid or conflicting replica operations."""


class ReplicaGapError(_OneMessage, ReplicaError, Conflict):
    """A page does not start at the replica's next expected sequence."""


class ReplicaEpochRegressionError(_OneMessage, ReplicaError, Conflict):
    """A page or demotion carries an older authority epoch than stored."""


class AuthorityHistoryError(PersistenceError, ValueError):
    """The recorded authority sequence cannot explain the current room; escapes the work-record capture of a driver transition or a remote-run receipt."""


class DriverStateError(PersistenceError, ValueError):
    """Base class for invalid or conflicting driver-state operations."""


class DriverValidationError(DriverStateError):
    """Raised when an identifier, clock, TTL, or payload is invalid."""


class RoomUnavailableError(_OneMessage, DriverStateError, NotFound):
    """Raised when the hosted room does not exist or was disbanded."""


class LeaseHeldError(_OneMessage, DriverStateError, Conflict):
    """Raised when another unexpired driver generation owns the room."""


class StaleLeaseError(_OneMessage, DriverStateError, Conflict):
    """Raised when a lease generation can no longer mutate room state."""


class TaskConflictError(_OneMessage, DriverStateError, Conflict):
    """Raised when an idempotency key is reused for different task state."""


class StaleTaskError(_OneMessage, DriverStateError, Conflict):
    """Raised when an obsolete task attempt or cancellation tries to commit."""


class InvalidTaskTransitionError(_OneMessage, DriverStateError, Conflict):
    """Raised when a requested task transition is not allowed."""


class PolicyProjectionError(PersistenceError, RuntimeError):
    """The policy checkpoint's integrity and bound failures (also a ``RuntimeError``)."""


class AttachmentError(PersistenceError, ValueError):
    """Base class for hosted-room attachment contract failures."""


class AttachmentNotFoundError(_OneMessage, AttachmentError, NotFound):
    """The attachment id is unknown, expired, or unavailable to this caller."""


class AttachmentConflictError(_OneMessage, AttachmentError, Conflict):
    """An idempotency or ownership key was reused with different content."""


class AttachmentQuotaError(AttachmentError):
    """A bounded room or gateway quota would be exceeded."""


class AttachmentIntegrityError(AttachmentError):
    """Canonical blob bytes no longer match their durable metadata."""


def _task_identifier(value: Any, *, label: str) -> str:
    """A room-driver identifier: a stripped string of at most 128 characters matching the identifier pattern."""
    if not isinstance(value, str):
        raise DriverValidationError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > _MAX_IDENTIFIER_CHARS or not _IDENTIFIER_RE.fullmatch(value):
        raise DriverValidationError(f"invalid {label}")
    return value


@dataclass(frozen=True, slots=True)
class HostedTaskIdentity:
    """Stable identity for one admitted room turn, each field checked and stripped."""

    room_id: str
    task_id: str
    thread_id: str
    turn_id: str

    def __post_init__(self) -> None:
        for field in ("room_id", "task_id", "thread_id", "turn_id"):
            object.__setattr__(self, field, _task_identifier(getattr(self, field), label=field))


@dataclass(frozen=True, slots=True)
class HostedOwnerClaim:
    """The owner claim ``create_room`` runs first in its transaction: the authority epoch, the claiming actor and the owner rule a stored owner is read by."""

    epoch: int
    actor: str
    rule: OwnerRule | None


@dataclass(frozen=True, slots=True)
class HostedMemberGuard:
    """The hosted member check ``admit(hosted=)`` and ``member_authorized`` run inside their transaction; ``home_current`` is the gateway's answer for the member's home, read before the call."""

    room_id: str
    owner: str
    gateway_id: str
    authority_epoch: int
    member_id: str
    profile: str
    home_current: bool
    task: HostedTaskIdentity | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class RoomLinkRecord:
    """One ``hosted_room_links`` row, built by the gateway's ``StoredRoomLink.as_record()``; ``grant`` is a secret and stays out of ``repr``."""

    room_id: str
    member_id: str
    target_url: str
    target_profile: str
    grant: str = dataclass_field(repr=False)
    catalog_json: str
    cancellation_scope_id: str
    trace_id: str
    transport_security: str
    status: str
    updated_at: float


@dataclass(frozen=True, slots=True)
class RoomGrantClaims:
    """The claim keys read from a decoded room grant, coerced by the caller."""

    room_id: str
    home_install_id: str
    authority_gateway_id: str
    authority_epoch: int
    member_id: str
    target_install_id: str
    target_profile: str
    issued_at: float


@dataclass(frozen=True, slots=True)
class RemoteRunIdentity:
    """The nine identity columns of a remote-run receipt."""

    room_id: str
    home_install_id: str
    authority_gateway_id: str
    authority_epoch: int
    member_id: str
    target_install_id: str
    target_profile: str
    task_id: str
    execution_generation: int


@dataclass(frozen=True, slots=True)
class RemoteRunReceipt:
    """One logical peer task attempt bound to its remote run handle: the identity columns plus ``run_id`` and ``session_id``."""

    room_id: str
    home_install_id: str
    authority_gateway_id: str
    authority_epoch: int
    member_id: str
    target_install_id: str
    target_profile: str
    task_id: str
    execution_generation: int
    run_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ReplicaPage:
    """A validated replication page: the page authority, its events as received, and ``latest_seq`` when the page carries a non-bool int."""

    authority_gateway_id: str
    authority_epoch: int
    events: tuple[Mapping[str, JsonValue], ...]
    latest_seq: int | None


@dataclass(frozen=True, slots=True)
class DriverLease:
    """A fenced room-driver lease held by one gateway process incarnation."""

    room_id: str
    gateway_id: str
    authority_epoch: int
    process_generation: str
    lease_generation: int
    expires_at: float
    reclaimed: bool = False


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    """The exact running generation authorized to settle one task."""

    identity: HostedTaskIdentity
    lease: DriverLease
    execution_generation: int
    cancel_generation: int


@dataclass(frozen=True, slots=True)
class LeaseRelease:
    """The outcome of ``release_lease``."""

    lease: DriverLease
    idempotent: bool


@dataclass(frozen=True, slots=True)
class RoomRecovery:
    """The queued and indeterminate task identities ``recover_room`` reads, in task order."""

    queued: tuple[HostedTaskIdentity, ...]
    indeterminate: tuple[HostedTaskIdentity, ...]


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Bounded active policy input at one durable room-log cursor; watermark keys are ``(thread_id, member_id)``."""

    through_seq: int
    stopped_through_seq: int
    events: tuple[Mapping[str, JsonValue], ...]
    watermarks: Mapping[tuple[str, str], int]


@dataclass(frozen=True, slots=True)
class AttachmentData:
    """Verified canonical bytes plus safe attachment metadata."""

    attachment: Mapping[str, JsonValue]
    data: bytes


@runtime_checkable
class HostedRoomRepository(Protocol):
    """Hosted room owners, transport bindings, member sessions, rooms and the room event log; the runtime check is an assembly diagnostic only, never contract coverage.

    Room ids, event ids, names, kinds, actors and payload bounds are validated by the caller before the call, except ``create_room``, which validates its payload first, and ``save_room_settings``, which validates the settings record it resolves.
    """

    def claim_owner(self, *, epoch: int, room_id: str, actor: str, owner: OwnerRule | None) -> None:
        """One write on the bound profile's store: the epoch fence (:class:`persistence.FenceStale`), then the stored owner must belong to ``actor`` under ``owner`` (``session_rules.legacy_local.same_owner``; without a rule only the equal stamp), else ``AuthorityRefusal('permission_denied')``. Writes nothing."""
        ...

    def create_room(self, *, placement: HostedPlacement, room_id: str, name: str, members: Sequence[Mapping[str, JsonValue]], authority_gateway_id: str, owner: HostedOwnerClaim | None = None, now: float | None = None) -> Mapping[str, JsonValue]:
        """Create a room, or return the identical existing room idempotently, in one write.

        Order: payload validation (``HostedRoomError``); with ``owner`` (``placement='profile'`` only), the owner claim in the same transaction (epoch fence; an absent owner key is inserted when neither a room nor a retired id holds ``room_id``, else the stored owner must belong to ``owner.actor`` under ``owner.rule``, as ``claim_owner`` reads it, or ``AuthorityRefusal('permission_denied')``); a retired id or a disbanded row refuses ``RoomConflictError('room_id belongs to a disbanded room')``; an existing room returns with ``idempotent=True`` when name and members match (a ``'legacy'`` room is adopted at the next epoch with an ``authority.claimed`` event, ``adopted`` and ``claim_event``), else ``RoomConflictError``; the active-room cap (256); the insert. The claim and the room commit or roll back together.
        """
        ...

    def bind_transport(self, *, epoch: int, session_id: str, binding: str) -> None:
        """One write on the bound profile's store: the epoch fence, then insert-if-absent of the caller-encoded ``binding`` for ``session_id``; a different stored text (byte-exact) refuses ``AuthorityRefusal('permission_denied')`` and an identical one is a no-op."""
        ...

    def label_room_session(self, *, epoch: int, session_id: str, title: str) -> None:
        """One write: the epoch fence, then the session row's title becomes ``title`` verbatim and the row is hidden; a missing row writes nothing; a title another session holds refuses :class:`persistence.TitleConflict`."""
        ...

    def ensure_room_member_session(self, *, session_id: str, title: str) -> str:
        """One write: an existing row must carry ``title`` and source ``bot_room``, else :class:`HostedRoomRefusal`; otherwise the sanitized title held by another session refuses :class:`HostedRoomRefusal`, else a hidden ``bot_room`` row is inserted. Returns ``session_id``."""
        ...

    def rename_room(self, *, placement: HostedPlacement, room_id: str, event_id: str, name: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """Rename a live room and append its ``room.renamed`` event in one write. A missing or disbanded room refuses ``RoomNotFoundError`` (``RoomHistoryExpiredError`` for a retired id with no row); an existing ``event_id`` replays with ``idempotent=True`` when it is the same rename, else ``EventConflictError``. Returns the room with ``latest_seq`` and ``event``."""
        ...

    def append_event(self, *, placement: HostedPlacement, room_id: str, event_id: str, kind: str, actor: Mapping[str, str], payload: Mapping[str, JsonValue], authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None = None, control_reserve: bool = False, now: float | None = None) -> Mapping[str, JsonValue]:
        """Append one immutable event at the room's next sequence in one write.

        Order: a stored ``event_id`` replays with ``idempotent=True`` when kind, actor, epoch and payload equal it (even after the room moved or disbanded), else ``EventConflictError``; the live room (``RoomNotFoundError``, ``RoomHistoryExpiredError``); the authority (``AuthorityConflictError('stale hosted room authority')``); ``expected_latest_seq`` (``EventCursorConflictError``); message attachments; the room and gateway capacity (``HostedRoomError``; control kinds use the reserve, and so does a ``gateway`` actor's event with ``control_reserve``, so a room whose log is full can still say why); insert and sequence advance. Returns the event with ``actor`` as given.
        """
        ...

    def save_room_settings(self, *, placement: HostedPlacement, room_id: str, event_id: str, purpose: str, notes: str, routing: str | None, lead_member_id: str | None, seated_member_ids: Collection[str], authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Append the room's settings as one ``room.context_set`` event by the system actor ``room-control``, resolving what the save omits inside the same write.

        Order: an omitted (None) ``routing`` or ``lead_member_id`` takes its value from the settings event stored under ``event_id`` (a retried save), else from the room's newest settings event, else ``everyone`` and no lead; ``session_rules.hosted_rooms.validate_context`` normalizes the resolved record (``HostedRoomError``); with no event stored under ``event_id``, ``lead`` routing whose lead is not in ``seated_member_ids`` refuses ``HostedRoomError('the lead must be a member of this Group Chat')``; then ``append_event``'s steps with the record as payload, where a retried save is compared in its stored event's own fields (``session_rules.hosted_rooms.retry_fields``: a field that event lacks counts only off its default). Returns ``{context, event}``: the resolved record and the event.
        """
        ...

    def disband_room(self, *, placement: HostedPlacement, room_id: str, expected_gateway_id: str, expected_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Tombstone a room permanently and idempotently in one write: replays bypass the fence (a retired id with no row returns ``history_expired``; a disbanded row returns its tombstone and stored event); then the authority; a held owed revoke of the room expiring after ``now`` refuses ``RoomHeldGrantsPendingError('Group chat grants still need revoking; retry Disband.')``, and the room's expired held rows are deleted; then the ``room.disbanded`` event, the fenced update, the retired id, the room's recorded member and attempt homes forgotten (``record_member_home``, ``record_attempt_home``; only the profile placement holds them) and the retention prune. Returns ``{room_id, disbanded_at, idempotent, event}``."""
        ...

    def claim_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str, expires_at: float, now: float | None = None) -> bool:
        """Take the one router-call claim of ``(room_id, discussion_event_id)`` for ``owner`` until ``expires_at``, in one write: inserted when absent, taken over when the stored claim expired at ``now``, else left to its holder. True when ``owner`` holds the claim after the write. Claims are liveness state: a hand-off or a pre-isolation import never copies them."""
        ...

    def release_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str) -> None:
        """Delete the claim of ``(room_id, discussion_event_id)`` in one write, only while ``owner`` holds it."""
        ...

    def record_member_home(self, *, room_id: str, member_id: str, home: str) -> None:
        """One write on the bound profile's store: ``home`` joins the member's recorded homes, after those already recorded, unless it is one of them. ``disband_room`` with ``placement='profile'`` forgets the room's recorded homes."""
        ...

    def member_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        """The member's recorded homes in the bound profile's store, in the order they were first recorded; ``()`` when none."""
        ...

    def record_attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int, home: str) -> None:
        """One write on the bound profile's store: first drop this member's attempt homes whose attempt is over (the driver task row on the profile placement is gone, at another execution generation, or outside ``running``, ``stopping``, ``deferred`` and ``indeterminate``), then record ``home`` for the attempt ``(task_id, generation)`` unless one is recorded (the first home wins). ``disband_room`` with ``placement='profile'`` forgets them with the member homes."""
        ...

    def attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int) -> str | None:
        """The home recorded for the attempt ``(task_id, generation)`` of this member, whatever its task's status, or None."""
        ...

    def stopping_attempt_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        """In one read, the homes recorded for this member's attempts whose driver task row on the profile placement is still that attempt (same execution generation) in ``stopping`` or ``deferred``; each may still answer for its Stop."""
        ...

    def prune_disbanded_rooms(self, *, placement: HostedPlacement, now: float | None = None) -> int:
        """Purge disbanded rooms past the 90-day retention or beyond the newest 512 tombstones, with their dependent rows, keeping their retired ids. Returns the purged room count."""
        ...

    def owner(self, room_id: str) -> str | None:
        """The stored owner of ``room_id`` in the bound profile's store, or None."""
        ...

    def owned_rooms(self) -> frozenset[str]:
        """Every room id with a stored owner in the bound profile's store."""
        ...

    def probe_room_owner(self, room_id: str) -> bool:
        """Whether the bound profile's store holds an owner for ``room_id``, without creating, migrating or waiting: False when that store, or the part of it that keeps owners, does not exist yet; a store failure raises ``RoomProbeUnavailableError('native room ownership is temporarily unavailable')``."""
        ...

    def transport_binding(self, session_id: str) -> str | None:
        """The stored transport binding text for ``session_id``, or None."""
        ...

    def member_authorized(self, guard: HostedMemberGuard, *, epoch: int, operation: str) -> bool:
        """The hosted member check as one read on the bound profile's store: the epoch fence (``FenceStale``), then owner key, live room, room authority, seated local member, ``guard.home_current``, then, with ``guard.task``, the task row's identity, generation, target and a status ``operation`` allows. False on any mismatch."""
        ...

    def list_rooms(self, *, placement: HostedPlacement, include_disbanded: bool = False, limit: int = 500, offset: int = 0) -> list[Mapping[str, JsonValue]]:
        """One page of rooms, live only unless ``include_disbanded``, ordered by ``updated_at`` descending then ``room_id``; each with ``latest_seq``."""
        ...

    def room_state(self, *, placement: HostedPlacement, room_id: str, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        """One room with ``latest_seq`` and, when one exists, ``authority_claim``: the newest ``authority.claimed`` event at the room's epoch. ``RoomNotFoundError`` or ``RoomHistoryExpiredError``."""
        ...

    def read_events(self, *, placement: HostedPlacement, room_id: str, since_seq: int = 0, limit: int = 100, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        """A monotonic log page after ``since_seq`` within the 2 MiB page budget: ``{events, cursor, latest_seq, has_more, authority}``. ``since_seq`` past the log refuses ``HostedRoomError('since_seq is ahead of the hosted room log')``."""
        ...

    def probe_hosted_room(self, *, placement: HostedPlacement, room_id: str) -> bool:
        """Whether a live room exists, without creating, migrating or waiting: False when the placement's store, or the part of it that keeps rooms, does not exist yet; a store failure raises ``RoomProbeUnavailableError('hosted room ownership is temporarily unavailable')``."""
        ...

    def latest_event_payload(self, *, placement: HostedPlacement, room_id: str, kind: str, before_seq: int | None = None) -> Mapping[str, JsonValue] | None:
        """The payload of the room's newest event of ``kind``, of the newest below ``before_seq`` when given, or None. One read of the event log that reads no room row, so a room the placement's store does not hold has none."""
        ...

    def event_by_id(self, *, placement: HostedPlacement, room_id: str, event_id: str) -> Mapping[str, JsonValue] | None:
        """The event stored under ``event_id`` in the room, live or disbanded, or None."""
        ...

    def fenced_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        """The live rooms of the install-wide store whose authority is ``handoff_gateway_id`` and whose id the hand-off record does not hold."""
        ...

    def record_room_handoff(self, *, room_ids: Collection[str], now: float | None = None) -> None:
        """One write on the install-wide store: each id joins the hand-off record at ``now``; an id already recorded keeps its first time."""
        ...

    def handed_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        """Every room of the install-wide store, live or disbanded, whose authority is ``handoff_gateway_id``, recorded or not."""
        ...

    def import_fenced_rooms(self, *, epoch: int | None, room_ids: Collection[str], owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float | None = None) -> int:
        """Copy the ``room_ids`` the bound profile's store has never known from the install-wide store and claim them, in one write on the profile's store; returns the rooms copied.

        Order: with ``epoch``, the epoch fence (``FenceStale``); the ids with neither a room row nor a retired id in the profile's store are copied with every row of theirs the install-wide store holds, except driver leases and rows of no room (a row the profile's store already holds fails the whole write with ``PermanentStoreError``); then each copied room is claimed: an ``authority.claimed`` event at the room's next sequence and epoch + 1, naming ``handoff_gateway_id`` as the previous and ``local_gateway_id`` as the new authority, the room moves to that authority and epoch with its sequence and revision advanced and ``updated_at`` at ``now``, and its owner key becomes ``owner`` unless one is stored. The install-wide store is only read.
        """
        ...


@runtime_checkable
class HostedRoomPeerRepository(Protocol):
    """Peer room links, grant revocations, peer reservations and remote-run receipts. The caller validates identifiers, grant scopes and link records before the call."""

    def upsert_room_link_record(self, *, placement: HostedPlacement, record: RoomLinkRecord, max_links: int, hold_replaced: bool = False) -> None:
        """Insert or replace the ``(room_id, member_id)`` link in one write; a new key when ``max_links`` links are stored refuses ``HostedRoomError('too many stored room links')``. With ``hold_replaced``, a replaced grant whose :func:`room_grant_hard_expiry` is later than the new grant's is kept, in the same write, as a held owed revoke at that expiry (``owe_room_grant_revoke``'s bounds; a full member refuses ``HostedRoomError('too many held group chat grant revokes')``). Then a held owed revoke of the new grant, if any, is dropped. The link carries a grant, so on POSIX the placement's file is then made owner-only (``0600``); a failed mode change is ignored."""
        ...

    def retire_room_link_records(self, *, placement: HostedPlacement, room_id: str, owe_routes: bool = False, revoked_digests: Collection[str] | None = None, now: float | None = None) -> int:
        """Drop one room's links in one write and release its held owed revokes; returns the links dropped. With ``owe_routes``, each link's grant first becomes an owed revoke until its :func:`room_grant_hard_expiry` (``owe_room_grant_revoke``'s bounds and refusals) and the room's held rows become due at ``now``. Without it the caller already revoked every link and held grant, so the room's held rows are deleted: only those whose ``grant_digest`` is in ``revoked_digests`` when it is given."""
        ...

    def owe_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        """Record in one write a revoke owed for a grant, keyed ``(room_id, member_id, grant_digest(grant))``, due at ``now``, expiring at ``expires_at`` capped at ``now + MAX_OWED_REVOKE_SECONDS``. A key already recorded is kept as it is. A member holding ``MAX_OWED_REVOKES_PER_MEMBER`` rows refuses ``OwedRevokesFullError('too many owed revokes for this group chat member')``; at ``MAX_OWED_REVOKES`` rows the expired rows are deleted first, and a table still full refuses ``OwedRevokesFullError('too many owed room grant revokes')``."""
        ...

    def hold_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        """Hold a grant until Disband commits, in one write: the room row (``RoomNotFoundError``, ``RoomHistoryExpiredError``), then ``owe_room_grant_revoke``'s row, held while the room is live and owed at once after its tombstone. On a live room either bound refuses ``HostedRoomError('too many held group chat grant revokes')``; on a disbanded room the bounds refuse as ``owe_room_grant_revoke`` does."""
        ...

    def settle_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, grant_digest: str, retry_at: float | None = None) -> None:
        """Forget one owed revoke (``retry_at`` None), else count an attempt and make it due at ``retry_at``; one write."""
        ...

    def drop_remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity, run_id: str) -> bool:
        """Delete one attempt's receipt only while it still names ``run_id``, in one write; when a row went, the work-record capture runs in the same transaction. True when a receipt was dropped."""
        ...

    def update_room_link_status(self, *, placement: HostedPlacement, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
        """Set a link's status and ``updated_at``; True when the link exists."""
        ...

    def delete_room_link_records(self, *, placement: HostedPlacement, room_id: str) -> int:
        """Delete every link of the room; returns the deleted count."""
        ...

    def revoke_room_grant_scope(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        """Revoke every grant issued at or before ``now`` for one exact room scope. Nothing is written when ``expires_at <= now``; else one write: drop expired revocations, upsert the scope's row (``expires_at`` and ``revoked_before`` only grow), revoke the matching peer reservation."""
        ...

    def reserve_peer_room(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        """Reserve a peer room for the claim's authority. ``expires_at <= now`` refuses ``HostedRoomError('peer room reservation must expire in the future')`` before any write; then one write: drop expired reservations, refuse ``AuthorityConflictError('peer room reservation authority changed')`` when a live reservation of the room and profile or the member's own row outranks the claim, revoke lower-epoch reservations, upsert the row."""
        ...

    def upsert_remote_run_receipt(self, *, placement: HostedPlacement, receipt: RemoteRunReceipt, now: float | None = None) -> None:
        """Bind one logical peer task attempt to its remote run in one write: an existing receipt with another ``run_id`` or ``session_id`` refuses ``HostedRoomError('remote run receipt conflicts with its logical task')``, an identical one touches ``updated_at``; a new one is inserted and the work-record capture runs in the same transaction, whose ``AuthorityHistoryError`` and ``HostedRoomError`` escape and roll the receipt back."""
        ...

    def list_room_link_records(self, *, placement: HostedPlacement) -> Sequence[Mapping[str, JsonValue]]:
        """Every link as a dict, ordered by ``room_id, member_id``."""
        ...

    def room_grant_is_revoked(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        """Whether the scope has an unexpired revocation at or after ``claims.issued_at``."""
        ...

    def peer_room_grant_is_current(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        """Whether a live, unrevoked reservation matches the claim's room, member, profile and authority."""
        ...

    def probe_peer_room_reservation(self, *, placement: HostedPlacement, room_id: str, target_profile: str, now: float | None = None) -> bool:
        """Whether a live unrevoked reservation of the room and profile exists, without creating, migrating or waiting; a store failure raises ``RoomProbeUnavailableError('peer room ownership is temporarily unavailable')``."""
        ...

    def remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity) -> Mapping[str, JsonValue] | None:
        """The receipt for one task attempt as a dict, or None."""
        ...

    def room_has_peer_obligations(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> bool:
        """Whether the room still has a link or an owed revoke, held or not, expiring after ``now``: a metadata-only Disband then needs a runtime to settle peer access. One read."""
        ...

    def owes_room_grant_revokes(self, *, placement: HostedPlacement) -> bool:
        """Whether any owed revoke, held or not, is recorded. One read."""
        ...

    def due_room_grant_revokes(self, *, placement: HostedPlacement, limit: int, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        """Up to ``limit`` unheld owed revokes due at ``now`` and unexpired, by ``next_attempt_at`` then ``created_at``, each as ``{room_id, member_id, grant_digest, target_url, grant, attempts}``, read in one read context. Only when an expired row exists does a second, write transaction delete every row expired at ``now``, so an idle caller takes no write lock."""
        ...

    def held_room_grant_revokes(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        """In one write: delete the room's held rows expired at ``now``, then return the rest as ``{target_url, grant}``: the retained grants Disband must revoke before it can forget them."""
        ...


@runtime_checkable
class HostedRoomReplicaRepository(Protocol):
    """Replicated room logs and the authority takeover primitives. The caller validates the room, page and reason and resolves ``local_gateway_id`` before the call."""

    def ingest_page(self, *, placement: HostedPlacement, room_id: str, room_name: str, members_json: str, page: ReplicaPage, now: float | None = None) -> Mapping[str, JsonValue]:
        """Store one page idempotently in one write: the replica cap (``ReplicaError('replica room capacity exhausted')``), epoch regression (``ReplicaEpochRegressionError``), a gap (``ReplicaGapError('page skips sequences the replica has not stored')``), the 256 MiB event budget, then the events and the replica row. Returns ``{room_id, stored_seq, ingested, authority, caught_up}``."""
        ...

    def promote_replica(self, *, placement: HostedPlacement, room_id: str, local_gateway_id: str, reason: str = 'authority-unreachable', now: float | None = None) -> Mapping[str, JsonValue]:
        """Continue a replicated room on this gateway at ``epoch + 1`` in one write: the replica (``ReplicaError('replica not found')``), not already ours, no live or retired room of that id (``RoomConflictError``), then the room row, the copied log and the ``authority.claimed`` event; the replica is dropped."""
        ...

    def demote_room(self, *, placement: HostedPlacement, room_id: str, observed_gateway_id: str, observed_epoch: int, local_gateway_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """Fence this gateway's stale room authority against a newer observed epoch in one write: the same lineage replays with ``idempotent=True``; an older epoch refuses ``ReplicaEpochRegressionError``; a room this gateway does not hold refuses ``ReplicaError``; else the ``authority.lost`` event and the room's new authority."""
        ...

    def replica_state(self, *, placement: HostedPlacement, room_id: str) -> Mapping[str, JsonValue]:
        """The stored replica's coverage and authority, else ``ReplicaError('replica not found')``."""
        ...


@runtime_checkable
class HostedRoomDriverRepository(Protocol):
    """Room driver leases and tasks. The caller runs the driver's validators first (``DriverValidationError``). Every write that changes a task runs the work-record capture after its update in the same transaction. Task dicts are the driver's task view."""

    def acquire_lease(self, *, placement: HostedPlacement, room_id: str, gateway_id: str, authority_epoch: int, process_generation: str, ttl_seconds: float, now: float | None = None) -> DriverLease:
        """Acquire, extend or take over the room's driver lease; ``RoomUnavailableError``, ``StaleLeaseError``, ``LeaseHeldError``."""
        ...

    def renew_lease(self, lease: DriverLease, *, placement: HostedPlacement, ttl_seconds: float, now: float | None = None) -> DriverLease:
        """Extend an active lease; ``StaleLeaseError('driver lease changed during renewal')``."""
        ...

    def release_lease(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> LeaseRelease:
        """Release an idle lease; an already released lease replays with ``idempotent=True``."""
        ...

    def admit_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, payload: Mapping[str, JsonValue], now: float | None = None) -> Mapping[str, JsonValue]:
        """Admit a queued task for a live room; an identical task replays, a different one refuses ``TaskConflictError``."""
        ...

    def start_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_cancel_generation: int, now: float | None = None) -> TaskAttempt:
        """Start the room's next queued task under an active lease."""
        ...

    def settle_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        """Settle a running attempt; the settlement replays before the lease check."""
        ...

    def settle_stopping_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        """Settle a stopping task that completed before its stop."""
        ...

    def resolve_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        """Settle an indeterminate task from reconciled evidence."""
        ...

    def resolve_indeterminate_cancellation(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, cancel_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """Cancel an indeterminate task with a cancellation proof."""
        ...

    def requeue_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Return an indeterminate task to the queue."""
        ...

    def defer_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, reason: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """Defer an indeterminate task as retryable."""
        ...

    def defer_unconfirmed_stop(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, reason: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """Defer a stopping task its member never confirmed stopped, as retryable, never cancelled: lease first, then the task must be ``stopping`` at both generations (``StaleTaskError('stopping task generation changed')``); the same deferral replays with ``idempotent=True``."""
        ...

    def requeue_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Return a deferred task to the queue."""
        ...

    def reopen_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Return a deferred task to ``indeterminate`` at its same generations (its peer still owns that generation): lease first, then ``deferred`` at both generations (``StaleTaskError('deferred task generation changed')``); the result and terminal time clear and ``started_at`` and ``indeterminate_at`` become ``now``, so the attempt's time limit counts from the retry."""
        ...

    def mark_attempt_indeterminate(self, attempt: TaskAttempt, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> None:
        """Mark exactly ``attempt`` indeterminate in one write, under ``lease`` (an active lease of the room, which need not be the one the attempt ran under): the lease check (``StaleLeaseError``), then a task no longer ``running`` at the attempt's generations is left as it is; else ``status='indeterminate'`` with ``indeterminate_at`` at ``now``, and the work-record capture."""
        ...

    def requeue_not_admitted_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, now: float | None = None) -> Mapping[str, JsonValue]:
        """Return a running attempt the member never admitted to the queue."""
        ...

    def cancel_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Cancel a queued or deferred task directly."""
        ...

    def begin_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Request the two-phase stop of running work."""
        ...

    def complete_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        """Complete an acknowledged stop."""
        ...

    def recover_room(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> RoomRecovery:
        """Mark running tasks of other leases indeterminate and read the queued and indeterminate identities."""
        ...

    def prune_published_terminal_tasks(self, *, placement: HostedPlacement, room_id: str, now: float | None = None, retention_seconds: float = 2592000.0, retain: int = 2048) -> int:
        """Delete published terminal tasks past the retention and count bounds; returns the deleted count."""
        ...

    def get_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue]:
        """One task view; ``TaskConflictError`` for a missing task, another turn or a stored payload that fails its check."""
        ...

    def list_tasks(self, *, placement: HostedPlacement, room_id: str, status: TaskStatus | None = None) -> tuple[Mapping[str, JsonValue], ...]:
        """The room's tasks in ``(source_event_seq, created_at, task_id)`` order, optionally of one status."""
        ...

    def get_task_for_turn(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue] | None:
        """The task bound to the identity's thread and turn, whatever its task id, or None."""
        ...


@runtime_checkable
class HostedRoomPolicyRepository(Protocol):
    """The room policy checkpoint: cursors, active policy events, threads, watermarks and terminal publication receipts.

    Every verb takes the caller's configured active-event budget ``max_active_events`` (default 64), a positive int, else ``ValueError('max_active_events must be a positive integer')`` before any read or write. The placement's store records the largest budget a verb gave it, never below 64, and every verb bounds a discussion's active events by that recorded capacity; the thread transcript bound follows it.
    """

    def snapshot(self, *, placement: HostedPlacement, room_id: str, latest_seq: int, max_active_events: int = 64) -> PolicySnapshot:
        """Sync the room's policy cursor through ``latest_seq``, then read the bounded snapshot; ``RoomNotFoundError`` and ``PolicyProjectionError``. A ``room.activity`` closes its thread only while the thread still belongs to the discussion it names; a ``room.continued`` reopens its discussion unless a newer user message on the thread came between the discussion's message and the Continue."""
        ...

    def compact_completed(self, *, placement: HostedPlacement, room_id: str, max_active_events: int = 64) -> None:
        """Drop the active policy events and rows of the room's completed threads in one write."""
        ...

    def publication_exists(self, *, placement: HostedPlacement, room_id: str, task_id: str, status: str, execution_generation: int, max_active_events: int = 64) -> bool:
        """Whether the task's terminal (or, for ``deferred``, this generation's deferral) publication is recorded."""
        ...

    def events_for_task(self, *, placement: HostedPlacement, room_id: str, source_event_seq: int, input_context: Mapping[str, JsonValue] | None = None, task_id: str | None = None, max_active_events: int = 64) -> tuple[Mapping[str, JsonValue], ...]:
        """The room events a task's policy projection reads, in sequence order; ``ValueError`` for a bad input context, ``PolicyProjectionError``."""
        ...


@runtime_checkable
class HostedRoomAttachmentRepository(Protocol):
    """Room attachment rows and their blobs, kept in ``hosted-room-attachments/`` beside the placement's file. The caller validates identifiers, manifests, content and ranges before the call (``AttachmentError``)."""

    def prepare(self, *, placement: HostedPlacement, now: float | None = None) -> None:
        """Run the store's attachment setup unless this process already ran it on that store: the attachment records, the reconciliation of the commitments their durable events name, and the prune of expired rows and their blobs. Every other verb runs it first too."""
        ...

    def put(self, *, placement: HostedPlacement, room_id: str, upload_id: str, kind: str, name: str, mime: str, data: bytes, now: float | None = None) -> Mapping[str, JsonValue]:
        """Prune expired uploads, then store one upload idempotently with its blob; ``AttachmentConflictError``, ``AttachmentQuotaError``, ``AttachmentIntegrityError``. Returns the attachment metadata."""
        ...

    def commit_message_with_receipt(self, *, placement: HostedPlacement, room_id: str, event_id: str, manifest: Sequence[Mapping[str, JsonValue]], recipient_member_ids: Sequence[str], viewer_access: bool = False, retention_seconds: float | None = None, hold_until_event: bool = False, now: float | None = None) -> tuple[list[Mapping[str, JsonValue]], tuple[str, ...]]:
        """Commit a message's uploads to its event and frozen recipient set in one write. Returns the normalized manifest and the ids this call moved from ``uploaded``."""
        ...

    def abort_message_commit(self, *, placement: HostedPlacement, room_id: str, event_id: str, attachment_ids: Sequence[str], now: float | None = None) -> int:
        """Return an unpublished commitment's rows to ``uploaded``; returns the changed count."""
        ...

    def read(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str | None, event_id: str | None = None, viewer: bool = False, now: float | None = None) -> AttachmentData:
        """A committed attachment's verified bytes for a recipient or viewer; ``AttachmentNotFoundError``, ``AttachmentIntegrityError``."""
        ...

    def describe(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str, event_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """A committed attachment's metadata for a recipient, without reading its bytes."""
        ...

    def read_range(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str, event_id: str, offset: int, length: int, now: float | None = None) -> AttachmentData:
        """One byte range of a committed attachment for a recipient; ``AttachmentError('attachment range starts past the end')``."""
        ...

    def read_viewer(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, event_id: str | None = None, recipient_member_id: str | None = None, authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> AttachmentData:
        """A viewer read fenced on the room's authority before and after the byte read; ``AttachmentNotFoundError``."""
        ...
