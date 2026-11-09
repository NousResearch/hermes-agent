"""Session storage ports: the session/message store and turn admission, as one plugin kind.

Two ports at the level today's call sites use, never SQL and never a connection:

- :class:`SessionStore`: session rows and their transcripts (create, read, append, end, titles,
  listing, deletion), the transcript rewrites (replace, rewind, in-place edits, reactions),
  compression publication (a durable child, or in-place compaction) and usage accounting.
- :class:`TurnAdmission`: the profile owner lock, the owner epoch and each session's FIFO of
  admitted inputs (admit, claim, steer, settle, cancel, resolve) whose claims issue the turn fences
  (``hermes_scope.Fence``), the receipt-idempotent runtime-session mutations, and the worker
  executions with their receipts.

A :class:`SessionStorageProvider` supplies both. This is the interface and its local backend only:
no production caller goes through it yet, and nothing in config.yaml selects a backend. Today's
callers still use ``SessionDB`` and ``hermes_state_runtime`` directly; moving them behind these
ports, call site by call site, is a follow-up, and until it lands a provider plugin stores
nothing that Hermes reads. :func:`resolve_session_storage` takes the backend's name explicitly:
``"local"`` (or empty) is ``hermes_state_port_local`` (today's per-profile ``state.db``, the ledger
in ``hermes_state_runtime`` and the ``gateway.lock`` owner lock); any other name is a provider from
``plugins/session_storage/<name>/`` or ``$HERMES_HOME/plugins/<name>/``. A named provider that is
missing or unavailable is an error, never a fallback to the local store.

Providers load in-process only. Under ``plugins.isolation: host`` a user provider is refused with
the reason: a host process would hand back opaque objects for the nested ports, and its calls would
not carry the caller's scope. Serving one from the plugin host needs typed proxies for the ports
and a validated scope envelope bound around every call, which is a follow-up.

Every operation reads its pin (tenant, agent, owner, actor; for a turn also the session, turn and
fence) from ``hermes_scope.current_scope()`` when it runs, never from a path, ``HERMES_HOME`` or an
argument; a port exposes no way to pin anything else. A backend refuses a pin it cannot serve
with :class:`StoragePinRefused`, and a turn-path write with no claimed turn bound with
``TurnFenceRequiredError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from hermes_scope import Fence

if TYPE_CHECKING:
    from hermes_state_rewind import RewindOutcome

# Sentinel: "claim whatever is first" (``hermes_state_runtime.claim_session_input``'s default).
ANY_HEAD: Any = object()

# The ownership levels a store can hold (``Scope`` docs): participant-on-agent, agent, owner-wide,
# tenant-wide. A backend declares which it stores and refuses writes at any other.
LEVELS = frozenset({"participant", "agent", "owner", "tenant"})


class StoragePinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""


class SessionStore(ABC):
    """Session rows and their transcripts, for the pinned agent."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """The session row, or ``None`` (missing and foreign look the same)."""

    @abstractmethod
    def create(self, session_id: str, source: str, **fields: Any) -> str:
        """Create (or upsert) a session row; returns its id."""

    @abstractmethod
    def end(self, session_id: str, reason: str) -> None:
        """Mark an open session ended."""

    @abstractmethod
    def append(self, session_id: str, messages: list[dict[str, Any]]) -> int:
        """Append *messages* in one transaction outside any turn (notes, imports); returns the
        count written. Inside a claimed turn of this store the turn's fence still applies."""

    @abstractmethod
    def append_turn(self, session_id: str, messages: list[dict[str, Any]]) -> int:
        """Append the running turn's *messages* in one transaction, fenced by its claim. Refused
        (``TurnFenceRequiredError``) when no claimed turn of *session_id*, or of a session it is a
        compression continuation of, is bound on this store."""

    @abstractmethod
    def read_conversation(self, session_id: str, *, include_row_ids: bool = False) -> list[dict[str, Any]]:
        """The session's active transcript, oldest first. ``include_row_ids`` adds each message's
        ``_row_id``, the id the row-addressed operations below take."""

    @abstractmethod
    def get_title(self, session_id: str) -> Optional[str]:
        """The session's title, or ``None``."""

    @abstractmethod
    def set_title(self, session_id: str, title: str) -> bool:
        """Set a user title; ``False`` when the session does not exist."""

    @abstractmethod
    def list_sessions(self, *, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """The pinned agent's sessions, newest first."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session and its transcript; ``False`` when there was nothing to delete."""

    # Transcript rewrites: each one transaction, refused inside a claimed turn that lost its fence.

    @abstractmethod
    def replace_messages(self, session_id: str, messages: list[dict[str, Any]], *, active_only: bool = False,
                         archive_dropped: bool = False, reject_active_turn_lease: bool = False) -> None:
        """Replace the session's transcript with *messages* (/retry, /undo). ``archive_dropped``
        keeps the replaced rows inactive instead of deleting them; ``active_only`` leaves already
        archived rows alone; ``reject_active_turn_lease`` refuses while a turn holds the session."""

    @abstractmethod
    def rewind_to_message(self, session_id: str, target_message_id: int, *,
                          preserve_compaction_handoff: bool = False,
                          expected_active_ids: Optional[list[int]] = None,
                          expected_target_content: Any = None) -> dict[str, Any]:
        """Deactivate every message from the user message *target_message_id* on; the expected
        active set and target content, when given, must still hold or nothing changes."""

    @abstractmethod
    def rewind_user_turn(self, session_id: str, user_ordinal: int, *, warm_history: Optional[list[dict[str, Any]]] = None,
                         require_retryable: bool = False, require_composite: bool = False,
                         adopt_row_ids: bool = False) -> RewindOutcome:
        """Rewind the active transcript to just before user turn *user_ordinal* (negative counts
        from the newest, ``-n`` is ``/undo n``): the :class:`~hermes_state_rewind.RewindOutcome`
        today's callers get."""

    @abstractmethod
    def set_user_message_content(self, session_id: str, row_id: int, content: Any,
                                 display_metadata: Optional[dict[str, Any]] = None) -> int:
        """Rewrite one active user message's content in place; returns the rows changed."""

    @abstractmethod
    def deactivate_message(self, session_id: str, row_id: int) -> int:
        """Deactivate one message (kept, never replayed); returns the rows changed."""

    @abstractmethod
    def set_message_reaction(self, session_id: str, row_id: int, emoji: Optional[str], *,
                             author: str = "user") -> Optional[list[dict[str, Any]]]:
        """Set or clear *author*'s reaction on a message; ``None`` for a message the session
        does not show."""

    # Compression publication.

    @abstractmethod
    def publish_compression_child(self, *, parent_session_id: str, child_session_id: str, source: str,
                                  messages: list[dict[str, Any]], **options: Any) -> None:
        """Close the parent and publish its compression child with *messages* in one transaction:
        readers see the live parent or the complete child, never an ended parent alone."""

    @abstractmethod
    def archive_and_compact(self, session_id: str, compacted_messages: list[dict[str, Any]],
                            **options: Any) -> int:
        """Compact in place under the same session id: the active rows become archived (still
        searchable) and *compacted_messages* the new active transcript; returns rows written."""

    # Usage.

    @abstractmethod
    def record_usage(self, session_id: str, *, queued: bool = True, **counts: Any) -> None:
        """Add a turn's token and cost counts to the session (``absolute=True`` replaces them).
        ``queued`` lets the backend coalesce the write; :meth:`flush_usage` waits for it."""

    @abstractmethod
    def record_auxiliary_usage(self, session_id: str, task: str, **counts: Any) -> None:
        """Record an auxiliary call's usage (titling, compression, vision) against the session,
        attributed to *task*, without touching the session's main counters."""

    @abstractmethod
    def read_auxiliary_usage(self, session_id: str) -> dict[str, dict[str, Any]]:
        """The auxiliary usage recorded against the session, per task: ``{task: {"api_calls",
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "estimated_cost_usd"}}``."""

    @abstractmethod
    def flush_usage(self, timeout: float = 5.0) -> bool:
        """Wait until every queued usage write has landed; ``False`` on timeout."""


class AdmissionGate(ABC):
    """The owner's admission gate: whether this owner may still start a turn (it is not draining or
    retiring). The backend asks it inside the claim's own transaction, after any wait for the
    writer, so a drain that began while the claim waited still refuses it."""

    @abstractmethod
    def require_open(self) -> None:
        """Return to allow the claim; raise to refuse it (the claim then changes nothing)."""


@dataclass(frozen=True)
class ClaimPreconditions:
    """What must still hold when :meth:`TurnAdmission.claim_next` commits, checked atomically by the
    backend with the claim.

    - ``head``: the admission id the caller validated as the FIFO head, ``None`` when it saw nothing
      queued, :data:`ANY_HEAD` to claim whatever is first. A different head claims nothing.
    - ``gate``: the owner's :class:`AdmissionGate`; a refusal propagates and nothing is claimed.
    """

    head: Any = ANY_HEAD
    gate: Optional[AdmissionGate] = None


class OwnerHandle(ABC):
    """A held owner lock (:meth:`TurnAdmission.acquire_owner`)."""

    @abstractmethod
    def release(self) -> None:
        """Give the lock up. Idempotent."""


class TurnAdmission(ABC):
    """The owner lock, owner epoch and per-session FIFO of admitted inputs, for the pinned agent.

    A claim issues the turn's :class:`~hermes_scope.Fence` (the owner epoch and the session's
    runtime generation it stamps); every later write of that turn is checked against it."""

    @abstractmethod
    def acquire_owner(self, *, purpose: Optional[str] = None) -> OwnerHandle:
        """Become the agent's one owner, or raise if another owner holds it. Nonblocking."""

    @abstractmethod
    def begin_epoch(self, *, instance_id: str) -> int:
        """Start a new owner epoch (the caller holds the owner lock); returns it."""

    @abstractmethod
    def recover(self, *, epoch: int) -> int:
        """After a crash: every started turn of an older epoch becomes ``unknown`` (never
        replayed); returns how many."""

    @abstractmethod
    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
              payload: dict[str, Any], intent: str = "queue") -> dict[str, Any]:
        """Admit one input to the session's FIFO; idempotent by ``(principal, session, request)``."""

    @abstractmethod
    def get(self, admission_id: str) -> Optional[dict[str, Any]]:
        """One admission, or ``None``."""

    @abstractmethod
    def find(self, *, principal_id: str, session_id: str, request_id: str) -> Optional[dict[str, Any]]:
        """The admission at its durable identity, or ``None``."""

    @abstractmethod
    def list(self, session_id: str, *, pending_only: bool = True) -> list[dict[str, Any]]:
        """The session's admissions in FIFO order."""

    @abstractmethod
    def claim_next(self, *, epoch: int, session_id: str,
                   preconditions: Optional[ClaimPreconditions] = None) -> dict[str, Any] | bool | None:
        """Claim the FIFO head: the started row (its fence stamped), ``None`` when nothing is
        claimable, ``False`` when the head is not ``preconditions.head``. A refusing gate raises
        and nothing is claimed."""

    @abstractmethod
    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[dict[str, Any], bool]:
        """Hand a still-queued admission to the session's running claim *generation*, so the drain
        never also claims it. ``(row, True)`` when this call made the transfer, ``(row, False)``
        for an exact retry of one already made."""

    @abstractmethod
    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        """A transfer whose text never reached the running turn goes back to its FIFO place;
        ``False`` when the claim's settlement already decided it."""

    @abstractmethod
    def settle(self, *, epoch: int, admission_id: str, generation: int, outcome: str,
               result: Optional[dict[str, Any]] = None,
               transfers: Optional[dict[str, bool]] = None) -> dict[str, Any]:
        """Record a claimed turn's outcome, fenced by its claim. *transfers* maps each admission
        steered into this turn to whether its text was delivered: delivered ones settle with the
        turn, undelivered ones are queued again."""

    @abstractmethod
    def cancel(self, *, epoch: int, admission_id: str) -> dict[str, Any]:
        """Cancel a queued admission."""

    @abstractmethod
    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int) -> dict[str, Any]:
        """Acknowledge a turn whose owner died; never requeues it."""

    @abstractmethod
    def generation(self, session_id: str) -> Optional[int]:
        """The session's current runtime generation, or ``None``."""

    @abstractmethod
    def mutate_session(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
                       expected_revision: int, operation: str, payload: dict[str, Any],
                       expected_generation: Optional[int] = None) -> dict[str, Any]:
        """One closed edit of a session (rename, archive, model, rewind, reset, delete, ...) with
        its retry receipt, fenced by the session's revision (and generation, for edits that
        invalidate a running turn). An exact retry returns the first result."""

    # Worker executions: work that runs outside the owner (cron, child, compute, kanban) and
    # persists through receipts the owner applies.

    @abstractmethod
    def register_worker(self, *, epoch: int, execution_id: str, session_id: str, generation: int,
                        kind: str, adoption_secret: str, require_idle: bool = False) -> dict[str, Any]:
        """Register a worker execution on the session at *generation*."""

    @abstractmethod
    def adopt_worker(self, *, epoch: int, execution_id: str, session_id: str, generation: int,
                     adoption_secret: str) -> dict[str, Any]:
        """A new owner epoch takes over a live worker execution that proves its secret."""

    @abstractmethod
    def apply_worker_op(self, *, epoch: int, execution_id: str, session_id: str, generation: int,
                        sequence: int, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply the worker's *sequence*-th operation (a transcript append, usage, compression,
        lifecycle, ``execution.finish``) with its receipt; an exact retry returns the stored
        result."""

    def fence(self, claimed: dict[str, Any]) -> Fence:
        """The fence a claimed row was issued under."""
        return Fence(claimed["owner_epoch"], claimed["generation"])


@dataclass(frozen=True)
class SessionStorage:
    """One opened pair of ports. ``close`` gives back whatever ``open`` took."""

    sessions: SessionStore
    admission: TurnAdmission
    close: Callable[[], None] = lambda: None


class SessionStorageProvider(ABC):
    """The plugin kind: supplies both ports. ``name``, ``open`` and ``assert_isolating`` are
    required; every other method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, db: Any = None) -> SessionStorage:
        """The ports for the bound scope. *db* is the local backend's per-profile handle (a
        ``SessionDB`` the caller already holds); a backend with no local file ignores it."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a Postgres backend whose role is a superuser or has ``BYPASSRLS`` fails).
        Required, with no default: a backend that has not checked must not pass. Nothing calls it
        yet: the first caller that serves a scope from a resolved provider must call it, and refuse
        the provider when it raises, before :meth:`open`."""


def resolve_session_storage(name: str) -> SessionStorageProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: storage never
    falls back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from hermes_state_port_local import LocalSessionStorageProvider
        return LocalSessionStorageProvider()
    from plugins.session_storage import SessionStorageRefused, load_session_storage
    try:
        provider = load_session_storage(name)
    except SessionStorageRefused as refused:
        raise LookupError(f"session storage {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"session storage {name!r}: no such session storage provider")
    if not provider.is_available():
        raise LookupError(f"session storage {name!r} is not available here")
    return provider
