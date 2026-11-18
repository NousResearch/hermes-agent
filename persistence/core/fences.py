"""Authority fence counters for the persistence boundary.

The session authority fences runtime ownership with three counters: the store-wide authority epoch, the per-session mutation revision, and the per-session execution generation. Each is an opaque monotonic value minted by the backend: callers carry it back verbatim and never do arithmetic beyond equality. A stale counter is refused with :class:`persistence.FenceStale`.

``TurnFence`` is the claimed-turn fence a write carries implicitly: the authority epoch and execution generation a
turn's admission claim stamped, and the key of the store that issued it (``ProfileScope.store_key``). Every write a
backend's repositories make reads :func:`current_turn_fence` for its own store key at the start of the write's own
transaction and, when one is returned, refuses with :class:`persistence.TurnFenceSuperseded` unless the epoch (and,
unless ``owner_only``, the session's generation) are still current, so a superseded turn writes nothing. No fence
(``None``) means the write is not a claimed turn's on this store and is checked by nothing new.

Which writes are a turn's: every write made in the turn's context (``hermes_scope.current_turn``) to the issuing store.
That includes the threads and tasks the turn starts (they copy the context) for their whole lifetime, also after the
turn's body returned, and work a shared thread applies on a turn's behalf under :func:`writes_as` (the queued token
writer applies each delta as the turn that queued it). It excludes writes to another store, the owner's own work
(``agent.runtime_scope.owner_operation``: claims, settlement, the drains; they keep the ledger's own epoch and
generation checks), and the generation half for work designed to outlive the turn
(``agent.runtime_scope.outlive_turn``, ``owner_only``).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, NewType, Optional

__all__ = [
    "AuthorityEpoch", "ExecutionGeneration", "SessionRevision", "TurnFence", "current_turn_fence", "fence_holds", "writes_as",
    "writing_turn",
]

AuthorityEpoch = NewType("AuthorityEpoch", int)
SessionRevision = NewType("SessionRevision", int)
ExecutionGeneration = NewType("ExecutionGeneration", int)


@dataclass(frozen=True, slots=True)
class TurnFence:
    """The lease one claimed turn writes under, as a backend checks it."""

    session_id: str
    turn_id: str
    epoch: AuthorityEpoch
    generation: ExecutionGeneration
    store_key: Optional[str]
    """The ``ProfileScope.store_key`` of the store that issued the claim; a write to any other store is not fenced."""
    owner_only: bool = False
    """Work designed to outlive the turn (an async delegation's completion): only the epoch half is checked, so a
    takeover still refuses it while the session's next claim does not."""


_FROM_CONTEXT: Any = object()
# An explicit provenance for the writes in this context, overriding the bound turn: a shared thread applying work
# queued by a turn (the token writer) writes as that turn; ``None`` writes as no turn.
_WRITES_AS: ContextVar[Any] = ContextVar("persistence_turn_fence_writes_as", default=_FROM_CONTEXT)


@contextmanager
def writes_as(mark: Any) -> Iterator[None]:
    """Make the writes in the body those of turn *mark* (a ``hermes_scope.TurnMark``, as :func:`writing_turn` returned
    it; ``None``: of no turn), whatever turn the context is bound to."""
    token = _WRITES_AS.set(mark)
    try:
        yield
    finally:
        _WRITES_AS.reset(token)


def writing_turn() -> Any:
    """The turn the writes made here belong to, on any store (``None``: no turn's): what a write deferred to another
    thread carries, to apply later under :func:`writes_as`."""
    mark = _WRITES_AS.get()
    if mark is _FROM_CONTEXT:
        from hermes_scope import current_turn
        mark = current_turn()
    return mark


def current_turn_fence(store_key: str) -> Optional[TurnFence]:
    """The fence a write to the store keyed ``store_key`` must check in its own transaction, from the bound claimed
    turn; ``None`` outside a claimed turn, inside the owner's own operations, or when the turn's fence was issued by
    another store."""
    mark = writing_turn()
    if mark is None or mark.store is None or mark.store != store_key:
        return None
    return TurnFence(session_id=mark.session, turn_id=mark.turn, epoch=AuthorityEpoch(mark.fence.epoch),
                     generation=ExecutionGeneration(mark.fence.generation), store_key=mark.store,
                     owner_only=mark.outlives)


def fence_holds(fence: TurnFence, epoch: Optional[int], generation: Optional[int]) -> bool:
    """Whether ``fence`` still holds its session's lease given the store's current authority ``epoch`` and the
    session's current execution ``generation`` (``None``: none recorded): the epoch must be the claim's and, unless
    ``owner_only``, the generation too."""
    if epoch is None or epoch != fence.epoch:
        return False
    return fence.owner_only or (generation is not None and generation == fence.generation)
