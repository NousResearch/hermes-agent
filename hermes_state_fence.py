"""Turn fences on session store writes.

A claimed turn runs under the fence its claim issued (``hermes_scope.Fence``: the owner epoch of the
owner that claimed it and the claimed session's runtime generation at the claim). The turn's identity
is the bound ``hermes_scope.TurnScope``; ``agent.runtime_scope.bind_turn`` adds the store that
issued the claim (``hermes_scope.TurnMark``), and a TurnScope bound without it is fenced on its
profile's ``state.db``. Every write made in the turn's context to that store runs
:func:`check_turn_lease` first, inside the write's own transaction (``SessionDB._execute_write``):
if the profile's owner epoch or the session's generation has moved since the claim, the turn no
longer holds its session's lease and the write is refused before anything of it lands.

Which writes are a turn's: every write made in the turn's context to the issuing store. That
includes the threads and tasks the turn starts (they copy the context) for their whole lifetime,
also after the turn's body returned, and token updates the turn queued for the background writer
(each carries the turn it was queued under, :func:`writes_as`). It excludes:

- writes to another store (another profile's ``state.db``: the fence says nothing about it);
- the owner's own work, which runs under ``agent.runtime_scope.owner_operation``: the drain that
  claims and settles turns (settlement keeps the ledger's own owner-epoch/generation checks), also
  when the drain task was started from a turn's context;
- the generation half of the fence for work designed to outlive the turn (an async delegation
  reports after it, ``agent.runtime_scope.outlive_turn``): it keeps the owner-epoch half, so a
  takeover still refuses it while the session's next claim does not;
- writes outside any claimed turn: CLI and TUI turns (they hold the durable turn lease instead),
  compression and maintenance the owner runs between turns, recovery, imports.

While the lease holds, a turn's write behaves exactly as before; the check is two indexed reads on
the writer connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from hermes_state_errors import TurnFenceSupersededError

_FROM_CONTEXT: Any = object()
# An explicit provenance for the writes in this context, overriding the bound turn: the queued token
# writer applies each update as the turn that queued it, the ledger writes as no turn.
_WRITES_AS: ContextVar[Any] = ContextVar("hermes_state_fence_writes_as", default=_FROM_CONTEXT)

_EPOCH_SQL = "SELECT epoch FROM runtime_epoch WHERE singleton=1"
_GENERATION_SQL = "SELECT runtime_generation FROM sessions WHERE id=?"


def store_key(path) -> str:
    """The identity a turn mark and a store compare: the store file's resolved path."""
    from hermes_constants import hermes_home_key
    return hermes_home_key(path)


def store_turn(db):
    """The claimed turn the writes made in this context belong to when *db* is the store that
    issued its fence, else ``None``."""
    mark = writing_turn()
    if mark is None or mark.store is None:
        return None
    path = getattr(db, "db_path", None)
    if path is None or store_key(path) != mark.store:
        return None
    return mark


@contextmanager
def writes_as(mark) -> Iterator[None]:
    """Make the writes in the body those of turn *mark* (``None``: of no turn), whatever turn the
    context is bound to: work applied on a turn's behalf by a shared thread (the token writer)."""
    token = _WRITES_AS.set(mark)
    try:
        yield
    finally:
        _WRITES_AS.reset(token)


def writing_turn():
    """The turn the writes made here would belong to, on any store (``None``: no turn's), as
    :func:`store_turn` reads it: what a write deferred to another thread must carry."""
    mark = _WRITES_AS.get()
    if mark is _FROM_CONTEXT:
        from hermes_scope import current_turn
        mark = current_turn()
    return mark


def lease_holds(conn, session_id: str, fence, *, owner_only: bool = False) -> bool:
    """Whether *fence* is still *session_id*'s lease on this store: the owner epoch and the runtime
    generation the claim stamped are both still current (*owner_only*: the epoch alone)."""
    epoch = conn.execute(_EPOCH_SQL).fetchone()
    if epoch is None or epoch[0] != fence.epoch:
        return False
    if owner_only:
        return True
    row = conn.execute(_GENERATION_SQL, (session_id,)).fetchone()
    return row is not None and row[0] == fence.generation


def check_turn_lease(conn, mark) -> None:
    """Refuse the current write of turn *mark* once it no longer holds its session's lease. Runs on
    the write's connection inside its transaction; ``None`` (no claimed turn of this store) checks
    nothing."""
    if mark is None or lease_holds(conn, mark.session, mark.fence, owner_only=mark.outlives):
        return
    raise TurnFenceSupersededError(
        f"turn {mark.turn!r} of session {mark.session!r} no longer holds its turn lease "
        f"(fence epoch={mark.fence.epoch}, generation={mark.fence.generation}); refusing its write")
