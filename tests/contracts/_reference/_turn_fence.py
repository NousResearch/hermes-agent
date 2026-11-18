"""The claimed-turn fence on the reference backend's writes.

The reference has no write transaction of its own beyond its backend-wide lock: each public verb is one atomic step
(``backend.run_verb``). :func:`fenced` wraps every public repository verb of a store handle so that, at the outermost
verb, the bound claimed turn's fence (``persistence.core.fences.current_turn_fence`` for the handle's
``ProfileScope.store_key``) is checked against the store's authority state (the authority epoch and the session's
execution generation the admission claim stamped) before the verb runs. While the fence holds the verb runs as
before. Once it no longer holds, the verb still runs, so a read answers as usual, but a verb that changed any of the
store's state is undone and refused with ``persistence.TurnFenceSuperseded``: nothing of it lands, as in a backend
that checks the fence inside the write's own transaction.

The holder-qualified lease releases are not fenced: they drop only the caller's own row, so a turn that lost its
lease still releases it (as the SQLite backend does).
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Any, Callable, Optional

from persistence.core.errors import TurnFenceSuperseded
from persistence.core.fences import TurnFence, current_turn_fence, fence_holds, writes_as
from persistence.core.identity import SessionId

if TYPE_CHECKING:
    from .backend import InMemoryStore

__all__ = ["fenced"]

_AUTHORITY_KEY = "authority"  # the scope-namespace entry ``_authority.authority_state`` keeps its state under

# Verbs whose write only ever drops the caller's own lease row.
_UNFENCED = frozenset({("coordination", "release_turn_lease"), ("coordination", "release_compression_lock")})


def fenced(store: InMemoryStore, attribute: str, name: str, method: Callable[..., Any], *,
           fence_for: Callable[[str], Optional[TurnFence]] = current_turn_fence,
           holds: Optional[Callable[[InMemoryStore, TurnFence], bool]] = None,
           refusing: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """``method`` (verb ``name`` of repository ``attribute`` on ``store``) with the claimed-turn fence; called under
    the backend lock, inside ``run_verb``. ``fence_for``, ``holds`` and ``refusing`` are the fence's three steps (the
    store's fence, whether it holds, the refused run); the reference passes none, so the defaults below run."""
    holds = holds or _holds
    refusing = refusing or _refusing_writes
    if (attribute, name) in _UNFENCED:
        @functools.wraps(method)
        def unfenced(*args: Any, **kwargs: Any) -> Any:
            with writes_as(None):
                return method(*args, **kwargs)
        return unfenced

    @functools.wraps(method)
    def call(*args: Any, **kwargs: Any) -> Any:
        fence = fence_for(store.scope.store_key) if store.backend._depth <= 1 else None
        if fence is None or holds(store, fence):
            return method(*args, **kwargs)
        return refusing(store, fence, method, args, kwargs)

    return call


def _authority(store: InMemoryStore) -> Optional[Any]:
    if not store.backend.scope_exists(store.scope):
        return None
    return store.backend.scope_store(store.scope).get(_AUTHORITY_KEY)


def _holds(store: InMemoryStore, fence: TurnFence) -> bool:
    authority = _authority(store)
    if authority is None:
        return False
    committed = authority.snapshot
    session = committed["sessions"].get(fence.session_id)
    return fence_holds(fence, committed["epoch"] or None, None if session is None else session["generation"])


def _empty(value: Any) -> bool:
    return isinstance(value, (dict, list, set, tuple)) and not value


def _refusing_writes(store: InMemoryStore, fence: TurnFence, method: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Run a verb of a superseded turn: its answer when it changed nothing, else the change undone and refused."""
    namespace = store.backend.scope_store(store.scope)
    authority = namespace.get(_AUTHORITY_KEY)
    saved = {key: copy.deepcopy(value) for key, value in namespace.items() if key != _AUTHORITY_KEY}
    saved_committed = None if authority is None else copy.deepcopy(authority.snapshot)
    before = repr((saved, saved_committed))

    def changed() -> bool:
        now = namespace.get(_AUTHORITY_KEY)
        # A read creates the tables it reads on first touch (``setdefault``); an empty new table is no write.
        current = {key: value for key, value in namespace.items()
                   if key != _AUTHORITY_KEY and (key in saved or not _empty(value))}
        return now is not authority or repr((current, None if now is None else now.snapshot)) != before

    def restore() -> None:
        namespace.clear()
        namespace.update(saved)
        if authority is not None:
            namespace[_AUTHORITY_KEY] = authority
            authority._committed = saved_committed

    refusal = TurnFenceSuperseded(
        SessionId(fence.session_id),
        f"turn {fence.turn_id!r} of session {fence.session_id!r} no longer holds its lease; refusing its write")
    try:
        result = method(*args, **kwargs)
    except BaseException as exc:
        if changed():
            restore()
            raise refusal from exc
        raise
    if changed():
        restore()
        raise refusal
    return result
