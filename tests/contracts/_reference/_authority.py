"""Shared authority state, failpoints, and copy-on-write publication for the reference backend.

This module is the bounded floor the authority capability repositories build on. It owns, per scope: the authority epoch singleton, per-session revision/generation counters, the admission/worker/receipt/retirement tables, the meta receipt space, and the one logical atomic boundary every composite verb commits through. It deliberately owns NOTHING else: scheduling, transport, process ownership, and restart recovery stay outside.

Atomicity model: :meth:`AuthorityState.transaction` stages a deep copy of BOTH the authority state and the sibling domain tables in the scope namespace (composite verbs atomically couple transcript/usage/session rows with their receipt, so the boundary must reach the tables the other capability repositories read, not just authority bookkeeping). Publication swaps the staged copies back with no failure window of its own: an armed failpoint fires between staged mutations (verbs call ``state.failpoints.hit(name)`` wherever the crash window under test sits) or at :data:`AuthorityState.PUBLISH` immediately before the swap; either way neither the authority state nor any domain table shows a partial transition, and an exactly-once retry can be proven against the untouched committed state.

Frozen API for capability repositories: ``authority_state``, ``AuthorityState`` (``snapshot``, ``transaction`` yielding a :class:`Staged` view, ``guarded`` for composites over sibling verbs, ``begin_epoch``, ``require_epoch``, ``failpoints``), ``Failpoints.arm``/``clear``, ``InjectedFailure``, and the module-level helpers (``session_runtime``, ``require_revision``, ``require_generation``, ``replay``/``ReplayHit``/``RESULT_SCRUBBED``, ``require_next_sequence``, ``record_retirement``, ``require_not_retired``). Capability repositories consume these helpers unchanged; missing functionality is reported rather than addressed with a local edit.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from persistence.core.errors import AuthorityRefusal
from persistence import (
    AuthorityEpoch,
    Conflict,
    FenceStale,
    RetiredSessionError,
    SessionId,
    StaleFenceKind,
    ProfileScope,
)

from .backend import InMemoryBackend

__all__ = [
    "RESULT_SCRUBBED",
    "AuthorityState",
    "Failpoints",
    "InjectedFailure",
    "ReplayHit",
    "Staged",
    "authority_state",
    "record_retirement",
    "replay",
    "require_generation",
    "require_next_sequence",
    "require_not_retired",
    "require_revision",
    "session_runtime",
]


class InjectedFailure(RuntimeError):
    """Raised by an armed failpoint to simulate a crash inside a composite transition; only the arming test observes it."""


class Failpoints:
    """Named failure-injection points, armed per test and self-disarming.

    ``arm(name, times=n)`` makes the next ``n`` hits of ``name`` raise :class:`InjectedFailure`; after that the point is disarmed, so the standard exactly-once-retry proof is arm once, expect the failure, retry, expect success.
    """

    def __init__(self) -> None:
        self._armed: dict[str, int] = {}

    def arm(self, name: str, *, times: int = 1) -> None:
        if times < 1:
            raise ValueError(f"failpoint times must be >= 1, got {times}")
        self._armed[name] = times

    def clear(self, name: Optional[str] = None) -> None:
        if name is None:
            self._armed.clear()
        else:
            self._armed.pop(name, None)

    def hit(self, name: str) -> None:
        remaining = self._armed.get(name)
        if not remaining:
            return
        if remaining == 1:
            self._armed.pop(name)
        else:
            self._armed[name] = remaining - 1
        raise InjectedFailure(name)


def _empty_state() -> dict[str, Any]:
    return {
        "epoch": 0,
        "instance_id": None,
        # session id -> {"revision": int, "generation": int}
        "sessions": {},
        # admission id -> admission record dict (the admissions repository defines the shape)
        "admissions": {},
        # execution id -> worker record dict incl. "last_sequence"
        "workers": {},
        # receipt key -> {"digest": str, "result": Any, optional "scrubbed": True}
        "receipts": {},
        # session id -> retirement marker dict (permanent; never removed)
        "retired": {},
        # durable key/value receipt space (mutation receipts, policy receipts, terminal evidence)
        "meta": {},
    }


_AUTHORITY_KEY = "authority"


@dataclass
class Staged:
    """One transaction's staged view: ``authority`` is the authority-state copy, ``scope`` the sibling domain tables copy (everything else in the scope namespace). Both publish together or not at all."""

    authority: dict[str, Any]
    scope: dict[str, Any]


class AuthorityState:
    """One scope's authority state: committed snapshot plus the copy-on-write transaction boundary over the whole scope namespace."""

    #: Failpoint the transaction itself checks immediately before publication.
    PUBLISH = "publish"

    def __init__(self, scope_store: dict[str, Any]) -> None:
        self.failpoints = Failpoints()
        self._committed: dict[str, Any] = _empty_state()
        # The owning scope namespace: sibling domain tables (sessions, messages,
        # usage, routing, ...) live beside the authority entry, and composite
        # verbs must publish writes to them atomically with their receipt.
        self._scope_store = scope_store

    def __repr__(self) -> str:
        return f"AuthorityState({self._committed!r})"

    @property
    def snapshot(self) -> dict[str, Any]:
        """The committed authority state, for reads. Callers never mutate it directly; every write goes through :meth:`transaction`."""
        return self._committed

    @contextmanager
    def transaction(self) -> Iterator[Staged]:
        """Stage mutations on deep copies of the authority state AND the sibling domain tables, publishing them as one atomic swap.

        An exception anywhere inside the block (including an armed failpoint, and the PUBLISH failpoint checked right before the swap) discards both staged copies; neither the committed authority state nor any domain table shows a partial transition. Publication mutates the scope namespace in place so repository reads through ``backend.scope_store`` observe the published tables without re-wiring.
        """
        staged = Staged(
            authority=copy.deepcopy(self._committed),
            scope={
                key: copy.deepcopy(value)
                for key, value in self._scope_store.items()
                if key != _AUTHORITY_KEY
            },
        )
        yield staged
        self.failpoints.hit(self.PUBLISH)
        self._scope_store.clear()
        self._scope_store.update(staged.scope)
        self._scope_store[_AUTHORITY_KEY] = self
        self._committed = staged.authority

    @contextmanager
    def guarded(self) -> Iterator[None]:
        """Run a composite verb on the committed tables directly, restoring them when it raises.

        For verbs that compose sibling repositories' own verbs (each of which commits on its own): the committed authority state and every sibling table are copied first, and an exception anywhere inside the block (an armed failpoint included) puts the copies back, so the composite is one atomic step under the backend lock. Code inside re-reads :attr:`snapshot` after a sibling verb, which may publish a new committed dict.
        """
        saved_authority = copy.deepcopy(self._committed)
        saved_scope = {key: copy.deepcopy(value) for key, value in self._scope_store.items() if key != _AUTHORITY_KEY}
        try:
            yield
            self.failpoints.hit(self.PUBLISH)
        except BaseException:
            self._scope_store.clear()
            self._scope_store.update(saved_scope)
            self._scope_store[_AUTHORITY_KEY] = self
            self._committed = saved_authority
            raise

    def begin_epoch(self, instance_id: str) -> AuthorityEpoch:
        """Advance the scope's authority epoch and record the new holder; monotonic, never reused."""
        with self.transaction() as staged:
            staged.authority["epoch"] += 1
            staged.authority["instance_id"] = instance_id
        return AuthorityEpoch(self._committed["epoch"])

    def require_epoch(self, state: dict[str, Any], epoch: object) -> None:
        """Refuse any epoch but the current one: a non-int (bool and float included), a never-issued zero, and any mismatch are all the epoch-kind stale fence. Takes ``object`` because the fence itself owns the type check."""
        if type(epoch) is not int or state["epoch"] == 0 or epoch != state["epoch"]:
            raise FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton") from AuthorityRefusal("stale_epoch")


def authority_state(backend: InMemoryBackend, scope: ProfileScope) -> AuthorityState:
    """The one :class:`AuthorityState` for a scope, created on first touch inside the scope's namespace so cross-scope isolation stays structural."""
    store = backend.scope_store(scope)
    state = store.get(_AUTHORITY_KEY)
    if state is None:
        state = store[_AUTHORITY_KEY] = AuthorityState(store)
    return state


def session_runtime(state: dict[str, Any], session_id: SessionId) -> dict[str, int]:
    """The per-session runtime counters, zero-initialized on first touch. ``state`` is a staged or committed authority dict."""
    entry = state["sessions"].get(session_id)
    if entry is None:
        entry = state["sessions"][session_id] = {"revision": 0, "generation": 0}
    return entry


def require_revision(entry: dict[str, int], session_id: SessionId, expected: int) -> None:
    if entry["revision"] != expected:
        raise FenceStale(StaleFenceKind.REVISION_CONFLICT, "session", str(session_id)) from AuthorityRefusal("revision_conflict")


def require_generation(entry: dict[str, int], session_id: SessionId, expected: int) -> None:
    if entry["generation"] != expected:
        raise FenceStale(StaleFenceKind.STALE_GENERATION, "session", str(session_id)) from AuthorityRefusal("stale_generation")


class ReplayHit:
    """A digest-matched receipt: ``result`` is a detached copy of the stored result, returned verbatim without re-executing (the result itself may be anything, including None)."""

    __slots__ = ("result",)

    def __init__(self, result: Any) -> None:
        self.result = result


#: Marker a retirement fold sets on a receipt whose result was scrubbed (the PR
#: retains only the closing receipt's result at retirement; earlier results are
#: user data that must not outlive the delete, while their digests stay so a
#: late duplicate is still recognised).
RESULT_SCRUBBED = "scrubbed"


def replay(receipts: dict[Any, dict[str, Any]], key: Any, digest: str, *, entity: str) -> Optional[ReplayHit]:
    """The one dedup-receipt rule: absent key returns None (caller executes), matching digest returns a DETACHED copy of the stored result (mutating it cannot alter committed evidence), mismatched digest is a :class:`persistence.Conflict`.

    The digest is checked before the scrubbed-result rule, so a mismatched late duplicate still surfaces as a conflict; a matching digest whose result was scrubbed at retirement refuses with the generation-kind stale fence instead of replaying.
    """
    old = receipts.get(key)
    if old is None:
        return None
    if old["digest"] != digest:
        raise Conflict(entity, str(key)) from AuthorityRefusal("admission_conflict")
    if old.get(RESULT_SCRUBBED):
        raise FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(key)) from AuthorityRefusal("stale_generation")
    return ReplayHit(copy.deepcopy(old["result"]))


def require_next_sequence(worker: dict[str, Any], sequence: int) -> None:
    """Strict next-sequence gate for receipted worker writes: exactly ``last_sequence + 1``, else an invalid-params refusal (``ValueError`` at the contract boundary)."""
    if sequence != worker["last_sequence"] + 1:
        raise ValueError(
            f"sequence {sequence} is not the strict successor of {worker['last_sequence']}"
        )


def record_retirement(state: dict[str, Any], session_id: SessionId, *, expires: bool = False) -> None:
    """Publish the retirement marker for a session id; re-recording is a no-op fact, never a conflict. ``expires`` marks a sweep's marker, which ages out at the retry horizon; a user delete's marker never does."""
    marker = state["retired"].setdefault(session_id, {})
    if expires and not marker:
        marker["expires"] = True


def require_not_retired(state: dict[str, Any], session_id: SessionId) -> None:
    """The creation fence: a retired id is never creatable again.

    Applies to creating an ABSENT id: the marker is checked only when no live row exists; a live row standing beside a stale marker is re-retired through the delete path, never refused here.
    """
    if session_id in state["retired"]:
        raise RetiredSessionError(session_id)
