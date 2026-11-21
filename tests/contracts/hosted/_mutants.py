"""Broken backends the hosted contract cases must reject (``test_mutants.py``).

Each mutant is a real registered backend (the in-memory reference, or the SQLite backend) with ONE guard removed, and
names the contract case(s) that must fail against it, with the failure that counts as the kill: the case's own
assertion (``AssertionError``, or ``pytest.raises`` finding no refusal) unless a killer names the exception it expects
to escape (a refusal the broken guard now raises where the contract says none is due).

Mutants are never registered as contract backends: nothing here calls ``register_backend``, and the regular run of
``tests/contracts`` neither parametrizes over nor accounts for them. ``test_mutants.py`` opens each one directly
and calls its killer cases with it.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping, Optional
from unittest import mock

import pytest

from persistence import ProfileScope, SessionId, TurnFenceSuperseded
from persistence.core.fences import TurnFence, current_turn_fence, writing_turn
from tests.contracts._reference import InMemoryBackend
from tests.contracts._reference import _turn_fence as reference_fence
from tests.contracts._scopes import pinned_scope, profile_scope
from tests.contracts.hosted import (
    test_hosted_activity_contract as activity,
    test_pinned_housekeeping_contract as housekeeping,
    test_scope_pinning_contract as pinning,
    test_turn_fence_contract as turn_fence,
    test_turn_lease_holder_contract as lease_holder,
    test_turn_write_families_contract as families,
)

__all__ = ["MUTANTS", "SCOPES", "Killer", "Mutant"]

# The contract suites' two test scopes (the conftest's pair).
SCOPES = (profile_scope("profile-a"), profile_scope("profile-b"))

_CONTRACT_FAILURE = (AssertionError, pytest.fail.Exception)


@dataclass(frozen=True)
class Killer:
    """A contract case that must fail against the mutant: ``case`` called with the backend, the two scopes and
    ``params``; ``reason`` is the failure that counts as the kill."""

    case: Callable[..., None]
    params: Mapping[str, Any] = field(default_factory=dict)
    reason: tuple[type[BaseException], ...] = _CONTRACT_FAILURE

    @property
    def node(self) -> str:
        suffix = "".join(f"[{key}={value}]" for key, value in sorted(self.params.items()))
        return f"{self.case.__module__.rsplit('.', 1)[-1]}::{self.case.__name__}{suffix}"


@dataclass(frozen=True)
class Mutant:
    """One broken backend: ``open()`` yields it (torn down on exit), ``base`` names the registered backend it breaks."""

    name: str
    base: str
    removed: str
    open: Callable[[], ContextManager[Any]]
    killers: tuple[Killer, ...]


# ── the reference with one guard removed ────────────────────────────────────


@contextmanager
def _opened(cls: type[InMemoryBackend]) -> Iterator[InMemoryBackend]:
    backend = cls()
    try:
        yield backend
    finally:
        backend.close()


_SHARED_PIN = pinned_scope("every-tenant", "every-agent")


class _PinIgnored(InMemoryBackend):
    """Every pinned scope is served from one shared store: the pin no longer chooses the store."""

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        return super().scope_store(_SHARED_PIN if scope.pin is not None else scope)

    def scope_exists(self, scope: ProfileScope) -> bool:
        return super().scope_exists(_SHARED_PIN if scope.pin is not None else scope)


class _PinAcceptedUndeclared(InMemoryBackend):
    """Declares no ``scope_pinning`` yet still opens a pinned scope instead of refusing it."""

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = dataclasses.replace(self.capabilities, scope_pinning=False)


class _CrossPin(InMemoryBackend):
    """A backend whose pinned handles fall through to another pin's store for one family of verbs (``_LEAKED``)."""

    _LEAKED: tuple[str, ...] = ()

    def open(self, scope: ProfileScope) -> Any:
        handle = super().open(scope)
        if scope.pin is not None and not getattr(handle, "_cross_pin", False):
            handle._cross_pin = True
            own = {name: getattr(handle.sessions, name) for name in ("get", *self._LEAKED)}
            handle._own_sessions = own
            for name in self._LEAKED:
                setattr(handle.sessions, name, functools.partial(self._fall_through, handle, name))
        return handle

    def _other_holding(self, handle: Any, session_id: SessionId) -> Optional[Any]:
        """Another pin's handle holding ``session_id`` when ``handle``'s own store does not."""
        if handle._own_sessions["get"](session_id) is not None:
            return None
        for other in list(self._handles.values()):
            if other is not handle and getattr(other, "_cross_pin", False) and other._own_sessions["get"](session_id) is not None:
                return other
        return None

    def _fall_through(self, handle: Any, name: str, session_id: SessionId, *args: Any, **kwargs: Any) -> Any:
        other = self._other_holding(handle, session_id)
        return (other or handle)._own_sessions[name](session_id, *args, **kwargs)


class _ForeignRowVisible(_CrossPin):
    """A pinned handle's session read answers another pin's row for an id its own store lacks."""

    _LEAKED = ("get",)


class _ForeignWriteApplied(_CrossPin):
    """A pinned handle's session writes land on another pin's row for an id its own store lacks."""

    _LEAKED = ("update_cwd", "set_title", "end", "patch_model_config")


class _SkipFence(InMemoryBackend):
    """The claimed-turn fence is not applied to the verbs in ``_SKIPPED``."""

    _SKIPPED: frozenset[tuple[str, str]] = frozenset()

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        if (attribute, name) in self._SKIPPED:
            return method
        return super().fence_verb(store, attribute, name, method)


class _AppendUnfenced(_SkipFence):
    """A superseded turn's transcript append is not fence-checked."""

    _SKIPPED = frozenset({("messages", "append"), ("messages", "append_batch")})


class _CompressionUnfenced(_SkipFence):
    """A superseded turn's compression continuation is not fence-checked."""

    _SKIPPED = frozenset({("lineage", "publish_compression_child")})


class _OwnerOnlyChecksGeneration(InMemoryBackend):
    """Work that outlives its turn (``owner_only``) is checked on the generation too: the session's next claim refuses
    a delegation's completion."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        return reference_fence.fenced(store, attribute, name, method, holds=lambda s, fence: reference_fence._holds(
            s, dataclasses.replace(fence, owner_only=False)))


class _OwnerOnlySkipsEpoch(InMemoryBackend):
    """Work that outlives its turn (``owner_only``) is not checked at all: it lands after another owner's takeover."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        return reference_fence.fenced(store, attribute, name, method,
                                      holds=lambda s, fence: fence.owner_only or reference_fence._holds(s, fence))


def _fence_of_any_store(store_key: str) -> Optional[TurnFence]:
    """The bound turn's fence whatever store it was issued by: the store-key filter dropped."""
    mark = writing_turn()
    if mark is None or mark.store is None:
        return None
    return current_turn_fence(mark.store)


class _StoreKeyIgnored(InMemoryBackend):
    """The bound turn's fence is checked on every store, not only the one that issued the claim."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        return reference_fence.fenced(store, attribute, name, method, fence_for=_fence_of_any_store)


def _stored(namespace: Mapping[str, Any]) -> str:
    return repr({key: value for key, value in namespace.items()
                 if key != "authority" and not (isinstance(value, (dict, list, set, tuple)) and not value)})


def _refused_after_writing(store: Any, fence: TurnFence, method: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """The superseded turn's verb runs and keeps what it wrote; the refusal comes only afterwards."""
    namespace = store.backend.scope_store(store.scope)
    before = _stored(namespace)
    result = method(*args, **kwargs)
    if _stored(namespace) != before:
        raise TurnFenceSuperseded(SessionId(fence.session_id), "refused after the write landed")
    return result


class _FenceAfterWrite(InMemoryBackend):
    """The fence is checked after the write instead of before it, and the write is not undone: partial state."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        return reference_fence.fenced(store, attribute, name, method, refusing=_refused_after_writing)


class _BoundPinUnchecked(InMemoryBackend):
    """A verb runs whatever hosted scope is bound: a kept repository serves another pin's activity, or a local store a
    hosted one."""

    def check_bound_pin(self, scope: ProfileScope) -> None:
        return None


class _SharedPlacementUnpinned(InMemoryBackend):
    """The ``'shared'`` hosted-room placement is chosen by the scope's folder, not its pin: every pin built in one
    folder shares it."""

    def shared_file_store(self, scope: ProfileScope, path: Path) -> dict[str, Any]:
        return self.file_store(path)


class _FenceMissesSharedFiles(InMemoryBackend):
    """The fence undoes and refuses only what a verb changed in the store's own namespace: a write to the ``'shared'``
    placement (a store file outside it) lands after the turn lost its lease."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        return reference_fence.fenced(store, attribute, name, method,
                                      refusing=functools.partial(reference_fence._refusing_writes, files=False))


class _LeaseRenewalBroken(InMemoryBackend):
    """An append by the holder of an expired turn lease lands but no longer renews the lease."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        if attribute != "messages" or name not in ("append", "append_batch"):
            return super().fence_verb(store, attribute, name, method)

        @functools.wraps(method)
        def unrenewed(*args: Any, **kwargs: Any) -> Any:
            namespace = store.backend.scope_store(store.scope)
            saved = copy.deepcopy(namespace.get("coord_turn_leases", {}))
            result = method(*args, **kwargs)
            # The append publishes its tables copy-on-write: read the lease table it left behind.
            leases = namespace.get("coord_turn_leases", {})
            for key, (holder, _expires) in list(leases.items()):
                if key in saved and saved[key][0] == holder:
                    leases[key] = saved[key]
            return result
        return super().fence_verb(store, attribute, name, unrenewed)


# ── the SQLite backend with one guard removed ───────────────────────────────


@contextmanager
def _sqlite(backend_cls: Optional[type] = None, **patches: Callable[..., Any]) -> Iterator[Any]:
    """The contract suites' SQLite backend, built as ``backend_cls`` (a ``ContractSQLiteBackend`` subclass) and with
    ``persistence.sqlite._holder`` names replaced by ``patches`` while it is open."""
    from persistence.sqlite import _holder
    from tests.contracts import _sqlite_seams
    from tests.contracts._sqlite_backend import _open_sqlite_backend

    with mock.patch.object(_sqlite_seams, "ContractSQLiteBackend", backend_cls or _sqlite_seams.ContractSQLiteBackend), \
            mock.patch.multiple(_holder, **patches) if patches else _nothing(), _open_sqlite_backend(SCOPES) as backend:
        yield backend


@contextmanager
def _nothing() -> Iterator[None]:
    yield


def _sqlite_isolation_claimed() -> ContextManager[Any]:
    from tests.contracts._sqlite_seams import ContractSQLiteBackend

    class _IsolationClaimed(ContractSQLiteBackend):
        """assert_isolating passes although the backend serves no pinned scope."""

        def assert_isolating(self) -> None:
            return None

    return _sqlite(_IsolationClaimed)


def _sqlite_owner_only_skips_epoch() -> ContextManager[Any]:
    from persistence.sqlite import _holder

    original = _holder.check_turn_fence

    def check(conn: Any, fence: Optional[TurnFence]) -> None:
        if fence is not None and fence.owner_only:
            return None
        return original(conn, fence)
    return _sqlite(check_turn_fence=check)


def _sqlite_serves_hosted_activity() -> ContextManager[Any]:
    return _sqlite(refuse_hosted_activity=lambda: None)


@contextmanager
def _sqlite_with(target: Callable[[], Any], **replacements: Any) -> Iterator[Any]:
    """The contract suites' SQLite backend with attributes of ``target()`` (a class of the SQLite backend, imported
    when the mutant opens) replaced while it is open."""
    with mock.patch.multiple(target(), **replacements), _sqlite() as backend:
        yield backend


def _sqlite_store_class() -> type:
    from persistence.sqlite._backend import SQLiteStore

    return SQLiteStore


def _sqlite_health_class() -> type:
    from persistence.sqlite.services import SQLiteStorageHealth

    return SQLiteStorageHealth


def _sqlite_hosted_files_class() -> type:
    from persistence.sqlite.gateway._hosted_store import HostedFiles

    return HostedFiles


def _sqlite_own_connections_serve_hosted_activity() -> ContextManager[Any]:
    return _sqlite_with(_sqlite_store_class, require_open=lambda self: self._backend._require_open())


def _sqlite_health_serves_hosted_activity() -> ContextManager[Any]:
    from persistence.sqlite._state.health import STORAGE_CORRUPT, storage_state
    from persistence.sqlite.services import probe_state_db

    from persistence import StorageState

    return _sqlite_with(
        _sqlite_health_class, probe=lambda self: probe_state_db(self._path),
        storage_state=lambda self: StorageState.CORRUPT if storage_state(self._path) == STORAGE_CORRUPT else StorageState.OK)


def _sqlite_shared_write_unfenced() -> ContextManager[Any]:
    return _sqlite_with(_sqlite_hosted_files_class, _claimed_turn_holder=lambda self: None)


def _sqlite_store_key_ignored() -> ContextManager[Any]:
    return _sqlite(turn_fence_for=lambda holder: _fence_of_any_store(""))


# ── the registry ────────────────────────────────────────────────────────────


def _families(*names: str, supersession: tuple[str, ...] = ("next_claim",)) -> tuple[Killer, ...]:
    case = families.test_each_turn_write_family_is_refused_once_superseded_and_leaves_no_partial_state
    return tuple(Killer(case, {"family": name, "supersession": how}) for name in names for how in supersession)


_FENCE_FAILS = (*_CONTRACT_FAILURE, TurnFenceSuperseded)

MUTANTS: dict[str, Mutant] = {mutant.name: mutant for mutant in (
    Mutant("pin_ignored", "in_memory", "the pin choosing the store: every pin shares one store",
           lambda: _opened(_PinIgnored),
           (Killer(pinning.test_pinned_handles_see_only_their_own_tenant),
            Killer(pinning.test_pinning_an_agent_never_widens_the_tenant),
            Killer(housekeeping.test_a_pinned_sweep_never_deletes_another_pins_rows),
            Killer(housekeeping.test_a_delegated_child_never_crosses_pins))),
    Mutant("pin_accepted_without_scope_pinning", "in_memory", "the refusal of a pin by a backend not declaring scope_pinning",
           lambda: _opened(_PinAcceptedUndeclared),
           (Killer(pinning.test_backend_without_pinning_refuses_a_pinned_scope),
            Killer(housekeeping.test_a_pinned_sweep_never_deletes_another_pins_rows))),
    Mutant("foreign_row_visible", "in_memory", "the pin filter on a session read",
           lambda: _opened(_ForeignRowVisible),
           (Killer(pinning.test_pinned_handles_see_only_their_own_tenant),
            Killer(housekeeping.test_a_delegated_child_never_crosses_pins))),
    Mutant("foreign_write_applied", "in_memory", "the pin filter on session writes",
           lambda: _opened(_ForeignWriteApplied),
           (Killer(pinning.test_cross_tenant_write_is_refused_or_invisible),)),
    Mutant("assert_isolating_always_passes", "sqlite", "assert_isolating's refusal on a backend that cannot isolate",
           _sqlite_isolation_claimed,
           (Killer(pinning.test_assert_isolating_matches_declared_pinning),)),
    Mutant("fence_skipped_on_append", "in_memory", "the claimed-turn fence on transcript appends",
           lambda: _opened(_AppendUnfenced),
           (Killer(turn_fence.test_append_after_newer_claim_is_refused_and_lands_nothing),
            Killer(turn_fence.test_append_after_epoch_takeover_is_refused),
            *_families("append", "append_batch", supersession=("next_claim", "takeover")))),
    Mutant("fence_skipped_on_compression_continuation", "in_memory", "the claimed-turn fence on the compression continuation",
           lambda: _opened(_CompressionUnfenced),
           (Killer(turn_fence.test_compression_continuation_write_is_fenced),
            *_families("compression_continuation", supersession=("next_claim", "takeover")))),
    Mutant("owner_only_checks_generation", "in_memory", "owner_only's exemption from the generation half",
           lambda: _opened(_OwnerOnlyChecksGeneration),
           (Killer(turn_fence.test_delegation_completion_keeps_owner_epoch_half, reason=_FENCE_FAILS),)),
    Mutant("owner_only_skips_epoch", "in_memory", "the epoch half of an owner_only fence",
           lambda: _opened(_OwnerOnlySkipsEpoch),
           (Killer(turn_fence.test_delegation_completion_keeps_owner_epoch_half),)),
    Mutant("sqlite_owner_only_skips_epoch", "sqlite", "the epoch half of an owner_only fence",
           _sqlite_owner_only_skips_epoch,
           (Killer(turn_fence.test_delegation_completion_keeps_owner_epoch_half),)),
    Mutant("store_key_filter_dropped", "in_memory", "the store-key filter: a turn's fence checked on every store",
           lambda: _opened(_StoreKeyIgnored),
           (Killer(turn_fence.test_write_to_another_store_is_not_fenced, reason=_FENCE_FAILS),)),
    Mutant("sqlite_store_key_filter_dropped", "sqlite", "the store-key filter: a turn's fence checked on every store",
           _sqlite_store_key_ignored,
           (Killer(turn_fence.test_write_to_another_store_is_not_fenced, reason=_FENCE_FAILS),)),
    Mutant("fence_checked_after_write", "in_memory", "the fence check before the write (the write is kept, then refused)",
           lambda: _opened(_FenceAfterWrite),
           (Killer(turn_fence.test_append_after_newer_claim_is_refused_and_lands_nothing),
            *_families("append_batch", "replace_transcript", "rewrite_pruned_rows", "compression_continuation",
                       "auxiliary_usage", "delegation_completion"))),
    Mutant("bound_pin_unchecked", "in_memory", "the bound hosted scope's pin checked at each verb",
           lambda: _opened(_BoundPinUnchecked),
           (Killer(activity.test_a_pinned_handle_serves_only_its_bound_pin),
            Killer(activity.test_a_local_store_serves_no_hosted_activity))),
    Mutant("sqlite_serves_hosted_activity", "sqlite", "the refusal of every read and write while a hosted scope is bound",
           _sqlite_serves_hosted_activity,
           (Killer(activity.test_a_local_store_serves_no_hosted_activity),)),
    Mutant("sqlite_own_connections_serve_hosted_activity", "sqlite",
           "the refusal of a verb that opens its own connection (the 'shared' hosted-room file) while a hosted scope is bound",
           _sqlite_own_connections_serve_hosted_activity,
           (Killer(activity.test_a_local_stores_kept_shared_hosted_rooms_serve_no_hosted_activity),)),
    Mutant("sqlite_health_serves_hosted_activity", "sqlite", "the refusal of the health service while a hosted scope is bound",
           _sqlite_health_serves_hosted_activity,
           (Killer(activity.test_a_local_stores_kept_health_serves_no_hosted_activity),)),
    Mutant("sqlite_shared_write_unfenced", "sqlite", "the claimed-turn fence on a write to the 'shared' hosted-room placement",
           _sqlite_shared_write_unfenced,
           (Killer(activity.test_a_superseded_turns_shared_placement_write_publishes_nothing),)),
    Mutant("shared_placement_unpinned", "in_memory", "the pin choosing the 'shared' hosted-room placement",
           lambda: _opened(_SharedPlacementUnpinned),
           (Killer(activity.test_pinned_shared_rooms_are_never_visible_across_pins),)),
    Mutant("fence_misses_shared_files", "in_memory", "the fence's undo of store files outside the store's namespace",
           lambda: _opened(_FenceMissesSharedFiles),
           (Killer(activity.test_a_shared_placement_write_is_fenced_after_takeover),)),
    Mutant("lease_holder_renewal_broken", "in_memory", "the renewal of an expired turn lease its holder's append owns",
           lambda: _opened(_LeaseRenewalBroken),
           (Killer(lease_holder.test_an_append_by_the_holder_renews_its_expired_lease),)),
)}
