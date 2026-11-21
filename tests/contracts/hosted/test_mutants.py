"""Every broken backend in ``_mutants.py`` is rejected by the contract case(s) it declares, for the declared reason.

For each mutant: it first survives its own-store setup (a session created, appended to and read back, and a pinned
scope opened or refused as it declares), so a mutant that crashes there is a harness error, not a kill the suite
takes credit for. Then each declared killer case is called with the mutant and must fail with the contract's own
failure (its assertion, or a ``pytest.raises`` that found no refusal, or the refusal the killer names), never a skip
and never a crash of the mutant itself (``AttributeError``, ``TypeError`` and the like). Each killer also passes on the
unbroken backend the mutant was built from, so the kill is the removed guard's.

The mutants are opened here directly; none is a registered contract backend.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from persistence import ScopePinRefused, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.contracts._reference import InMemoryBackend, open_reference_backend
from tests.contracts._scopes import pinned_scope
from tests.contracts.conftest import registered_backend_factory
from tests.contracts.hosted._mutants import MUTANTS, SCOPES, Killer, Mutant, _sqlite

# The guards the hosted contract must be shown to hold: one mutant per guard, at least.
REQUIRED = {
    "pin_ignored", "pin_accepted_without_scope_pinning", "foreign_row_visible", "foreign_write_applied",
    "assert_isolating_always_passes", "fence_skipped_on_append", "fence_skipped_on_compression_continuation",
    "owner_only_checks_generation", "owner_only_skips_epoch", "store_key_filter_dropped", "fence_checked_after_write",
    "lease_holder_renewal_broken", "bound_pin_unchecked", "sqlite_serves_hosted_activity", "shared_placement_unpinned",
    "fence_misses_shared_files", "sqlite_own_connections_serve_hosted_activity", "sqlite_health_serves_hosted_activity",
    "sqlite_shared_write_unfenced",
}

# What a mutant raising these means: the mutant (or the harness) is broken, not that a contract case caught it.
_HARNESS_ERRORS = (AttributeError, TypeError, NameError, ImportError, KeyError, RecursionError)


@contextmanager
def _unbroken(base: str) -> Iterator[Any]:
    if base == "in_memory":
        with open_reference_backend() as backend:
            yield backend
    else:
        with _sqlite() as backend:
            yield backend


def _run(killer: Killer, backend_name: str, backend: Any) -> None:
    arguments = {"backend_name": backend_name, "backend": backend, "scope_a": SCOPES[0], "scope_b": SCOPES[1], **killer.params}
    wanted = inspect.signature(killer.case).parameters
    killer.case(**{name: value for name, value in arguments.items() if name in wanted})


def _cases() -> list[Any]:
    return [pytest.param(mutant, killer, id=f"{name}-{killer.node}")
            for name, mutant in sorted(MUTANTS.items()) for killer in mutant.killers]


def test_the_registry_covers_every_required_guard_and_registers_no_backend() -> None:
    assert REQUIRED <= set(MUTANTS)
    for name, mutant in MUTANTS.items():
        assert mutant.killers, f"mutant {name!r} declares no killer"
        assert registered_backend_factory(name) is None
        assert mutant.base in ("in_memory", "sqlite") and registered_backend_factory(mutant.base) is not None


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_the_mutant_survives_its_own_store_setup(name: str) -> None:
    """The mutant serves its own profile as the unbroken backend does, and opens or refuses a pin as it declares."""
    with MUTANTS[name].open() as backend:
        store = backend.open(SCOPES[0])
        store.sessions.create(NewSession(session_id=SessionId("own"), source="cli"))
        store.messages.append(SessionId("own"), NewMessage(role="user", content="own text"))
        assert [str(row.content) for row in store.messages.read_transcript(SessionId("own"))] == ["own text"]
        pinned = pinned_scope("tenant-a", "agent-1", profile_dir=SCOPES[0].profile_dir)
        if backend.capabilities.scope_pinning or name == "pin_accepted_without_scope_pinning":
            assert backend.open(pinned).scope == pinned
        else:
            with pytest.raises(ScopePinRefused):
                backend.open(pinned)


def _judge(mutant: Mutant, killer: Killer) -> None:
    """Pass when ``killer`` passes on the unbroken base and fails against ``mutant`` for its declared reason."""
    with _unbroken(mutant.base) as backend:
        _run(killer, mutant.base, backend)
    with mutant.open() as backend:
        try:
            _run(killer, mutant.base, backend)
        except pytest.skip.Exception as skipped:  # pragma: no cover - a skip is never a kill
            pytest.fail(f"{killer.node} skipped against {mutant.name!r} instead of failing: {skipped}")
        except BaseException as failure:  # every escape is judged below
            assert not isinstance(failure, _HARNESS_ERRORS), (
                f"harness error: {mutant.name!r} crashed in {killer.node} ({failure!r}); not a kill")
            assert isinstance(failure, killer.reason), (
                f"{killer.node} failed against {mutant.name!r} with {failure!r}, not the declared {killer.reason}")
            return
    pytest.fail(f"the contract case {killer.node} passed broken backend {mutant.name!r} ({mutant.removed} removed)")


@pytest.mark.parametrize(("mutant", "killer"), _cases())
def test_the_declared_case_kills_the_mutant_for_the_declared_reason(mutant: Mutant, killer: Killer) -> None:
    _judge(mutant, killer)


class _Crashing(InMemoryBackend):
    """Breaks every verb with an AttributeError: a broken mutant, not a broken guard."""

    def fence_verb(self, store: Any, attribute: str, name: str, method: Any) -> Any:
        def crash(*args: Any, **kwargs: Any) -> Any:
            raise AttributeError("mutant setup crashed")
        return crash


def test_the_harness_counts_neither_a_survivor_nor_a_crash_as_a_kill() -> None:
    """The judge refuses an unbroken "mutant" (the case passes) and a crashing one (an AttributeError is no kill)."""
    killer = MUTANTS["fence_skipped_on_append"].killers[0]
    survivor = Mutant("unbroken", "in_memory", "nothing", open_reference_backend, (killer,))
    with pytest.raises(pytest.fail.Exception, match="passed broken backend 'unbroken'"):
        _judge(survivor, killer)

    @contextmanager
    def crashing() -> Iterator[Any]:
        backend = _Crashing()
        try:
            yield backend
        finally:
            backend.close()
    with pytest.raises(AssertionError, match="harness error"):
        _judge(Mutant("crashing", "in_memory", "everything", crashing, (killer,)), killer)
