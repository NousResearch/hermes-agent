"""Fixtures the authority backend tests share: a seeded profile store with its backend handle, deterministic ids for
both stores, the bound store of scope A, a frozen clock for the publishing mutations, the delete events the legacy
commit hands its plugin hook, and the fixed wall clock the coordination parity cases request, so the legacy store and
the backend see one time."""

from __future__ import annotations

import itertools
import time
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from persistence.authority import runtime as rt
from tests.persistence_sqlite._legacy_store import ORACLE_TREE, legacy_import, legacy_store
from tests.persistence_sqlite.authority._admission_support import EPOCH, BackendSeed, begin_epoch
from tests.persistence_sqlite.authority._support import seed_legacy
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

NOW = 1_800_000_000.0


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fixed = Clock()
    monkeypatch.setattr(time, "time", fixed)
    return fixed


@pytest.fixture
def fixed_ids(monkeypatch) -> Callable[[], None]:
    """Make ``uuid4`` deterministic for admission ids and message uids (the legacy and the backend modules that mint
    them import it by name); the returned function restarts the sequence (call it before each run)."""
    import agent.message_metadata
    import session_rules.turns

    counter = {"it": itertools.count(1)}

    def fake() -> uuid.UUID:
        return uuid.UUID(int=next(counter["it"]))

    for module in (uuid, agent.message_metadata, session_rules.turns):
        monkeypatch.setattr(module, "uuid4", fake)

    def restart() -> None:
        counter["it"] = itertools.count(1)

    return restart


@pytest.fixture
def seeded(request, backend, scope_a):
    """``seed(fn)`` builds scope A's store: its first epoch, then ``fn(db, epoch)``; returns ``(path, store, values)``
    where ``values`` is what ``fn`` returned. A test that compares with the legacy store (it asks for
    ``legacy_store``) builds it with the legacy store as ``db``; any other test builds it through the backend, with a
    :class:`BackendSeed` as ``db``."""
    legacy = "legacy_store" in request.fixturenames

    def seed(fn: Callable[[Any, int], Any] | None = None):
        path = store_for(scope_a)
        values = {}

        def build(db):
            epoch = begin_epoch(db, "owner")
            assert epoch == EPOCH
            if fn is not None:
                values["v"] = fn(db, epoch)

        if legacy:
            seed_legacy(path, build)
            store = backend.open(scope_a)
        else:
            store = backend.open(scope_a)
            build(BackendSeed(store))
        store.ready()  # the holder is open before a run, as the legacy store is before its function runs
        return path, store, values.get("v")
    return seed


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def frozen(monkeypatch) -> Callable[[], None]:
    """Both runs stamp the same clock and the publishing commands mint the same ids (the legacy store's commands only
    when the checkout is named); the returned call restarts the id sequence."""
    import gateway.session_lifecycle as lifecycle
    from persistence.sqlite.authority import _mutation_publish

    minting = [_mutation_publish]
    if ORACLE_TREE is not None:
        minting += [legacy_import(f"hermes_state_mutation_{name}") for name in ("branch", "compress", "reset")]
    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)
    for module in (lifecycle, _mutation_publish):
        monkeypatch.setattr(module, "_now", lambda: datetime(2026, 9, 16, 10, 0, 0))
    counter = [itertools.count(1)]
    ids = SimpleNamespace(uuid4=lambda: uuid.UUID(int=next(counter[0])))
    for module in minting:
        monkeypatch.setattr(module, "uuid", ids)

    def restart() -> None:
        counter[0] = itertools.count(1)
    return restart


@pytest.fixture
def emitted(monkeypatch, legacy_store) -> list[rt.SessionGone]:
    """The events the legacy commit hands ``emit_session_deleted``."""
    import hermes_state_common

    events: list[rt.SessionGone] = []
    monkeypatch.setattr(hermes_state_common, "emit_session_deleted",
                        lambda db, ids, *, reason, primary=None, sessions_dir=None: events.append(rt.SessionGone(tuple(ids), reason)))
    return events
