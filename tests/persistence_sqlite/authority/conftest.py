"""Fixtures the authority backend tests share: a seeded profile store with its backend handle, the bound store of scope A, a frozen clock for the publishing mutations, and the fixed wall clock the coordination cases request."""

from __future__ import annotations

import itertools
import time
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tests.persistence_sqlite.authority._admission_support import EPOCH, BackendSeed, begin_epoch
from tests.persistence_sqlite.conftest import store_for

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
def seeded(backend, scope_a):
    """``seed(fn)`` builds scope A's store through the backend: its first epoch, then ``fn(db, epoch)`` with a :class:`BackendSeed` as ``db``; returns ``(path, store, values)`` where ``values`` is what ``fn`` returned."""

    def seed(fn: Callable[[Any, int], Any] | None = None):
        path = store_for(scope_a)
        values = {}

        def build(db):
            epoch = begin_epoch(db, "owner")
            assert epoch == EPOCH
            if fn is not None:
                values["v"] = fn(db, epoch)

        store = backend.open(scope_a)
        build(BackendSeed(store))
        store.ready()  # the holder is open before the test's first verb
        return path, store, values.get("v")
    return seed


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def frozen(monkeypatch) -> Callable[[], None]:
    """The publishing commands stamp a fixed clock and mint sequential ids; the returned call restarts the id sequence."""
    import gateway.session_lifecycle as lifecycle
    from persistence.sqlite.authority import _mutation_publish

    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)
    for module in (lifecycle, _mutation_publish):
        monkeypatch.setattr(module, "_now", lambda: datetime(2026, 9, 16, 10, 0, 0))
    counter = [itertools.count(1)]
    ids = SimpleNamespace(uuid4=lambda: uuid.UUID(int=next(counter[0])))
    monkeypatch.setattr(_mutation_publish, "uuid", ids)

    def restart() -> None:
        counter[0] = itertools.count(1)
    return restart
