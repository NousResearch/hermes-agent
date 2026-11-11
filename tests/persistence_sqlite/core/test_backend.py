"""The assembled SQLite backend: handles bound to one profile, one registry holder per file, read-only assemblies,
sweep recovery, error translation and the executor behind the awaitable form."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
import shutil
import sqlite3
import threading

import pytest

from persistence import (
    PermanentStoreError, ProfileScope, StoreContention, StoreQuarantined, StoreReplaced, StoreUnavailable,
)
from persistence.sqlite import SQLiteBackend, translated
from persistence.sqlite import _registry
from hermes_state_errors import StateDbCorruptError, StateDbReplacedError
from tests.persistence_sqlite.conftest import store_for


def test_a_profile_the_backend_holds_no_store_for_is_refused(backend):
    foreign = ProfileScope(profile_dir="/contract-homes/elsewhere", name="elsewhere")
    with pytest.raises(ValueError):
        backend.open(foreign)
    assert _registry.stats()["live_generations"] == 0


def test_opening_a_handle_touches_no_file(backend, scope_a):
    store = backend.open(scope_a)
    assert store.scope == scope_a and store.store_path == store_for(scope_a)
    assert not store_for(scope_a).exists()
    assert _registry.stats()["live_generations"] == 0


def test_verbs_take_no_scope(backend, scope_a):
    store = backend.open(scope_a)
    for repository in (store.state_meta, store.health):
        for name, verb in inspect.getmembers(repository, inspect.ismethod):
            if not name.startswith("_"):
                assert "scope" not in inspect.signature(verb).parameters, name


def test_each_profile_has_its_own_file(backend, scope_a, scope_b):
    backend.open(scope_a).state_meta.set("k", "a")
    assert store_for(scope_a).exists() and not store_for(scope_b).exists()
    row = sqlite3.connect(store_for(scope_a)).execute("SELECT value FROM state_meta WHERE key = 'k'").fetchone()
    assert row == ("a",)


def test_handles_and_backends_on_one_file_share_the_registry_holder(scope_a, scope_b, backend):
    other = SQLiteBackend({scope_a.hermes_home_key: store_for(scope_a)})
    try:
        first, second = backend.open(scope_a), backend.open(scope_a)
        first.state_meta.set("k", "1")
        other.open(scope_a).state_meta.set("k", "2")
        assert first.holder() is second.holder() is other.open(scope_a).holder()
        assert _registry.stats() == {"live_generations": 1, "retired_generations": 0, "total_refcounts": 2}
        assert second.state_meta.get("k") == "2"
    finally:
        other.close()
    assert _registry.stats()["total_refcounts"] == 1
    backend.close()
    assert _registry.stats() == {"live_generations": 0, "retired_generations": 0, "total_refcounts": 0}


def test_reopen_drops_the_reference_and_keeps_the_data(backend, scope_a):
    store = backend.open(scope_a)
    store.state_meta.set("k", "kept")
    backend.reopen()
    assert _registry.stats()["live_generations"] == 0
    assert store.state_meta.get("k") == "kept"


def test_a_swept_holder_is_acquired_again(backend, scope_a):
    store = backend.open(scope_a)
    store.state_meta.set("k", "before")
    swept = store.holder()
    assert _registry.close_all_under(store_for(scope_a).parent) == 1
    store.state_meta.set("k", "after")
    assert store.holder() is not swept
    assert store.state_meta.get("k") == "after"


@pytest.mark.platforms("posix")
def test_a_handle_opened_after_its_file_is_replaced_writes_the_replacement_while_older_handles_stay_fenced(backend, scope_a):
    old = backend.open(scope_a)
    old.state_meta.set("k", "original")
    path = store_for(scope_a)
    copy = path.with_name("replacement.db")
    shutil.copyfile(path, copy)
    os.replace(copy, path)
    new = backend.open(scope_a)
    new.state_meta.set("k", "new")
    assert new.state_meta.get("k") == "new"
    with pytest.raises(StoreReplaced):
        old.state_meta.set("k", "old")
    assert sqlite3.connect(path).execute("SELECT value FROM state_meta WHERE key = 'k'").fetchone() == ("new",)
    backend.close()
    assert _registry.stats() == {"live_generations": 0, "retired_generations": 0, "total_refcounts": 0}


def test_close_refuses_further_use(backend, scope_a):
    store = backend.open(scope_a)
    store.state_meta.set("k", "v")
    backend.close()
    with pytest.raises(RuntimeError):
        store.state_meta.get("k")
    with pytest.raises(RuntimeError):
        backend.open(scope_a)


def test_read_only_backend_reads_refuses_writes_and_never_creates(backend, scope_a, scope_b):
    backend.open(scope_a).state_meta.set("k", "v")
    backend.close()
    reader = SQLiteBackend({s.hermes_home_key: store_for(s) for s in (scope_a, scope_b)}, read_only=True)
    try:
        store_a = reader.open(scope_a)
        assert store_a.read_only and store_a.state_meta.get("k") == "v"
        with pytest.raises(PermanentStoreError):
            store_a.state_meta.set("k", "w")
        assert store_a.state_meta.get("k") == "v"
        with pytest.raises(StoreUnavailable):
            reader.open(scope_b).state_meta.get("k")
        assert not store_for(scope_b).exists()
        assert _registry.stats()["live_generations"] == 0
    finally:
        reader.close()


def test_list_prefix_answers_the_store_like_rows_in_table_order(backend, scope_a):
    # The store's LIKE matches 'LOOP:x' for 'loop:' too, and the rows come back in the order they were stored.
    repository = backend.open(scope_a).state_meta
    for key in ("loop:b", "LOOP:x", "loop:a", "loopy"):
        repository.set(key, key.upper())
    repository.set("loop:b", "again")
    assert [entry.key for entry in repository.list_prefix("loop:")] == ["loop:b", "LOOP:x", "loop:a"]


@pytest.mark.parametrize(("raised", "expected"), [
    (StateDbReplacedError("replaced"), StoreReplaced),
    (StateDbCorruptError("corrupt"), StoreQuarantined),
    (sqlite3.OperationalError("database is locked"), StoreContention),
    (sqlite3.OperationalError("attempt to write a readonly database"), PermanentStoreError),
    (sqlite3.OperationalError("unable to open database file"), StoreUnavailable),
])
def test_store_errors_cross_the_boundary_typed(raised, expected):
    with pytest.raises(expected) as caught:
        with translated():
            raise raised
    assert caught.value.__cause__ is raised


def test_value_errors_pass_through_untranslated():
    with pytest.raises(ValueError) as caught:
        with translated():
            raise ValueError("bad input")
    assert type(caught.value) is ValueError


_marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")


def test_awaitable_form_runs_on_the_backend_executor_with_the_callers_context(backend, scope_a, monkeypatch):
    store = backend.open(scope_a)
    store.state_meta.set("k", "v")
    holder, seen = store.holder(), []
    real_get_meta = holder.get_meta

    def get_meta(key):
        seen.append((threading.current_thread().name, threading.get_ident(), _marker.get()))
        return real_get_meta(key)

    monkeypatch.setattr(holder, "get_meta", get_meta)

    async def call():
        _marker.set("caller")
        return threading.get_ident(), await store.aio.state_meta.get("k")

    loop_thread, value = asyncio.run(call())
    [(name, ident, marker)] = seen
    assert name.startswith("hermes-sqlite-store") and ident != loop_thread
    assert (marker, value) == ("caller", "v")
    assert _marker.get() == "unset"


def test_awaitable_form_returns_the_verbs_exception(backend, scope_a):
    backend.open(scope_a).state_meta.set("k", "v")
    backend.close()
    reader = SQLiteBackend({scope_a.hermes_home_key: store_for(scope_a)}, read_only=True)
    try:
        store = reader.open(scope_a)
        with pytest.raises(ValueError):
            asyncio.run(store.aio.state_meta.get(""))
        with pytest.raises(PermanentStoreError) as refused:
            asyncio.run(store.aio.state_meta.set("k", "w"))
        assert isinstance(refused.value.__cause__, sqlite3.OperationalError)
    finally:
        reader.close()


def test_awaitable_calls_share_one_bounded_pool(backend, scope_a, monkeypatch):
    store = backend.open(scope_a)
    store.state_meta.set("k", "v")
    holder, names = store.holder(), set()
    real_get_meta = holder.get_meta

    def get_meta(key):
        names.add(threading.current_thread().name)
        return real_get_meta(key)

    monkeypatch.setattr(holder, "get_meta", get_meta)

    async def burst():
        return await asyncio.gather(*(store.aio.state_meta.get("k") for _ in range(64)))

    assert asyncio.run(burst()) == ["v"] * 64
    assert names and all(name.startswith("hermes-sqlite-store") for name in names)
    assert len(names) <= min(32, (os.cpu_count() or 1) + 4)
