"""What the SQLite lock verbs let through, against the legacy store.

A store replaced under a live handle raises from every compression-lock and turn-lease write: the legacy
``StateDbReplacedError`` is no ``sqlite3.Error``, so none of the legacy methods caught it, and the backend raises
``StoreReplaced``. A store failure the compression-lock writes log and answer with their empty value still raises from
the holder read, which caught nothing.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pytest

from hermes_state_errors import StateDbReplacedError
from persistence import PersistenceError, StoreReplaced
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.usefixtures("clock", "legacy_store")

# Each lock write as (legacy call, backend call); every one of them reaches the store.
LOCK_WRITES: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "try_acquire_compression_lock": (
        lambda db: db.try_acquire_compression_lock("s", "me"),
        lambda c: c.try_acquire_compression_lock(session_id="s", holder="me")),
    "refresh_compression_lock": (
        lambda db: db.refresh_compression_lock("s", "me"),
        lambda c: c.refresh_compression_lock(session_id="s", holder="me")),
    "release_compression_lock": (
        lambda db: db.release_compression_lock("s", "me"),
        lambda c: c.release_compression_lock(session_id="s", holder="me")),
    "try_acquire_turn_lease": (
        lambda db: db.try_acquire_session_turn_lease("s", "me"),
        lambda c: c.try_acquire_turn_lease(session_id="s", holder="me")),
    "refresh_turn_lease": (
        lambda db: db.refresh_session_turn_lease("s", "me"),
        lambda c: c.refresh_turn_lease(session_id="s", holder="me")),
    "release_turn_lease": (
        lambda db: db.release_session_turn_lease("s", "me"),
        lambda c: c.release_turn_lease(session_id="s", holder="me")),
}


def _replace_file(path: Path) -> None:
    """Put a copy of the store at its path, so the path no longer names the file the open handle opened."""
    copy = path.with_name("replacement.db")
    shutil.copyfile(path, copy)
    os.replace(copy, path)


@pytest.mark.parametrize("verb", sorted(LOCK_WRITES))
def test_a_replaced_store_raises_from_every_lock_write(backend, scope_a, tmp_path, verb):
    from hermes_state import SessionDB
    from hermes_state_errors import StateDbReplacedError as LegacyReplaced

    legacy_call, backend_call = LOCK_WRITES[verb]
    legacy_path = tmp_path / "legacy" / "state.db"
    legacy_path.parent.mkdir()
    db = SessionDB(db_path=legacy_path)
    try:
        db.create_session("s", "cli")
        _replace_file(legacy_path)
        with pytest.raises(LegacyReplaced):
            legacy_call(db)
    finally:
        db.close()

    store = backend.open(scope_a)
    store.coordination.reset_hygiene_failure_streak(session_key="open-the-store")
    _replace_file(store_for(scope_a))
    with pytest.raises(StoreReplaced) as caught:
        backend_call(store.coordination)
    assert isinstance(caught.value.__cause__, StateDbReplacedError)


def _drop_locks_table(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE compression_locks")


def test_a_store_failure_is_logged_by_the_lock_writes_and_raised_by_the_holder_read(backend, scope_a, tmp_path):
    from hermes_state import SessionDB

    legacy_path = tmp_path / "legacy" / "state.db"
    legacy_path.parent.mkdir()
    db = SessionDB(db_path=legacy_path)
    try:
        _drop_locks_table(legacy_path)
        legacy = (db.try_acquire_compression_lock("s", "me"), db.refresh_compression_lock("s", "me"),
                  db.release_compression_lock("s", "me"))
        with pytest.raises(sqlite3.OperationalError) as legacy_read:
            db.get_compression_lock_holder("s")
    finally:
        db.close()
    assert legacy == (False, False, None)

    coord = backend.open(scope_a).coordination
    coord.reset_hygiene_failure_streak(session_key="open-the-store")
    _drop_locks_table(store_for(scope_a))
    assert (coord.try_acquire_compression_lock(session_id="s", holder="me"),
            coord.refresh_compression_lock(session_id="s", holder="me"),
            coord.release_compression_lock(session_id="s", holder="me")) == legacy
    with pytest.raises(PersistenceError) as caught:
        coord.get_compression_lock_holder(session_id="s")
    assert type(caught.value.__cause__) is type(legacy_read.value) and str(caught.value) == str(legacy_read.value)
