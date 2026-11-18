"""A reader opened with ``check_drift`` refuses, on its first use, a store that lacks a declared table or column, with
the reason ``StoreAdmin.schema_drift_reason`` gives, read on the reader's own connection. Without it the reader opens
such a store as before."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from hermes_cli.sqlite_safe_read import offline_file_access
from persistence.core.errors import DeclarativeDrift
from persistence_assembly import open_persistence, open_persistence_reader, open_store_admin, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite.conftest import profile_home


def _store(name: str):
    home = profile_home(name)
    with open_persistence(profile_scope(home=home), eager=True) as store:
        seed_session(store, "s", "cli")
    return home / "state.db"


def _lagging(name: str):
    path = _store(name)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
        conn.commit()
    return path


def test_a_lagging_store_is_refused_with_the_admin_reason():
    path = _lagging("alpha")
    reason = open_store_admin(path).schema_drift_reason()
    assert reason is not None and "last_read_at" in reason
    with open_persistence_reader(path, check_drift=True) as reader:
        with pytest.raises(DeclarativeDrift) as caught:
            reader.ready()
        # The refused open closed its connection: raw access to the file is not blocked by it.
        with offline_file_access(path, what="forensic-backup"):
            pass
    assert str(caught.value) == reason


def test_without_check_drift_a_lagging_store_still_opens():
    path = _lagging("alpha")
    with open_persistence_reader(path) as reader:
        reader.ready()


def test_an_up_to_date_store_opens_with_check_drift():
    path = _store("alpha")
    with open_persistence_reader(path, check_drift=True) as reader:
        assert reader.sessions.first_session_started_at() is not None
