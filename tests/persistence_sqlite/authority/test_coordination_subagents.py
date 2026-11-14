"""The SQLite subagent records (``resumable_subagents``): the table main's schema declares, created on every writable
open, and the records of a profile whose store was removed, which no write or read creates again."""

from __future__ import annotations

import re
import sqlite3

import hermes_state_common
from hermes_constants import get_hermes_home, profile_tombstone_path
from persistence.authority.subagents import SubagentState, SubagentWrite
from persistence.sqlite import SQLiteBackend
from tests.contracts._scopes import profile_scope
from tests.persistence_sqlite.conftest import store_for


def _columns(conn: sqlite3.Connection) -> list[tuple]:
    return [tuple(row) for row in conn.execute("PRAGMA table_info('resumable_subagents')")]


def _main_columns() -> list[tuple]:
    statement = re.search(r"CREATE TABLE IF NOT EXISTS resumable_subagents \(.*?\);", hermes_state_common.SCHEMA_SQL,
                          re.DOTALL)
    assert statement is not None
    with sqlite3.connect(":memory:") as conn:
        conn.execute(statement.group(0))
        return _columns(conn)


def test_a_fresh_store_has_mains_resumable_subagents_table(store, scope_a):
    store.ready()
    with sqlite3.connect(store_for(scope_a)) as conn:
        assert _columns(conn) == _main_columns()
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='resumable_subagents' "
                               "AND sql IS NOT NULL").fetchall()
    assert indexes == []


def test_a_removed_profiles_records_are_never_created_again():
    removed = profile_scope("removed")
    home = get_hermes_home() / "profiles" / removed.name
    tomb = profile_tombstone_path(home)
    tomb.parent.mkdir(parents=True, exist_ok=True)
    tomb.write_text("deleted")
    backend = SQLiteBackend({removed.hermes_home_key: home / "state.db"})
    try:
        coord = backend.open(removed).coordination
        record = SubagentWrite(state=SubagentState.RUNNING, pinned=True, stop_cause=None, owner_pid=1,
                               owner_started_at=1, document={"goal": "g"}, expires_at=None)
        # A late write answers as a write that did nothing; the reads answer no row; none creates the profile.
        assert coord.put_subagent(subagent_id="sa", write=record, expected=None) is False
        assert coord.checkpoint_subagent(subagent_id="sa", cwd="/w", repo_baselines={}) is False
        assert coord.mark_subagents_gateway_shutdown(owner_pid=1) == 0
        assert coord.supersede_subagents_for_session(note="n", session_key="sk") == 0
        assert coord.get_subagent(subagent_id="sa") is None
        assert coord.list_pinned_subagents(max_resumable=5) == ()
        assert coord.has_owed_subagents() is False
        assert not home.exists()
    finally:
        backend.close()


def test_a_store_without_the_table_gains_it_on_open(backend, scope_a):
    backend.open(scope_a).ready()
    path = store_for(scope_a)
    backend.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE resumable_subagents")
    reopened = SQLiteBackend({scope_a.hermes_home_key: path})
    try:
        reopened.open(scope_a).ready()
        with sqlite3.connect(path) as conn:
            assert _columns(conn) == _main_columns()
    finally:
        reopened.close()
