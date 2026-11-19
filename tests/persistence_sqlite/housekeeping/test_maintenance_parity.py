"""``SQLiteSessionMaintenance`` on a store seeded through the backend's own session and SQL writes: the prune refusals of open rows and negative ages, and the session projection digest the TUI change watcher reads."""

from __future__ import annotations

import time
from functools import partial
from types import SimpleNamespace

import pytest

from persistence.housekeeping.maintenance import SESSION_PROJECTION_FIELDS, PruneCriteria
from persistence.sqlite.conversation._store.sessions import create_session
from tests.fixtures.storage_holder import omit_session_list_revision


NOW = time.time()
DAY = 86400.0
OLD = NOW - 200 * DAY
MEDIA_ROOT = "/nonexistent/native-inputs"


def _sql(db, sql, params=()):
    db._execute_write(lambda conn: conn.execute(sql, params))


def _session(db, sid, *, source="cli", started=OLD, ended=OLD + DAY, **cols):
    """A session row with the given columns; ``ended=None`` leaves it open."""
    db.create_session(sid, source)
    values = {"started_at": started, "ended_at": ended, "end_reason": "user_exit" if ended else None,
              "last_activity_at": None, **cols}
    _sql(db, f"UPDATE sessions SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?", (*values.values(), sid))


def _message(db, sid, at=OLD, *, active=1):
    _sql(db, "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, 'user', 'x', ?, ?)",
         (sid, at, active))


@pytest.fixture
def seeded(backend, scope_a):
    """``seeded(seed)``: the backend handle over its store, the seed run on the backend's own session and SQL writes
    under the ``SessionDB`` names it calls."""
    def build(seed):
        store = backend.open(scope_a)
        holder = store.holder()
        seed(SimpleNamespace(create_session=partial(create_session, holder), _execute_write=holder._execute_write))
        return store

    return build


def _seed_selection(db):
    _session(db, "old-a", title="Alpha Report", model="GPT-X", billing_provider="OpenAI", user_id="u1", chat_id="c1",
             chat_type="dm", git_branch="feat/Login", cwd="/work/proj", input_tokens=100, output_tokens=50,
             actual_cost_usd=0.5, tool_call_count=3, message_count=4)
    _message(db, "old-a")
    _session(db, "old-b", source="telegram", title="beta", cwd="/work/proj_other", estimated_cost_usd=2.0, archived=1)
    _session(db, "old-c", pinned=1, end_reason="compression")
    _session(db, "tail-c", parent_session_id="old-c")
    _session(db, "root-d", end_reason="compression")
    _session(db, "tip-d", parent_session_id="root-d", started=NOW - DAY, ended=NOW - DAY / 2)
    _session(db, "open-e", ended=None)
    _session(db, "orphan-f", end_reason="startup_orphan_reap", ended=NOW - DAY)
    _session(db, "recent-g", started=NOW - 2 * DAY, ended=NOW - DAY)
    _session(db, "cwd-h", cwd="/work/proj/sub", title="Gamma", message_count=1)


def test_prune_refuses_open_rows_and_negative_age(seeded):
    store = seeded(_seed_selection)
    with pytest.raises(ValueError, match="include_open is archive-only"):
        store.maintenance.prune_sessions(criteria=PruneCriteria(include_open=True), media_root=MEDIA_ROOT)
    with pytest.raises(ValueError, match="older_than_days must be >= 0"):
        store.maintenance.prune_sessions(-1, media_root=MEDIA_ROOT)
    with pytest.raises(ValueError, match="older_than_days must be >= 0"):
        store.maintenance.list_prune_candidates(-1)


def test_session_projection_digest_matches_legacy_probe(seeded):
    from tui_gateway.change_watcher import _session_db_content_sig

    store = seeded(_seed_selection)
    maintenance = store.maintenance
    store.holder()._execute_write(omit_session_list_revision)
    assert maintenance.session_projection_revision() is None
    digest = maintenance.session_projection_digest()
    assert digest is not None and digest[0] == SESSION_PROJECTION_FIELDS
    _, watched_digest = _session_db_content_sig(store.store_path)
    assert digest == watched_digest
    # The activity columns the heartbeat restamps are not part of it; a projected column is.
    store.holder()._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = 'old-a'", (NOW,))
    assert maintenance.session_projection_digest(busy_timeout_s=1.0) == digest
    store.holder()._write_sql("UPDATE sessions SET title = 'renamed' WHERE id = 'old-a'")
    changed = maintenance.session_projection_digest()
    _, watched_changed = _session_db_content_sig(store.store_path)
    assert changed != digest and changed == watched_changed
    # The read connection's own busy budget is restored after the probe.
    with store.holder()._read_ctx() as conn:
        budget = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    maintenance.session_projection_digest(busy_timeout_s=0.01)
    with store.holder()._read_ctx() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == budget


@pytest.mark.parametrize('caller_snapshot', [False, True], ids=['repo-owned', 'caller-owned'])
def test_session_projection_stamp_uses_one_wal_snapshot(tmp_path, monkeypatch, caller_snapshot):
    """A committed DDL plus title change must not yield a mixed revision/cookie."""
    import sqlite3
    from contextlib import closing, contextmanager

    from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
    from tests.fixtures.storage_holder import raw_holder

    path = tmp_path / 'state.db'
    with open_persistence(profile_scope(home=tmp_path), eager=True):
        with closing(sqlite3.connect(path)) as writer, writer:
            assert writer.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
            writer.execute("INSERT INTO sessions (id, source, started_at, title) VALUES ('s', 'cli', 1, 'old')")

    def stamp(conn):
        revision = conn.execute("SELECT value FROM state_meta WHERE key='sessions_list_revision'").fetchone()[0]
        return int(revision), conn.execute('PRAGMA main.schema_version').fetchone()[0]

    with closing(sqlite3.connect(path)) as writer, open_persistence_reader(path) as reader:
        old = stamp(writer)
        holder = raw_holder(reader)
        native = holder._read_ctx_with_timeout
        commits = []
        after_probe = []

        def commit_between_reads(statement):
            if statement.casefold() != 'pragma main.schema_version' or commits:
                return
            with writer:
                writer.execute("UPDATE sessions SET title='new' WHERE id='s'")
                writer.execute('CREATE TABLE extra_schema_object (id INTEGER)')
            commits.append(stamp(writer))

        @contextmanager
        def interleave(timeout):
            with native(timeout) as conn:
                if caller_snapshot:
                    conn.execute('BEGIN')
                    assert stamp(conn) == old
                conn.set_trace_callback(commit_between_reads)
                try:
                    yield conn
                    after_probe.append(conn.in_transaction)
                    if caller_snapshot:
                        assert stamp(conn) == old
                finally:
                    conn.set_trace_callback(None)
                    if conn.in_transaction:
                        conn.rollback()

        with monkeypatch.context() as schedule:
            schedule.setattr(holder, '_read_ctx_with_timeout', interleave)
            observed = reader.maintenance.session_projection_revision()
        assert observed == old
        assert commits and commits[0][0] != old[0] and commits[0][1] != old[1]
        assert after_probe == [caller_snapshot]
        assert reader.maintenance.session_projection_revision() == commits[0]


def test_reinstalled_revision_trigger_invalidates_a_stamp_read_before_the_gap(tmp_path):
    """A rename made while a revision trigger was missing is not hidden behind the old stamp."""
    import sqlite3
    from contextlib import closing

    from persistence.sqlite._schema_ddl import SESSIONS_LIST_REVISION_TRIGGERS
    from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

    path = tmp_path / 'state.db'
    with open_persistence(profile_scope(home=tmp_path), eager=True):
        with closing(sqlite3.connect(path)) as writer, writer:
            writer.execute("INSERT INTO sessions (id, source, started_at, title) VALUES ('s', 'cli', 1, 'old')")
    with open_persistence_reader(path) as reader:
        before = reader.maintenance.session_projection_revision()
    with closing(sqlite3.connect(path)) as writer, writer:
        writer.execute(f'DROP TRIGGER {SESSIONS_LIST_REVISION_TRIGGERS[2]}')
        writer.execute("UPDATE sessions SET title = 'renamed' WHERE id = 's'")
    with open_persistence_reader(path) as reader:
        assert reader.maintenance.session_projection_revision() is None  # the gap reads as unsupported
    with open_persistence(profile_scope(home=tmp_path), eager=True):
        pass
    with open_persistence_reader(path) as reader:
        after = reader.maintenance.session_projection_revision()
    assert before is not None and after is not None and after != before
