"""``SQLiteSessionMaintenance`` on a store seeded through the backend's own session and SQL writes: the prune refusals of open rows and negative ages, and the session projection digest the TUI change watcher reads."""

from __future__ import annotations

import time
from functools import partial
from types import SimpleNamespace

import pytest

from persistence.housekeeping.maintenance import SESSION_PROJECTION_FIELDS, PruneCriteria
from persistence.sqlite.conversation._store.sessions import create_session


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
    digest = maintenance.session_projection_digest()
    assert digest is not None and digest[0] == SESSION_PROJECTION_FIELDS
    assert digest == _session_db_content_sig(store.store_path)
    # The activity columns the heartbeat restamps are not part of it; a projected column is.
    store.holder()._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = 'old-a'", (NOW,))
    assert maintenance.session_projection_digest(busy_timeout_s=1.0) == digest
    store.holder()._write_sql("UPDATE sessions SET title = 'renamed' WHERE id = 'old-a'")
    changed = maintenance.session_projection_digest()
    assert changed != digest and changed == _session_db_content_sig(store.store_path)
    # The read connection's own busy budget is restored after the probe.
    with store.holder()._read_ctx() as conn:
        budget = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    maintenance.session_projection_digest(busy_timeout_s=0.01)
    with store.holder()._read_ctx() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == budget
