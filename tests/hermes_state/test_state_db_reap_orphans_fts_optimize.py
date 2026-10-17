"""The standing state.db maintenance must reap orphan message rows and merge both FTS5 indexes.

``messages.session_id`` references ``sessions(id)`` with no ``ON DELETE CASCADE``, so a
session row removed outside ``prune_sessions`` strands its whole transcript in ``messages``
and in BOTH FTS5 shadow indexes (``messages_fts`` + ``messages_fts_trigram``).  Those rows
had no reaper and no retention rule (they belong to no session to grow old), and the FTS
merge only ever ran inside ``vacuum()`` — which the freelist-ratio gate skips exactly on a
dense store.  These tests pin the two behaviours the maintenance path now owns, plus the
per-run record it writes.  All three are red on the pre-fix base (``reap_orphan_messages``
does not exist, no ``fts_indexes_optimized`` key, no ``last_state_db_maintenance`` row).
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _message_ids(db: SessionDB, session_id: str) -> list:
    return [row[0] for row in
            db._conn.execute("SELECT id FROM messages WHERE session_id = ?", (session_id,))]


def _count(db: SessionDB, where: str, params: tuple) -> int:
    return int(db._conn.execute(f"SELECT count(*) FROM messages WHERE {where}", params).fetchone()[0])


def _fts_match(db: SessionDB, table: str, term: str) -> int:
    return int(db._conn.execute(
        f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", (term,)).fetchone()[0])


def test_orphan_messages_are_reaped_but_a_live_session_is_spared(db):
    db.create_session("live", source="cli")
    db.append_message("live", role="user", content="still here")
    db.create_session("dead", source="cli")
    db.append_message("dead", role="user", content="stranded one")
    db.append_message("dead", role="assistant", content="stranded two")

    assert len(_message_ids(db, "dead")) == 2
    assert _fts_match(db, "messages_fts", "stranded") == 2

    # Remove the session row the ONLY way an orphan can be born on a store that enforces
    # foreign keys (the default): with FK enforcement OFF — a legacy importer, an older
    # writer that never issued ``PRAGMA foreign_keys=ON``, or a bulk cleanup. The FK has no
    # ON DELETE CASCADE, so with enforcement ON this DELETE is refused and no orphan exists.
    db._conn.execute("PRAGMA foreign_keys = OFF")
    db._conn.execute("DELETE FROM sessions WHERE id = ?", ("dead",))
    db._conn.commit()
    db._conn.execute("PRAGMA foreign_keys = ON")

    reaped = db.reap_orphan_messages()

    assert reaped == 2
    assert _count(db, "session_id = ?", ("dead",)) == 0
    # The live session's transcript is untouched.
    assert _count(db, "session_id = ?", ("live",)) == 1
    # Both FTS5 indexes drop the orphans too: the delete triggers own them.
    assert _fts_match(db, "messages_fts", "stranded") == 0
    assert _fts_match(db, "messages_fts_trigram", "stranded") == 0
    assert _fts_match(db, "messages_fts", "still") == 1
    # A no-op reap on a store with no orphans returns 0 and deletes nothing.
    assert db.reap_orphan_messages() == 0
    assert _count(db, "session_id = ?", ("live",)) == 1


def test_maintenance_optimizes_both_fts_indexes_even_when_vacuum_is_off(db, monkeypatch):
    db.create_session("s", source="cli")
    db.append_message("s", role="user", content="hello world")

    calls: list = []
    real_optimize = db.optimize_fts
    monkeypatch.setattr(db, "optimize_fts", lambda: (calls.append(1), real_optimize())[1])

    result = db.maybe_auto_prune_and_vacuum(
        retention_days=0, min_interval_hours=0, vacuum=False)

    # The FTS merge runs as its OWN maintenance step, not only inside vacuum().
    assert calls == [1]
    assert result["fts_indexes_optimized"] >= 2  # messages_fts + messages_fts_trigram
    assert result["reaped_orphan_messages"] == 0
    # The index is still queryable after the merge (a merged, not broken, index).
    assert _fts_match(db, "messages_fts", "hello") == 1


def test_maintenance_writes_a_recorded_row_per_run(db, tmp_path):
    db.create_session("s", source="cli")
    db.append_message("s", role="user", content="hello")

    db.maybe_auto_prune_and_vacuum(retention_days=5, min_interval_hours=0, vacuum=False)

    raw = db.get_meta("last_state_db_maintenance")
    assert raw, "the run must record its counters and resulting store geometry"
    record = json.loads(raw)
    assert record["retention_days"] == 5
    assert record["reaped_orphan_messages"] == 0
    assert record["fts_indexes_optimized"] >= 2
    assert record["page_count"] and record["page_size"]
    assert record["freelist_count"] is not None

    # And the per-run history line agrees with the last-run row.
    lines = [json.loads(line) for line in
             (tmp_path / "logs" / "state-db-maintenance.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert lines and lines[-1]["page_count"] == record["page_count"]
