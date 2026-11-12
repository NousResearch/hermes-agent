"""``SessionDB.transcript_read_snapshot``: every read inside sees one generation on a WAL store; a
DELETE-journal store (reads go through the writer connection under the write lock) is not pinned, so a
write from the same thread cannot wait on its own lock; there ``consistent()`` turns false once any write
lands, this process's or another connection's."""
import threading

import hermes_yaml as yaml

from hermes_agent.hermes_state import SessionDB


def _count(db):
    return db._read_one("SELECT COUNT(*) FROM messages WHERE session_id = 's1'")[0]


def test_reads_inside_a_snapshot_see_one_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._wal_active
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="hello")
        with db.transcript_read_snapshot() as consistent:
            assert _count(db) == 1
            db.append_message("s1", role="assistant", content="committed meanwhile")
            assert _count(db) == 1  # the snapshot's generation
            with db.transcript_read_snapshot() as inner:  # re-entrant: the same transaction
                assert _count(db) == 1 and inner()
            assert consistent()
        assert _count(db) == 2
    finally:
        db.close()


def test_a_delete_journal_store_is_not_pinned_and_any_write_makes_it_inconsistent(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    try:
        assert db._wal_active is False
        db.create_session("s1", source="cli")
        done = []

        other = SessionDB(db_path=home / "state.db")

        def read_and_write():
            with db.transcript_read_snapshot() as consistent:
                done.append((consistent.pinned, consistent()))
                db.append_message("s1", role="user", content="hello")
                done.append((_count(db), consistent()))
            with db.transcript_read_snapshot() as consistent:
                assert consistent()
                other.append_message("s1", role="assistant", content="from another connection")
                done.append((_count(db), consistent()))
        worker = threading.Thread(target=read_and_write, daemon=True)
        worker.start()
        worker.join(10)
        assert done == [(False, True), (1, False), (2, False)]
        other.close()
    finally:
        db.close()
