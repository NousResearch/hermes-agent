"""``messages.read_snapshot`` on SQLite: every read inside sees one generation on a WAL store; a
DELETE-journal store (reads go through the writer connection under the write lock) is not pinned, so a
write from the same thread cannot wait on its own lock; there ``consistent()`` turns false once any write
lands, this process's or another connection's."""
import sqlite3
import threading

import hermes_yaml as yaml

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


def _count(store):
    return raw_holder(store)._read_one("SELECT COUNT(*) FROM messages WHERE session_id = 's1'")[0]


def _append(store, role, content):
    seed_messages(store, "s1", {"role": role, "content": content})


def test_reads_inside_a_snapshot_see_one_generation(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "s1", "cli")
        assert raw_holder(store)._wal_active
        _append(store, "user", "hello")
        with store.messages.read_snapshot() as consistent:
            assert _count(store) == 1
            _append(store, "assistant", "committed meanwhile")
            assert _count(store) == 1  # the snapshot's generation
            with store.messages.read_snapshot() as inner:  # re-entrant: the same transaction
                assert _count(store) == 1 and inner()
            assert consistent()
        assert _count(store) == 2
    finally:
        store.close()


def test_a_delete_journal_store_is_not_pinned_and_any_write_makes_it_inconsistent(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = open_persistence(profile_scope(home=home))
    try:
        seed_session(store, "s1", "cli")
        assert raw_holder(store)._wal_active is False
        done = []

        # Another connection on the same file (one holder serves every handle of a path in this process).
        other = sqlite3.connect(str(home / "state.db"), timeout=10, check_same_thread=False)

        def read_and_write():
            with store.messages.read_snapshot() as consistent:
                done.append((consistent.pinned, consistent()))
                _append(store, "user", "hello")
                done.append((_count(store), consistent()))
            with store.messages.read_snapshot() as consistent:
                assert consistent()
                with other:
                    other.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'assistant', ?, 0)",
                                  ("from another connection",))
                done.append((_count(store), consistent()))
        worker = threading.Thread(target=read_and_write, daemon=True)
        worker.start()
        worker.join(10)
        assert done == [(False, True), (1, False), (2, False)]
        other.close()
    finally:
        store.close()
