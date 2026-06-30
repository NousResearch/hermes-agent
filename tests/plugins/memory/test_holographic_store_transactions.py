import sqlite3

from plugins.memory.holographic.store import MemoryStore


def test_duplicate_add_does_not_leave_write_transaction_open(tmp_path):
    db_path = tmp_path / "memory_store.db"
    store = MemoryStore(db_path=db_path, hrr_dim=64)
    store._hrr_available = False

    try:
        fact_id = store.add_fact("Jordan prefers concise docs")
        assert store.add_fact("Jordan prefers concise docs") == fact_id
        assert not store._conn.in_transaction

        other = sqlite3.connect(str(db_path), timeout=0.1)
        try:
            other.execute(
                "INSERT INTO facts (content) VALUES (?)",
                ("A separate connection can still write",),
            )
            other.commit()
        finally:
            other.close()
    finally:
        store.close()
