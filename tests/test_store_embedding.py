import pytest
import tempfile
from pathlib import Path
from plugins.memory.holographic.store import MemoryStore


class TestStoreEmbedding:
    def test_add_fact_with_embedding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            fact_id = store.add_fact("test content", category="test")
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            assert row is not None
            assert "embedding" in row.keys()
            store.close()
    
    def test_backfill_existing_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MemoryStore(db_path=str(db_path))
            store._conn.execute(
                "INSERT INTO facts (content, category) VALUES (?, ?)",
                ("old fact", "test")
            )
            store._conn.commit()
            count = store.backfill_existing_facts()
            assert count == 1
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE content = ?", ("old fact",)
            ).fetchone()
            assert row["embedding"] is not None
            store.close()
