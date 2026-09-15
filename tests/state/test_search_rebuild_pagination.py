"""Pagination applies to the combined indexed and pending-backfill result set."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("indexed_rows", [0, 2])
def test_rebuild_search_pages_partition_the_available_hits(tmp_path, monkeypatch, indexed_rows):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("search", "cli")
        expected = {
            db.append_message("search", "user", f"paginationneedle entry {i}")
            for i in range(6)
        }
        # Pause a real deferred rebuild before any chunk, or after one partial chunk.
        with db._lock:
            db._reset_fts_index_to_empty(db._conn)
            db._seed_fts_rebuild_markers(db._conn, force=True)
            db._conn.commit()
        if indexed_rows:
            monkeypatch.setattr(db, "_FTS_REBUILD_CHUNK_ROWS", indexed_rows)
            assert db.fts_rebuild_step()
        assert db.fts_rebuild_status() is not None

        def hits(limit, offset=0):
            return [row["id"] for row in db.search_messages(
                "paginationneedle", limit=limit, offset=offset, fields=("id",))]

        whole = hits(20)
        pages = [hits(2, offset) for offset in range(0, 8, 2)]
        assert set(whole) == expected
        assert [row for page in pages for row in page] == whole
        assert pages[-1] == []
        assert len(whole) == len(set(whole))

        while db.fts_rebuild_step():
            pass
        assert set(hits(20)) == expected
        assert [row for offset in range(0, 8, 2) for row in hits(2, offset)] == hits(20)
    finally:
        db.close()
