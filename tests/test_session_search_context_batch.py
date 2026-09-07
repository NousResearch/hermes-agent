"""Batched context preserves neighboring rows and projection fast paths."""

from hermes_state import SessionDB


def test_search_context_batches_hits_without_cross_session_neighbors(tmp_path):
    with SessionDB(db_path=tmp_path / "search.db") as db:
        for sid in ("one", "two"):
            db.create_session(sid, "cli")
            for i in range(5):
                db.append_message(sid, "user", f"{sid} needleprobe {i}")
        sql = []
        with db._read_ctx() as conn:
            conn.set_trace_callback(sql.append)
        rows = db.search_messages("needleprobe", limit=20)
        assert len(rows) == 10
        for row in rows:
            contents = [r["content"] for r in row["context"]]
            assert all(c.startswith(row["session_id"]) for c in contents)
            indices = [int(c.rsplit(" ", 1)[1]) for c in contents]
            assert indices == list(range(indices[0], indices[-1] + 1))
            assert len(indices) in (2, 3)
        assert sum("WITH target AS" in q for q in sql) == 1
        sql.clear()
        projected = db.search_messages("needleprobe", fields=["id"], limit=20)
        assert len(projected) == len(rows)
        assert not any("WITH target AS" in q for q in sql)
