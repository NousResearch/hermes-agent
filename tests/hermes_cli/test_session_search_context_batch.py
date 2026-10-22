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


def test_context_batches_keep_tied_timestamp_order_and_duplicate_hits(tmp_path):
    with SessionDB(db_path=tmp_path / "ties.db") as db:
        db.create_session("ties", "cli")
        db.append_messages_batch("ties", [
            {"role": "user", "content": f"needleprobe {i}", "timestamp": 10.0}
            for i in range(502)
        ])
        rows = db.search_messages("needleprobe", limit=502)
        assert len(rows) == 502
        by_id = sorted(rows, key=lambda row: row["id"])
        for i, row in enumerate(by_id):
            expected = [f"needleprobe {j}" for j in range(max(0, i - 1), min(502, i + 2))]
            assert [r["content"] for r in row["context"]] == expected
        duplicate = db._finalize_search_matches([dict(by_id[250]), dict(by_id[250])])
        assert duplicate[0]["context"] == duplicate[1]["context"] == by_id[250]["context"]
