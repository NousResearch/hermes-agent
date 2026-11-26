"""Batched context preserves neighboring rows and projection fast paths."""

from persistence.sqlite.conversation._store.messages import append_message, append_messages_batch
from persistence.sqlite.conversation._store.search import _finalize_search_matches, search_messages
from persistence.sqlite.conversation._store.sessions import create_session
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


def test_search_context_batches_hits_without_cross_session_neighbors(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        db = raw_holder(store)
        for sid in ("one", "two"):
            create_session(db, sid, "cli")
            for i in range(5):
                append_message(db, sid, "user", f"{sid} needleprobe {i}")
        sql = []
        with db._read_ctx() as conn:
            conn.set_trace_callback(sql.append)
        rows = search_messages(db, "needleprobe", limit=20)
        assert len(rows) == 10
        for row in rows:
            contents = [r["content"] for r in row["context"]]
            assert all(c.startswith(row["session_id"]) for c in contents)
            indices = [int(c.rsplit(" ", 1)[1]) for c in contents]
            assert indices == list(range(indices[0], indices[-1] + 1))
            assert len(indices) in (2, 3)
        assert sum("WITH target AS" in q for q in sql) == 1
        sql.clear()
        projected = search_messages(db, "needleprobe", fields=["id"], limit=20)
        assert len(projected) == len(rows)
        assert not any("WITH target AS" in q for q in sql)


def test_context_batches_keep_tied_timestamp_order_and_duplicate_hits(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        db = raw_holder(store)
        create_session(db, "ties", "cli")
        append_messages_batch(db, "ties", [
            {"role": "user", "content": f"needleprobe {i}", "timestamp": 10.0}
            for i in range(502)
        ])
        rows = search_messages(db, "needleprobe", limit=502)
        assert len(rows) == 502
        by_id = sorted(rows, key=lambda row: row["id"])
        for i, row in enumerate(by_id):
            expected = [f"needleprobe {j}" for j in range(max(0, i - 1), min(502, i + 2))]
            assert [r["content"] for r in row["context"]] == expected
        duplicate = _finalize_search_matches(db, [dict(by_id[250]), dict(by_id[250])])
        assert duplicate[0]["context"] == duplicate[1]["context"] == by_id[250]["context"]
