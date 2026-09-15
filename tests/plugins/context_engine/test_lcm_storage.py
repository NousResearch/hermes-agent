"""Storage contract: raw-message retention, FTS, and paging bounds.

The invariants here are the ones the acceptance criteria name directly:
messages are append-only, paging is bounded by the store (not the caller), and
search is bounded and confined to one session.
"""

from __future__ import annotations

from plugins.context_engine.lcm.storage import LCMStore


def _msgs(prefix: str, count: int, role: str = "user") -> list[dict]:
    return [{"role": role, "content": f"{prefix} {index}"} for index in range(count)]


def _store(tmp_path, **kwargs) -> LCMStore:
    defaults = dict(max_page_size=5, page_size=3, body_chars=2_000, max_search_results=3)
    defaults.update(kwargs)
    return LCMStore(tmp_path / "lcm.db", **defaults).open()


def test_store_records_are_append_only_and_positionally_aligned(tmp_path):
    store = _store(tmp_path)
    messages = [{"role": "system", "content": "sys"}] + _msgs("u", 4)
    ids = store.sync_transcript("s1", messages)

    assert len(ids) == len(messages)
    assert all(ids), "every appended message gets an id"
    assert len(set(ids)) == len(ids), "ids are unique"

    # Re-syncing the same transcript must not duplicate rows...
    again = store.sync_transcript("s1", messages)
    assert again == ids, "overlap keeps the original ids"
    assert store.stats("s1")["messages"] == len(messages)

    # ...and a longer transcript only appends the new tail.
    longer = messages + _msgs("tail", 2, role="assistant")
    longer_ids = store.sync_transcript("s1", longer)
    assert longer_ids[: len(ids)] == ids
    assert all(longer_ids[len(ids):])
    assert store.stats("s1")["messages"] == len(longer)


def test_store_sessions_are_isolated(tmp_path):
    store = _store(tmp_path)
    store.sync_transcript("s1", _msgs("alpha", 2))
    store.sync_transcript("s2", _msgs("beta", 2))

    assert store.stats("s1")["messages"] == 2
    assert store.stats("s2")["messages"] == 2
    assert store.search("s1", "beta") == []
    assert store.search("s2", "alpha") == []


def test_a_partial_new_tail_is_appended_not_dropped(tmp_path):
    """A caller handing us only the new messages must not lose them."""
    store = _store(tmp_path)
    store.sync_transcript("s1", [{"role": "user", "content": "first"}])

    ids = store.sync_transcript("s1", [{"role": "assistant", "content": "second"}])
    assert ids == [""] or ids[0], "the row is recorded"
    assert store.stats("s1")["messages"] == 2
    assert store.search("s1", "second")


def test_a_rewritten_transcript_does_not_duplicate_recorded_history(tmp_path):
    """A post-compaction transcript re-read must not re-append what survives."""
    store = _store(tmp_path)
    original = [{"role": "user", "content": f"row {i}"} for i in range(6)]
    store.sync_transcript("s1", original)
    before = store.stats("s1")["messages"]

    rewritten = [original[0], {"role": "assistant", "content": "[summary]"}, original[5]]
    store.sync_transcript("s1", rewritten)

    assert store.stats("s1")["messages"] == before, "no duplicated history"


def test_page_window_is_bounded_and_resumable(tmp_path):
    store = _store(tmp_path, max_page_size=5, page_size=4)
    store.sync_transcript("s1", _msgs("row", 11))

    first = store.page_window("s1")
    assert first["returned"] == 4
    assert first["page_size"] == 4
    assert first["total"] == 11
    assert first["has_more"] is True
    assert first["next_cursor"] == first["items"][-1]["seq"]

    seen = [row["seq"] for row in first["items"]]
    cursor = first["next_cursor"]
    while cursor:
        window = store.page_window("s1", after_seq=cursor)
        assert window["returned"] <= 4
        seen.extend(row["seq"] for row in window["items"])
        cursor = window["next_cursor"] if window["has_more"] else None

    assert seen == sorted(seen)
    assert len(seen) == 11


def test_page_size_request_is_clamped_to_the_ceiling(tmp_path):
    store = _store(tmp_path, max_page_size=5, page_size=2)
    store.sync_transcript("s1", _msgs("row", 12))

    window = store.page_window("s1", page_size=10_000)
    assert window["page_size"] == 5, "an oversized request is clamped, not honored"
    assert window["returned"] <= 5


def test_full_text_search_is_bounded_and_matches_body(tmp_path):
    store = _store(tmp_path, max_search_results=2)
    assert store.fts_available() is True, "the FTS5 index is part of the schema"
    store.sync_transcript(
        "s1",
        [
            {"role": "user", "content": "the migration ledger keeps lineage"},
            {"role": "assistant", "content": "lineage edges are per source"},
            {"role": "user", "content": "unrelated chatter about lunch"},
        ],
    )

    hits = store.search("s1", "lineage")
    assert len(hits) == 2, "capped by max_search_results even though more match"
    assert all(hit["match"] == "fts" for hit in hits)
    assert {hit["message_id"] for hit in hits} <= {
        row["message_id"] for row in store.page_window("s1", page_size=5)["items"]
    }

    assert store.search("s1", "lunch")[0]["body"].startswith("unrelated")
    # Operator-ish input must not raise (FTS5 syntax is neutralised).
    assert store.search("s1", 'lineage AND "(" OR *') is not None


def test_like_fallback_still_returns_bounded_results(tmp_path):
    store = _store(tmp_path, max_search_results=2)
    store.sync_transcript("s1", _msgs("needle", 4))
    store._fts_available = False  # simulate a host without the FTS extension

    hits = store.search("s1", "needle")
    assert len(hits) == 2
    assert all(hit["match"] == "like" for hit in hits)


def test_mark_compacted_flags_without_deleting(tmp_path):
    store = _store(tmp_path)
    ids = store.sync_transcript("s1", _msgs("row", 3))

    assert store.mark_compacted("s1", ids[:2]) == 2
    stats = store.stats("s1")
    assert stats["messages"] == 3, "compaction never deletes raw messages"
    assert stats["compacted_messages"] == 2
    # Flagged rows are still readable and still searchable.
    assert store.message("s1", ids[0])["compacted"] is True
    assert store.search("s1", "row 0")
