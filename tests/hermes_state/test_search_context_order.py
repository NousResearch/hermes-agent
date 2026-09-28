"""Search context follows transcript order even when wall clocks regress (#126762)."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("timestamps", [(10, 20, 30, 40), (40, 10, 30, 20), (10, 10, 10, 10)])
@pytest.mark.parametrize("route", ["fts", "like"])
def test_search_context_matches_session_transcript(tmp_path, timestamps, route):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        for sid in ("one", "two", "single"):
            db.create_session(sid, "cli")
        for i, timestamp in enumerate(timestamps):
            for sid in ("one", "two"):
                db.append_message(sid, "user" if i % 2 == 0 else "assistant",
                                  f"needleprobe {sid} {i}", timestamp=timestamp)
        db.append_message("single", "user", "needleprobe single", timestamp=15)

    with SessionDB(path) as db:
        filters = {"role_filter": ["user", "assistant", "tool"]} if route == "like" else {}
        transcripts = {sid: db.get_messages(sid) for sid in ("one", "two", "single")}
        for sort in (None, "newest", "oldest"):
            matches = db.search_messages("needleprobe", limit=20, sort=sort, **filters)
            assert len(matches) == sum(len(rows) for rows in transcripts.values())
            for match in matches:
                transcript = transcripts[match["session_id"]]
                index = next(i for i, row in enumerate(transcript) if row["id"] == match["id"])
                expected = transcript[max(0, index - 1):index + 2]
                assert match["context"] == [
                    {"role": row["role"], "content": row["content"]} for row in expected
                ]
        assert db.search_messages("no_such_needle", **filters) == []
        assert all(set(row) == {"id"} for row in db.search_messages(
            "needleprobe", fields=["id"], **filters))
