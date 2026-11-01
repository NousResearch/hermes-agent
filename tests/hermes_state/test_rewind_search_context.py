"""Search context excludes rewound turns while retaining compacted history."""

from hermes_state import SessionDB


def test_search_context_skips_rewound_neighbors(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    ids = [
        db.append_message("s1", "user" if index % 2 == 0 else "assistant", f"history {index}")
        for index in range(12)
    ]

    db.rewind_user_turn("s1", -1)
    db.archive_and_compact("s1", [{"role": "assistant", "content": "compaction summary"}])

    matches = db.search_messages("history 9", limit=5, fields={"id", "context"})
    hit = next(match for match in matches if match["id"] == ids[9])
    assert [message["content"] for message in hit["context"]] == [
        "history 8", "history 9", "compaction summary"
    ]
    db.close()
