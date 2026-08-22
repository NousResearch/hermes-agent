from hermes_state import SessionDB


def test_export_lineage_can_include_compacted_but_not_rewound_rows(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "archive-compacted"
        db.create_session(session_id=session_id, source="webui")
        db.append_message(session_id, role="user", content="original user")
        db.append_message(session_id, role="assistant", content="original answer")

        db.archive_and_compact(
            session_id,
            [
                {"role": "user", "content": "compaction context"},
                {"role": "assistant", "content": "compaction summary"},
            ],
        )
        db.append_message(session_id, role="user", content="after compaction")
        db.append_message(session_id, role="assistant", content="rewound answer")

        def _rewind_last(conn):
            conn.execute(
                "UPDATE messages SET active = 0, compacted = 0 "
                "WHERE session_id = ? AND content = ?",
                (session_id, "rewound answer"),
            )

        db._execute_write(_rewind_last)

        live = db.export_session_lineage(session_id)
        durable = db.export_session_lineage(
            session_id,
            include_compacted=True,
        )

        assert [message["content"] for message in live["messages"]] == [
            "compaction context",
            "compaction summary",
            "after compaction",
        ]
        assert [message["content"] for message in durable["messages"]] == [
            "original user",
            "original answer",
            "compaction context",
            "compaction summary",
            "after compaction",
        ]
        assert durable["message_count"] == 5
        assert durable["lineage_session_ids"] == [session_id]
    finally:
        db.close()
