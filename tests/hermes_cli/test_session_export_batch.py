from hermes_state import SessionDB


def test_export_all_batches_message_reads_without_changing_export_rows(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    try:
        for index, source in enumerate(("cli", "telegram", "cli", "cli")):
            session_id = f"session-{index}"
            db.create_session(session_id=session_id, source=source)
            db.append_messages_batch(
                session_id,
                [
                    {"role": "user", "content": f"question {index}"},
                    {
                        "role": "assistant",
                        "content": f"answer {index}",
                        "tool_calls": [{"id": f"call-{index}", "type": "function"}],
                    },
                ],
            )

        sessions = db.search_sessions(source="cli", limit=100000)
        expected = [
            {**session, "messages": db.get_messages(session["id"])}
            for session in sessions
        ]

        original_read_all = db._read_all
        read_calls = 0

        def counted_read_all(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return original_read_all(*args, **kwargs)

        monkeypatch.setattr(db, "_read_all", counted_read_all)

        assert db.export_all(source="cli") == expected
        assert read_calls <= 2
    finally:
        db.close()
