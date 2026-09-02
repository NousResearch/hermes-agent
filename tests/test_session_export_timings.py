"""Session export timing evidence."""

from hermes_state import SessionDB


def test_export_session_includes_text_free_timing_evidence(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test-model")
        db.append_message("s1", "user", "secret prompt", timestamp=1000.0)
        db.append_message(
            "s1",
            "assistant",
            "",
            tool_calls=[{"id": "call-1", "function": {"name": "terminal"}}],
            timestamp=1001.25,
        )
        db.append_message(
            "s1",
            "tool",
            "secret tool output",
            tool_name="terminal",
            tool_call_id="call-1",
            timestamp=1003.0,
        )
        db.append_message("s1", "assistant", "done", timestamp=1003.5)

        exported = db.export_session("s1")
    finally:
        db.close()

    timings = exported["timings"]
    assert timings["source"] == "message_timestamps"
    assert timings["available"] is True
    assert timings["complete"] is False
    assert timings["wall_clock_ms"] == 3500
    assert timings["largest_gap_ms"] == 1750
    assert timings["message_timestamps"] == {"available": 4, "missing": 0}
    assert timings["role_counts"] == {"user": 1, "assistant": 2, "tool": 1}
    assert timings["tool_calls_emitted"] == 1
    assert timings["tool_result_count"] == 1
    assert timings["intervals"] == [
        {
            "from_message_id": 1,
            "to_message_id": 2,
            "from_role": "user",
            "to_role": "assistant",
            "gap_ms": 1250,
        },
        {
            "from_message_id": 2,
            "to_message_id": 3,
            "from_role": "assistant",
            "to_role": "tool",
            "gap_ms": 1750,
        },
        {
            "from_message_id": 3,
            "to_message_id": 4,
            "from_role": "tool",
            "to_role": "assistant",
            "gap_ms": 500,
        },
    ]
    assert "secret prompt" not in str(timings)
    assert "secret tool output" not in str(timings)


def test_export_all_includes_timing_evidence(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test-model")
        db.append_message("s1", "user", "hello", timestamp=10.0)
        db.append_message("s1", "assistant", "hi", timestamp=11.0)

        exported = db.export_all()
    finally:
        db.close()

    assert exported[0]["timings"]["wall_clock_ms"] == 1000
    assert exported[0]["timings"]["intervals"][0]["gap_ms"] == 1000
