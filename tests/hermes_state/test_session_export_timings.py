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


def test_lineage_export_timings_span_the_merged_messages_and_import_ignores_their_size(tmp_path):
    """A compression lineage exports as one logical session: its timings must cover every
    segment's messages, not just the last segment's. On import the derived block is stripped
    from the per-session size measurement so a long lineage's intervals cannot trip the limit."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="root", source="cli", model="m")
        db.append_message("root", "user", "first", timestamp=100.0)
        db.append_message("root", "assistant", "ok", timestamp=101.0)
        db.end_session("root", end_reason="compression")
        db.create_session(session_id="child", source="cli", model="m", parent_session_id="root")
        db.append_message("child", "user", "second", timestamp=200.0)
        db.append_message("child", "assistant", "done", timestamp=200.5)

        exported = db.export_session_lineage("child")
        assert exported["lineage_session_ids"] == ["root", "child"]
        assert exported["timings"]["wall_clock_ms"] == 100_500
        assert exported["timings"]["message_timestamps"] == {"available": 4, "missing": 0}
        assert exported["segments"][-1]["timings"]["wall_clock_ms"] == 500

        exported["timings"]["intervals"] = [{"pad": "x" * 100} for _ in range(60_000)]  # ~6 MiB
        target = SessionDB(db_path=tmp_path / "target.db")
        try:
            report = target.import_sessions([exported])
        finally:
            target.close()
    finally:
        db.close()
    assert report["errors"] == [] and report["imported"] == 1


def test_corrupt_timestamp_rows_count_as_missing_instead_of_aborting_export(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test-model")
        db.append_message("s1", "user", "hello", timestamp=10.0)
        db.append_message("s1", "assistant", "hi", timestamp=11.0)
        # Writers refuse bad stamps; emulate a pre-existing corrupt row directly.
        db._conn.execute("UPDATE messages SET timestamp = 8.4e252 WHERE id = 2")
        db._conn.commit()
        timings = db.export_session("s1")["timings"]
    finally:
        db.close()

    assert timings["message_timestamps"] == {"available": 1, "missing": 1}
    assert timings["wall_clock_ms"] == 0 and timings["intervals"] == []
