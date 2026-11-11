"""Session export timing evidence, over the SQLite store's export verbs and the export dict the CLI writes."""

from hermes_cli.session_export import to_legacy_export_dict
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session


def test_export_session_includes_text_free_timing_evidence(backend, scope_a):
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, session_id="s1", source="cli", model="test-model")
    append_message(holder, "s1", "user", "secret prompt", timestamp=1000.0)
    append_message(
        holder,
        "s1",
        "assistant",
        "",
        tool_calls=[{"id": "call-1", "function": {"name": "terminal"}}],
        timestamp=1001.25,
    )
    append_message(
        holder,
        "s1",
        "tool",
        "secret tool output",
        tool_name="terminal",
        tool_call_id="call-1",
        timestamp=1003.0,
    )
    append_message(holder, "s1", "assistant", "done", timestamp=1003.5)

    exported = to_legacy_export_dict(store.maintenance.export_session("s1"))

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


def test_lineage_export_timings_span_the_merged_messages_and_import_ignores_their_size(backend, scope_a, scope_b):
    """A compression lineage exports as one logical session: its timings must cover every
    segment's messages, not just the last segment's. On import the derived block is stripped
    from the per-session size measurement so a long lineage's intervals cannot trip the limit."""
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, session_id="root", source="cli", model="m")
    append_message(holder, "root", "user", "first", timestamp=100.0)
    append_message(holder, "root", "assistant", "ok", timestamp=101.0)
    end_session(holder, "root", end_reason="compression")
    create_session(holder, session_id="child", source="cli", model="m", parent_session_id="root")
    append_message(holder, "child", "user", "second", timestamp=200.0)
    append_message(holder, "child", "assistant", "done", timestamp=200.5)

    exported = to_legacy_export_dict(store.maintenance.export_session_lineage("child"), session_id="child")
    assert exported["lineage_session_ids"] == ["root", "child"]
    assert exported["timings"]["wall_clock_ms"] == 100_500
    assert exported["timings"]["message_timestamps"] == {"available": 4, "missing": 0}
    assert exported["segments"][-1]["timings"]["wall_clock_ms"] == 500

    exported["timings"]["intervals"] = [{"pad": "x" * 100} for _ in range(60_000)]  # ~6 MiB
    report = backend.open(scope_b).maintenance.import_sessions([exported])
    assert list(report.errors) == [] and report.imported == 1


def test_corrupt_timestamp_rows_count_as_missing_instead_of_aborting_export(backend, scope_a):
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, session_id="s1", source="cli", model="test-model")
    append_message(holder, "s1", "user", "hello", timestamp=10.0)
    append_message(holder, "s1", "assistant", "hi", timestamp=11.0)
    # Writers refuse bad stamps; emulate a pre-existing corrupt row directly.
    holder._write_sql("UPDATE messages SET timestamp = 8.4e252 WHERE id = 2")
    timings = to_legacy_export_dict(store.maintenance.export_session("s1"))["timings"]

    assert timings["message_timestamps"] == {"available": 1, "missing": 1}
    assert timings["wall_clock_ms"] == 0 and timings["intervals"] == []
