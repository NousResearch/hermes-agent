"""Message verbs on SQLite whose statements keep the legacy readers' shape: the spillover reference read's fixed row
cap, the latest dialogue message's one content cell, and the row stamps an archive-dropped replace leaves on the
rows it keeps."""

from persistence.conversation.messages import NewMessage, ReplaceMode
from persistence.conversation.sessions import NewSession


def test_spillover_reference_rows_read_at_most_one_past_the_trusted_set(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("spill", "cli"))
    for i in range(3):
        store.messages.append("spill", NewMessage("assistant", f"see spillover/file-{i}.txt"))
    store.messages.append("spill", NewMessage("user", "nothing to see"))
    rows = store.messages.spillover_reference_rows()
    assert sorted(content for content, _ in rows) == [f"see spillover/file-{i}.txt" for i in range(3)]
    with store.holder()._lock:
        store.holder()._conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('spill', 'tool', ?, 0)",
            [(f"spillover/bulk-{i}",) for i in range(20001)])
    assert len(store.messages.spillover_reference_rows()) == 20001


def test_latest_dialogue_message_is_the_stored_content_cell(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("dialog", "cli"))
    store.messages.append("dialog", NewMessage("user", "  first  question "))
    store.messages.append("dialog", NewMessage("tool", "tool output"))
    assert store.messages.latest_dialogue_message("dialog") == "  first  question "
    assert store.messages.latest_dialogue_message("missing") is None


def test_archive_dropped_replace_leaves_kept_rows_without_a_snapshot(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("swap", "cli"))
    store.messages.append("swap", NewMessage("user", "first", timestamp=1.0))
    store.messages.append("swap", NewMessage("assistant", "old answer", timestamp=2.0))
    result = store.messages.replace_transcript(
        "swap", [NewMessage("user", "first", timestamp=1.0), NewMessage("assistant", "new answer", timestamp=3.0)],
        ReplaceMode.ARCHIVE_DROPPED)
    kept, written = result.rows
    assert kept.db_row_snapshot is None
    assert isinstance(written.db_row_snapshot, str) and written.db_row_snapshot
