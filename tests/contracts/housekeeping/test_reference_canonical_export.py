"""Imports and transcript rewrites share export rows and durable counters."""

import pytest

from persistence import ExportTooLarge
from persistence.conversation.messages import NewMessage, ReplaceMode
from persistence.housekeeping.maintenance import PruneCriteria

STAMP = 1699920000.0


def _payload(*words: str) -> dict:
    return {"id": "s", "source": "cli", "started_at": STAMP, "messages": [{"role": "user", "content": word, "timestamp": STAMP} for word in words]}


@pytest.mark.parametrize("mode, stored_count", [(ReplaceMode.DESTRUCTIVE, 1), (ReplaceMode.ARCHIVE_DROPPED, 3)])
def test_import_replacement_is_visible_to_export_and_pruning(backend, scope_a, mode, stored_count):
    store_a = backend.open(scope_a)
    assert store_a.maintenance.import_sessions([_payload("old-a", "old-b")]).imported == 1
    assert [row.content for row in store_a.messages.read_transcript("s")] == ["old-a", "old-b"]
    store_a.messages.replace_transcript("s", (NewMessage("user", "replacement", timestamp=STAMP),), mode=mode)
    exported = store_a.maintenance.export_session("s")
    assert exported.session.message_count == 1
    assert [row.content for row in exported.messages] == ["replacement"]
    # The export guard counts every stored row: an archiving replace keeps the dropped rows.
    assert store_a.maintenance.assert_export_safe("s", 10) == stored_count
    store_a.sessions.end("s", "done")
    candidates = store_a.maintenance.list_prune_candidates(criteria=PruneCriteria(min_messages=1, max_messages=1))
    assert [candidate.message_count for candidate in candidates] == [1]


def test_export_guard_counts_inactive_stored_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_payload("a", "b")])
    rows = store_a.messages.read_transcript("s")
    assert store_a.messages.deactivate_message("s", rows[0].id) is True
    assert len(store_a.maintenance.export_session("s").messages) == 1
    with pytest.raises(ExportTooLarge):
        store_a.maintenance.assert_export_safe("s", 1)
    assert store_a.maintenance.assert_export_safe("s", 10) == 2
