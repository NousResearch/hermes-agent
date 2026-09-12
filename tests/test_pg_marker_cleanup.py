"""PostgreSQL marker cleanup requires an explicit native-backup decision."""

import pytest

from tests.test_pg_clone_rows import _raw_rows, message_store  # noqa: F401


def _seed_markers(db):
    affected = db.append_message(
        "parent", role="assistant", content="[memory]",
        tool_calls=[{"id": "call-1", "type": "function",
                     "function": {"name": "memory", "arguments": "{}"}}],
    )
    db.append_message("parent", role="assistant", content="[memory]")
    db.append_message("parent", role="assistant", content="ordinary assistant text")
    return affected


def test_postgres_marker_cleanup_requires_native_backup_before_mutation(message_store):
    db, raw = message_store
    affected = _seed_markers(db)
    before = _raw_rows(raw, "parent")
    dry_run = db.purge_stale_tool_call_markers(dry_run=True)
    assert dry_run["row_ids"] == [affected]
    assert dry_run["backup_path"] is None
    assert _raw_rows(raw, "parent") == before

    with pytest.raises(RuntimeError, match="PostgreSQL-native backup.*--no-backup"):
        db.purge_stale_tool_call_markers()
    assert _raw_rows(raw, "parent") == before
    assert not list(db.db_path.parent.glob("*.pre-clean-markers-backup-*"))


def test_postgres_marker_cleanup_opt_out_preserves_other_message_fields(message_store):
    db, raw = message_store
    affected = _seed_markers(db)
    before = _raw_rows(raw, "parent")
    result = db.purge_stale_tool_call_markers(backup=False)
    expected = [{**row, "content": ""} if row["id"] == affected else row for row in before]
    def canonical(rows):
        return [{key: value for key, value in row.items()
                 if key not in {"display_identity", "display_order", "fts_content"}} for row in rows]
    assert canonical(_raw_rows(raw, "parent")) == canonical(expected)
    assert result["row_ids"] == [affected]
    assert result["backup_path"] is None
    assert db.purge_stale_tool_call_markers()["rows_affected"] == 0
