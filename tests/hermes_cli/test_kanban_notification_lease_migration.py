"""Delivery claims survive upgrades and reject acknowledgements from stale owners."""
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as notify
import pytest


def test_legacy_subscription_migration_preserves_pending_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="retained receipt")
        notify.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="one")
        for column in ("notify_claim_owner", "notify_claimed_at", "notify_claimed_cursor"):
            conn.execute(f"ALTER TABLE kanban_notify_subs DROP COLUMN {column}")
        kbc._migrate_add_optional_columns(conn)
        kb.complete_task(conn, tid, result="finished")
        _, cursor, events = notify.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="one", claim_owner="current")
        assert [event.kind for event in events] == ["completed"]
        with pytest.raises(RuntimeError, match="no longer held"):
            notify.advance_notify_cursor(conn, task_id=tid, platform="telegram", chat_id="one",
                                         new_cursor=cursor, claim_owner="stale")
        notify.advance_notify_cursor(conn, task_id=tid, platform="telegram", chat_id="one",
                                     new_cursor=cursor, claim_owner="current")
        assert notify.unseen_events_for_sub(conn, task_id=tid, platform="telegram", chat_id="one")[1] == []
