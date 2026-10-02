"""Regression for #130344: session transfers retain keep and Bot Chat protections."""

import json
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def stores(tmp_path):
    donor = SessionDB(db_path=tmp_path / "donor.db")
    restored = SessionDB(db_path=tmp_path / "restored.db")
    try:
        yield donor, restored
    finally:
        restored.close()
        donor.close()


def test_backup_restore_keeps_pinned_history_through_startup_retention(stores):
    donor, restored = stores
    old = time.time() - 200 * 86400
    for sid in ("keep", "archived-keep", "expire"):
        donor.create_session(sid, source="cli")
        donor.append_message(sid, "user", f"history of {sid}", timestamp=old)
        donor.end_session(sid, "completed")
        donor._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (old, old, sid))
    donor._conn.commit()
    donor.set_session_pinned("keep", True)
    donor.set_session_pinned("archived-keep", True)
    donor.set_session_archived("archived-keep", True)
    payload = [donor.export_session(sid, include_inactive=True)
               for sid in ("keep", "archived-keep", "expire")]
    # Exports predating durable flags must still import as ordinary, visible sessions.
    payload.append({"id": "legacy", "source": "cli",
                    "messages": [{"role": "user", "content": "older export"}]})
    result = restored.import_sessions(json.loads(json.dumps(payload)))
    assert result["ok"] and result["imported"] == len(payload), result

    maintenance = restored.maybe_auto_prune_and_vacuum(
        retention_days=90, min_interval_hours=0, vacuum=False)

    for sid in ("keep", "archived-keep"):
        row = restored.get_session(sid)
        assert row is not None, "Startup retention deleted a restored pinned session"
        assert row["pinned"] == donor.get_session(sid)["pinned"]
        assert row["archived"] == donor.get_session(sid)["archived"]
        assert restored.get_messages(sid)[0]["content"] == f"history of {sid}"
    assert maintenance["pruned"] == 1 and "error" not in maintenance, maintenance
    assert restored.get_session("expire") is None
    assert donor.get_session("expire") is not None, "Recipient cleanup must not mutate the donor"
    legacy = restored.get_session("legacy")
    assert all(legacy[flag] == 0 for flag in ("archived", "pinned", "hidden"))


def test_adopted_hidden_bot_lineage_keeps_listing_rename_and_archive_guards(stores):
    donor, restored = stores
    old = time.time() - 30 * 86400
    parent, tip = "compressed-bot", "current-bot"
    donor.create_session(parent, source="tui")
    donor.append_message(parent, "user", "before compression", timestamp=old)
    donor.set_session_pinned(parent, True)
    donor.set_session_hidden(parent, True)
    donor.end_session(parent, "compression")
    donor.create_session(tip, source="tui", parent_session_id=parent)
    donor.append_message(tip, "user", "after compression", timestamp=old)
    donor.set_session_title(tip, donor.CANONICAL_BOT_CHAT_TITLE)
    donor.set_session_hidden(tip, True)
    donor._conn.execute("UPDATE sessions SET started_at = ? WHERE id IN (?, ?)", (old, parent, tip))
    donor._conn.commit()
    restored.create_session("local", source="tui")
    restored.append_message("local", "user", "recipient's unrelated history")

    result = restored.adopt_session_lineage_from(donor, parent)

    assert result["adopted"] and result["donor_retired"] and result["imported"] == 2, result
    visible = {row["id"] for row in restored.list_sessions_rich(min_message_count=1)}
    assert visible == {"local"}, "Adoption exposed the hidden Bot Chat in ordinary session listings"
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        restored.set_session_title(tip, "renamed")
    assert restored.archive_stale_sessions(idle_days=3) == 0
    assert restored.get_session(tip)["archived"] == 0
    assert restored.get_session(parent)["pinned"] == donor.get_session(parent)["pinned"]
    for sid in (parent, tip):
        assert restored.get_session(sid)["hidden"] == donor.get_session(sid)["hidden"]
        assert restored.get_messages(sid)[0]["content"] == donor.get_messages(sid)[0]["content"]
        assert donor.get_session(sid)["archived"] == 1

    # Retrying adoption must not restore the donor's retirement archive over the local copy.
    for sid in (parent, tip):
        again = restored.adopt_session_lineage_from(donor, sid)
        assert again["adopted"] and again["imported"] == 0 and sid in again["skipped_ids"], again
    assert restored.get_session(tip)["archived"] == 0
    assert {row["id"] for row in restored.list_sessions_rich(min_message_count=1)} == {"local"}
