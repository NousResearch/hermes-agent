"""A hidden conversation stays hidden through compression.

The sidebar admits a compression lineage by its ROOT's ``hidden`` flag, while the canonical Bot Chat
rename guard and the desktop keep-list read the TIP's. A continuation published with ``hidden=0`` under a
hidden root made one conversation both hidden and visible: the canonical "Bot Chat" tip lost its rename
guard, and a lineage hidden at the root vanished from every listing while its live tip read as visible.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return SessionDB(tmp_path / "state.db")


def _compress(db, parent, child):
    holder = f"holder-{child}"
    assert db.try_acquire_compression_lock(parent, holder, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="tui", system_prompt="p",
        messages=[{"role": "user", "content": f"summary for {child}"}], compression_lock_holder=holder)


def _hidden(db, *ids):
    return {sid: db.get_session(sid)["hidden"] for sid in ids}


def test_canonical_bot_chat_keeps_its_identity_through_compression(db):
    db.create_session("root", "tui")
    db.append_message("root", "user", "hello")
    assert db.set_session_title("root", "Bot Chat")
    assert db.set_session_hidden("root", True)

    for parent, child in (("root", "mid"), ("mid", "tip")):
        _compress(db, parent, child)
        # The rotating agent carries the title onto the child after publishing it.
        assert db.set_session_title(child, "Bot Chat")

    assert _hidden(db, "root", "mid", "tip") == {"root": 1, "mid": 1, "tip": 1}
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        db.set_session_title("tip", "Renamed")
    listed = db.list_sessions_rich(limit=50, order_by_last_active=True, include_hidden=True)
    assert [row["id"] for row in listed if row.get("_lineage_root_id") == "root"] == ["tip"]


def test_a_mixed_lineage_heals_at_its_next_compression(db):
    """Stores written before the fix hold visible segments under a hidden root."""
    db.create_session("root", "tui")
    db.append_message("root", "user", "hello")
    assert db.set_session_hidden("root", True)
    _compress(db, "root", "mid")
    db._conn.execute("UPDATE sessions SET hidden = 0 WHERE id = 'mid'")
    db._conn.commit()

    _compress(db, "mid", "tip")

    assert db.get_session("tip")["hidden"] == 1


def test_a_visible_chat_stays_visible(db):
    db.create_session("root", "tui")
    db.append_message("root", "user", "hello")
    _compress(db, "root", "tip")

    assert _hidden(db, "root", "tip") == {"root": 0, "tip": 0}
    listed = db.list_sessions_rich(limit=50, order_by_last_active=True)
    assert [row["id"] for row in listed] == ["tip"]
