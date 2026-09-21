"""Gateway main sessions must never carry ``_delegate_from`` (#109073).

A row with ``session_key`` set is a gateway main lane (e.g. Telegram DM).
That marker excludes the row from every picker (``list_sessions_rich``,
``list_recent_sessions_bounded``, /resume) while the gateway keeps routing
into it by ``session_key`` — chat works, the conversation vanishes from
desktop. Historic insert/merge paths could copy a delegate child's
``model_config`` onto the main row (often with the contradictory
``_reset_from`` pair). These tests lock the write-time strip + startup heal
salvaged from PR #109081.
"""

from __future__ import annotations

import json

import pytest

from hermes_state import SessionDB


PARENT = "parent_for_delegate_strip"
SK = "agent:main:telegram:dm:410541755"


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(PARENT, "cli")
    yield store
    store.close()


def _mc(session: dict) -> dict:
    raw = session.get("model_config")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _listed_ids(db: SessionDB) -> set[str]:
    return {
        s["id"]
        for s in db.list_sessions_rich(
            min_message_count=1, include_archived=True, limit=100
        )
    }


def test_insert_strips_delegate_from_when_session_key_set(db):
    """create_session with session_key must not persist _delegate_from."""
    sid = "gw_insert_strip"
    db.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK,
        chat_id="410541755",
        chat_type="dm",
        parent_session_id=PARENT,
        model_config={
            "_delegate_from": PARENT,
            "_reset_from": PARENT,
            "_usage_anchor": {"model": "x"},
        },
    )
    db.append_message(sid, "user", "still here")
    db.append_message(sid, "assistant", "ok")

    cfg = _mc(db.get_session(sid))
    assert "_delegate_from" not in cfg
    assert cfg.get("_reset_from") == PARENT
    assert sid in _listed_ids(db)


def test_merge_strips_delegate_from_when_session_key_set(db):
    """patch_session_model_config must not plant _delegate_from on a gateway row."""
    sid = "gw_merge_strip"
    db.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK + ":merge",
        chat_id="410541755",
        chat_type="dm",
        model_config={"_reset_from": PARENT},
    )
    db.append_message(sid, "user", "clean")
    db.append_message(sid, "assistant", "ok")

    db.patch_session_model_config(
        sid, {"_delegate_from": PARENT, "_usage_anchor": {"model": "y"}}
    )

    cfg = _mc(db.get_session(sid))
    assert "_delegate_from" not in cfg
    assert cfg.get("_usage_anchor", {}).get("model") == "y"
    assert sid in _listed_ids(db)


def test_insert_keeps_delegate_from_on_child_without_session_key(db):
    """Delegate children (no session_key) must keep the marker unchanged."""
    sid = "delegate_child_keep"
    db.create_session(
        sid,
        "delegate",
        parent_session_id=PARENT,
        model_config={"_delegate_from": PARENT},
    )
    cfg = _mc(db.get_session(sid))
    assert cfg.get("_delegate_from") == PARENT


def test_startup_heal_strips_polluted_gateway_row(tmp_path):
    """Historic pollution is repaired on the next SessionDB open."""
    db_path = tmp_path / "heal_state.db"
    store = SessionDB(db_path=db_path)
    store.create_session(PARENT, "cli")
    sid = "gw_heal_strip"
    store.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK + ":heal",
        chat_id="410541755",
        chat_type="dm",
        model_config={"_reset_from": PARENT},
    )
    store.append_message(sid, "user", "pollute me")
    store.append_message(sid, "assistant", "ok")
    # Bypass write guards the way a historic bad upsert would: raw SQL.
    with store._lock:
        store._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "_delegate_from": PARENT,
                        "_reset_from": PARENT,
                        "_usage_anchor": {"model": "z"},
                    }
                ),
                sid,
            ),
        )
        store._conn.commit()
    assert "_delegate_from" in _mc(store.get_session(sid))
    assert sid not in _listed_ids(store)
    store.close()

    healed = SessionDB(db_path=db_path)
    try:
        cfg = _mc(healed.get_session(sid))
        assert "_delegate_from" not in cfg
        assert cfg.get("_reset_from") == PARENT
        assert sid in _listed_ids(healed)
    finally:
        healed.close()
