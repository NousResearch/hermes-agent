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


def test_insert_does_not_mutate_caller_model_config(db):
    """Strip must copy; callers may reuse the same dict for a child spawn."""
    sid = "gw_insert_no_mutate"
    caller_cfg = {
        "_delegate_from": PARENT,
        "_reset_from": PARENT,
    }
    db.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK + ":nomutate",
        chat_id="410541755",
        chat_type="dm",
        model_config=caller_cfg,
    )
    assert caller_cfg.get("_delegate_from") == PARENT
    assert "_delegate_from" not in _mc(db.get_session(sid))


def test_insert_empty_session_key_keeps_delegate_from(db):
    """Empty string is not a gateway key (same as heal: session_key != '')."""
    sid = "empty_key_keep"
    db.create_session(
        sid,
        "delegate",
        parent_session_id=PARENT,
        session_key="",
        model_config={"_delegate_from": PARENT},
    )
    assert _mc(db.get_session(sid)).get("_delegate_from") == PARENT


def test_merge_sole_delegate_from_becomes_null(db):
    """Stripping the only key must store NULL, not '{}'."""
    sid = "gw_merge_empty"
    db.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK + ":merge_empty",
        chat_id="410541755",
        chat_type="dm",
    )
    db.patch_session_model_config(sid, {"_delegate_from": PARENT})
    raw = db.get_session(sid).get("model_config")
    assert raw is None or raw == {} or raw == "{}"
    assert "_delegate_from" not in _mc(db.get_session(sid))


def test_merge_missing_row_is_noop(db):
    """patch on a missing id must not raise (on_missing=skip)."""
    db.patch_session_model_config("no_such_session", {"_delegate_from": PARENT})
    assert db.get_session("no_such_session") is None


def test_startup_heal_sole_marker_becomes_null(tmp_path):
    """Heal must store NULL when json_remove leaves only '{}'."""
    db_path = tmp_path / "heal_null.db"
    store = SessionDB(db_path=db_path)
    store.create_session(PARENT, "cli")
    sid = "gw_heal_null"
    store.create_session(
        sid,
        "telegram",
        user_id="410541755",
        session_key=SK + ":heal_null",
        chat_id="410541755",
        chat_type="dm",
    )
    with store._lock:
        store._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_delegate_from": PARENT}), sid),
        )
        store._conn.commit()
    store.close()

    healed = SessionDB(db_path=db_path)
    try:
        row = healed.get_session(sid)
        raw = row.get("model_config")
        assert raw is None or raw == {} or raw == "{}"
        assert "_delegate_from" not in _mc(row)
    finally:
        healed.close()


def test_startup_heal_leaves_child_without_session_key(tmp_path):
    """Heal must not strip _delegate_from from delegate children."""
    db_path = tmp_path / "heal_child.db"
    store = SessionDB(db_path=db_path)
    store.create_session(PARENT, "cli")
    sid = "delegate_child_heal_keep"
    store.create_session(
        sid,
        "delegate",
        parent_session_id=PARENT,
        model_config={"_delegate_from": PARENT},
    )
    # Also plant a polluted gateway row so the heal UPDATE actually runs
    # (probe returns a hit) rather than short-circuiting before children
    # could be touched by a buggy WHERE clause.
    gw = "gw_heal_sibling"
    store.create_session(
        gw,
        "telegram",
        user_id="410541755",
        session_key=SK + ":heal_sibling",
        chat_id="410541755",
        chat_type="dm",
    )
    with store._lock:
        store._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({"_delegate_from": PARENT, "_reset_from": PARENT}), gw),
        )
        store._conn.commit()
    store.close()

    healed = SessionDB(db_path=db_path)
    try:
        assert _mc(healed.get_session(sid)).get("_delegate_from") == PARENT
        assert "_delegate_from" not in _mc(healed.get_session(gw))
    finally:
        healed.close()
