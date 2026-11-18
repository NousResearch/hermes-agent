"""A `/new` must survive its own create window.

``reset_session()`` publishes the fresh routing entry (sessions.json) and only
then INSERTs the session row. A concurrent ``get_or_create_session()`` landing
between those two steps sees ``row is None``; ``_is_session_ended_in_db`` treats
a missing row as gone on purpose (the #42422 rule: reusing a deleted id lets
run_agent's INSERT OR IGNORE resurrect it). The route is then "self-healed" —
back to the *predecessor* session, silently cancelling the /new.

Measured in the wild 2026-10-28 01:35:41: `new_command` → `session_reset` →
248 ms later `routing key ... -> 20261028_013541_7f36de8c is ended in state.db`
while that row's `started_at` was 57 ms in the future; the next user message
went to the old 597K-token session and the compression warning came straight
back.

A fresh-reset route with no row is a *pending create*, not a deleted session:
keep it, and let the peer refresh insert the row as designed.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8494508720",
        chat_type="dm",
        user_id="8494508720",
    )


def _db(rows: dict) -> MagicMock:
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _store(tmp_path, db) -> SessionStore:
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = db
    store._loaded = True
    return store


def _entry(key: str, session_id: str, **kw) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key, session_id=session_id,
        created_at=now - timedelta(hours=2), updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM, chat_type="dm", **kw,
    )


def test_reader_inside_create_window_keeps_the_fresh_session(tmp_path):
    """The exact 01:35:41 shape: reader arrives before the INSERT lands."""
    source = _source()
    rows = {"sid_old": {"id": "sid_old", "end_reason": None}}
    db = _db(rows)
    store = _store(tmp_path, db)
    key = store._generate_session_key(source)
    store._entries[key] = _entry(key, "sid_old")

    seen = {}

    def _create_session(**kw):
        # The window: sessions.json already points at the new id, no row yet.
        if "during" not in seen:
            seen["during"] = store.get_or_create_session(source).session_id
        rows[kw["session_id"]] = {"id": kw["session_id"], "end_reason": None}

    db.find_latest_gateway_session_for_peer.return_value = {
        "id": "sid_old", "started_at": (datetime.now() - timedelta(hours=2)).timestamp(),
    }
    db.create_session.side_effect = _create_session

    new_entry = store.reset_session(key)

    assert new_entry is not None
    assert seen["during"] == new_entry.session_id, "reader inside the create window was re-routed"
    assert store._entries[key].session_id == new_entry.session_id


def test_missing_row_for_an_old_route_still_heals(tmp_path):
    """The #42422 guard stays: a deleted row on a non-fresh route is still stale."""
    source = _source()
    db = _db({"sid_gone": {"id": "sid_gone", "end_reason": None}})
    store = _store(tmp_path, db)
    key = store._generate_session_key(source)
    store._entries[key] = _entry(key, "sid_gone")
    # The row is deleted; the routing entry survived (hard delete, #42422).
    db.get_session.side_effect = lambda sid: None
    db.find_latest_gateway_session_for_peer.return_value = None

    result = store.get_or_create_session(source)

    assert result.session_id != "sid_gone"
    assert store._entries[key].session_id == result.session_id
