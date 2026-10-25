"""session.info carries archived as a JSON boolean (Desktop ``archived === true``)."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority, _create
from tui_gateway.contracts.registry import CANONICAL_METHODS


@pytest.mark.asyncio
async def test_session_info_archived_hidden_bot_chat_is_json_true(tmp_path, monkeypatch):
    """Live-evidence shape: archived=1, title still Bot Chat, auto_archived=0, end_reason NULL.

    Desktop's isLiveHiddenBotChatProbe uses archived === true. SQLite 1 on the wire
    would still look like a live hidden Bot Chat (#139779).
    """
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        sid = await _create(conn, authority, store, 'cli', 'bot')
        db = authority.db
        assert db.set_session_title(sid, 'Bot Chat')
        assert db.set_session_hidden(sid, True)
        db._write_sql(
            'UPDATE sessions SET archived = 1, auto_archived = 0, end_reason = NULL WHERE id = ?', (sid,))
        row = db.get_session(sid)
        assert row['title'] == 'Bot Chat'
        assert row['archived']
        assert not row['auto_archived']
        assert row['end_reason'] is None

        reply = await conn.dispatch({'id': 11, 'method': 'session.info', 'params': {'session_id': sid}})
        assert 'result' in reply, reply
        info = reply['result']
        CANONICAL_METHODS['session.info'].result.model_validate(info)
        assert info['archived'] is True
        assert info['title'] == 'Bot Chat'
        assert json.dumps({'archived': info['archived']}) == '{"archived": true}'
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_session_info_live_hidden_bot_chat_archived_is_false(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        sid = await _create(conn, authority, store, 'cli', 'bot')
        db = authority.db
        assert db.set_session_title(sid, 'Bot Chat')
        assert db.set_session_hidden(sid, True)
        reply = await conn.dispatch({'id': 12, 'method': 'session.info', 'params': {'session_id': sid}})
        assert 'result' in reply, reply
        info = reply['result']
        assert info['archived'] is False
        assert info['title'] == 'Bot Chat'
        assert json.dumps({'archived': info['archived']}) == '{"archived": false}'
    finally:
        await conn.close()
        authority.db.close()
