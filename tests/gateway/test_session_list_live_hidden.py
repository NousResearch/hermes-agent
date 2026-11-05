"""The live session.list picker honours hidden/archived like every stored picker (N110)."""
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority, _create
from persistence_assembly import close_stores_under
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row


@pytest.mark.asyncio
async def test_live_list_hides_hidden_and_archived_unless_asked(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        chat = await _create(conn, authority, store, 'cli', 'chat')
        hidden = await _create(conn, authority, store, 'cli', 'hidden')
        archived = await _create(conn, authority, store, 'cli', 'archived')
        authority.persistence.sessions.set_hidden(hidden, True)
        authority.persistence.sessions.set_archived(archived, True)
        listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [chat]
        every = await conn.dispatch({'id': 10, 'method': 'session.list', 'params': {'include_hidden': True}})
        assert {row['id'] for row in every['result']['sessions']} == {chat, hidden, archived}
    finally:
        await conn.close()
        close_stores_under(authority.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_title_list_skips_deliberately_archived_bot_chat(tmp_path, monkeypatch):
    """Bots roster session.list title= must not reopen a retired hidden Bot Chat (#139779)."""
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        sid = await _create(conn, authority, store, 'cli', 'bot')
        db = authority.persistence
        assert db.sessions.set_title(sid, 'Bot Chat')
        assert db.sessions.set_hidden(sid, True)
        raw_holder(db)._execute_write(lambda c: c.execute(
            'UPDATE sessions SET archived = 1, auto_archived = 0 WHERE id = ?', (sid,)))
        listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {'title': 'Bot Chat'}})
        assert listed['result']['sessions'] == []
        assert session_row(db, sid)['archived']
        assert session_row(db, sid)['title'] == 'Bot Chat'
    finally:
        await conn.close()
        close_stores_under(authority.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_title_list_still_resolves_recoverable_auto_archived_bot_chat(tmp_path, monkeypatch):
    """Recoverable ws_orphan_reap archives stay findable by title (#92687)."""
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        sid = await _create(conn, authority, store, 'cli', 'bot')
        db = authority.persistence
        assert db.sessions.set_title(sid, 'Bot Chat')
        assert db.sessions.set_hidden(sid, True)
        db.sessions.end(sid, 'ws_orphan_reap')
        raw_holder(db)._execute_write(lambda c: c.execute(
            'UPDATE sessions SET archived = 1 WHERE id = ?', (sid,)))  # pre-#139779 reaper state
        listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {'title': 'Bot Chat'}})
        sessions = listed['result']['sessions']
        assert len(sessions) == 1
        assert sessions[0]['session_id'] == sid or sessions[0]['id'] == sid
        assert session_row(db, sid)['title'] == 'Bot Chat'
    finally:
        await conn.close()
        close_stores_under(authority.persistence.scope.profile_dir)
