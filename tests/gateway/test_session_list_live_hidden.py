"""The live session.list picker honours hidden/archived like every stored picker (N110)."""
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority, _create
from persistence_assembly import close_stores_under


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
