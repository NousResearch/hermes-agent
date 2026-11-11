"""Desktop's Pin is ``PATCH /api/sessions/{id}`` (operation ``sidebar``). A pin means "keep this
visible": a hidden session the user pins must come back into the sidebar listing, while the canonical
Bot Chat stays hidden behind the bot row (#106171)."""
import httpx
import pytest
from fastapi import FastAPI

from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB

AUTH = {'Authorization': 'Bearer standalone-token'}


@pytest.mark.asyncio
async def test_desktop_pin_of_hidden_session_makes_it_visible(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, '_SESSION_TOKEN', 'standalone-token')
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        for sid in ('tucked', 'canon'):
            db.create_session(sid, source='desktop')
            db.append_message(sid, 'user', 'hello')
            assert db.set_session_hidden(sid, True)
        assert db.set_session_title('canon', SessionDB.CANONICAL_BOT_CHAT_TITLE)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost',
                                     headers=AUTH) as client:
            for sid in ('tucked', 'canon'):
                revision = db.get_session(sid)['runtime_revision'] or 0
                pinned = await client.patch(f'/api/sessions/{sid}', json={
                    'request_id': f'pin-{sid}', 'expected_revision': revision, 'pinned': True})
                assert pinned.status_code == 200, pinned.text
        assert {sid: (row['pinned'], row['hidden'])
                for sid in ('tucked', 'canon') for row in [db.get_session(sid)]} == {
            'tucked': (1, 0), 'canon': (1, 1)}
        assert {row['id'] for row in db.list_sessions_rich(min_message_count=1)} == {'tucked'}
