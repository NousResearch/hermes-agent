"""``POST /api/sessions/{id}/chat`` on a session another lane owns is a refused admission, not a crash.

The API door deliberately refuses a turn on a chat-platform-owned session (``admit_api_steps``:
``permission_denied``); the session route must answer it the way ``/v1/runs`` does (409 via
``store_error_response``) rather than escaping as aiohttp's bare 500.
"""
import pytest
import pytest_asyncio
from types import SimpleNamespace
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest_asyncio.fixture
async def door(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    from hermes_state import SessionDB
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    authority.db.create_session('tg', 'telegram')
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = authority.db
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    try:
        async with TestClient(TestServer(app)) as cli:
            yield cli
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store._db.close()


@pytest.mark.asyncio
async def test_chat_on_a_telegram_owned_session_is_a_409_refusal(door):
    resp = await door.post("/api/sessions/tg/chat", json={"message": "hi"})
    assert resp.status == 409, await resp.text()
    body = await resp.json()
    assert body["error"]["code"] == "permission_denied"
