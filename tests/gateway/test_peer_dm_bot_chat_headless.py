"""A headless gateway's first peer DM lands in a Bot Chat its own authority FIFO owns.

``hermes peer dm`` creates the remote canonical chat through ``POST /api/sessions`` when it is
missing. With no Desktop/CLI in the loop that create used to mint an API row nothing can ever bind
(the peer is a remote sender, never a native ticket), so every DM answered 502 "admission refused:
not_found" (#139370). The titled create now goes through the owner's own Bot Chat creation step;
a peer-created API row left over from before is refused by a named reason, never a bare not_found.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

AUTHOR = {"id": "bot:cto", "name": "cto", "is_bot": True}


@pytest_asyncio.fixture
async def headless(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})
    from hermes_state import SessionDB
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    # The gateway owns its home's session authority; no Desktop/CLI ever made a "Bot Chat".
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = authority.db
    runner.adapters[Platform.API_SERVER] = adapter
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    executed = []

    async def execute(authority, ref, row):
        executed.append((ref.session_id, row['principal_id'], row['payload']['text']))
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    app = web.Application()
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    try:
        async with TestClient(TestServer(app)) as cli:
            yield SimpleNamespace(cli=cli, authority=authority, executed=executed)
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store._db.close()


def _api_admissions(authority):
    return authority.db._read_all("SELECT 1 FROM session_admissions WHERE principal_id='api'")


@pytest.mark.asyncio
async def test_headless_peer_dm_creates_the_bot_chat_through_its_owner(headless):
    # The peer client's own creation path (hermes_cli/subcommands/peer.py::_ensure_bot_chat).
    created = await headless.cli.post("/api/sessions", json={"title": "Bot Chat", "source": "bot_peer_dm"})
    assert created.status == 201, await created.text()
    session_id = (await created.json())["session"]["id"]
    row = headless.authority.db.get_session_by_title('Bot Chat')
    assert row['id'] == session_id and str(row['chat_id']).startswith('local-'), row

    for text in ('one', 'two'):
        resp = await headless.cli.post(f"/api/sessions/{session_id}/chat", json={"message": text, "author": AUTHOR})
        body = await resp.json()
        assert resp.status == 200, body
        assert body['message']['content'] == 'pong' and body['delivery_id']
    route = headless.authority.sessions[session_id].route
    assert headless.executed == [(session_id, 'automation:' + route, 'one'),
                                 (session_id, 'automation:' + route, 'two')]
    assert _api_admissions(headless.authority) == [], 'the Bot Chat was never bound as an API conversation'
    # Once it exists the titled create keeps today's answer: the title is taken.
    again = await headless.cli.post("/api/sessions", json={"title": "Bot Chat", "source": "bot_peer_dm"})
    assert again.status == 400 and (await again.json())['error']['code'] == 'invalid_title'


@pytest.mark.asyncio
async def test_peer_created_api_bot_chat_is_refused_by_name(headless):
    # A row an older gateway minted through the plain API create: no FIFO can own it.
    db = headless.authority.db
    db.create_session('api_1_stale', 'bot_peer_dm')
    db.set_session_title('api_1_stale', 'Bot Chat')
    resp = await headless.cli.post("/api/sessions/api_1_stale/chat", json={"message": "ping", "author": AUTHOR})
    body = await resp.json()
    assert resp.status == 502, body
    assert body['error']['code'] == 'bot_chat_not_local', body
    assert 'not_found' not in body['error']['message']
    assert headless.executed == [] and _api_admissions(headless.authority) == []
