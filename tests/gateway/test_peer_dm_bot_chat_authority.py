"""A peer turn into the canonical Bot Chat under session authority is that chat's own admission.

``hermes peer dm`` / ``peer run`` reach the API edge; under the unified runtime the Bot Chat is
a LOCAL session whose only writer is the authority FIFO, so the turn goes through the
``bot_relay.deliver`` door (``gateway/session_bot.py``) and the reply is its admission's
receipt. Binding the chat as an API conversation would be a second writer (``permission_denied``).
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_contract import Principal

AUTHOR = {"id": "bot:cto", "name": "cto", "is_bot": True}


@pytest_asyncio.fixture
async def peer(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setattr('tools.bot_mode_dm._LIVE_WAIT_SECONDS', 2.0)
    from hermes_state import SessionDB
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'bot', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Bot Chat')
    authority.db.create_session('scratch', 'api_server')
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = authority.db
    from gateway.config import Platform
    runner.adapters[Platform.API_SERVER] = adapter
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    executed = []

    async def execute(authority, ref, row):
        text = row['payload']['text']
        executed.append((ref.session_id, row['principal_id'], text, row['payload'].get('local_automation_v1', {}).get('turn_author')))
        if text == 'fail':
            raise RuntimeError('provider said 429')
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    try:
        async with TestClient(TestServer(app)) as cli:
            yield SimpleNamespace(cli=cli, authority=authority, chat=chat.session_id, executed=executed,
                                  adapter=adapter, home=tmp_path)
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store._db.close()


@pytest.mark.asyncio
async def test_peer_dm_into_the_bot_chat_is_the_chats_own_admission(peer):
    resp = await peer.cli.post(f"/api/sessions/{peer.chat}/chat", json={"message": "ping", "author": AUTHOR})
    body = await resp.json()
    assert resp.status == 200, body
    assert body['message']['content'] == 'pong' and body['delivery_id']
    assert peer.executed == [(peer.chat, 'automation:' + peer.authority.sessions[peer.chat].route, 'ping', AUTHOR)]
    # The receipt is durable and exact: the same delivery reads back settled, no re-execution.
    record = json.loads((peer.home / 'runtime' / 'bot_live_delivery' / f"{body['delivery_id']}.json").read_text())
    assert record['status'] == 'settled' and record['session_id'] == peer.chat
    rows = peer.authority.db._read_all("SELECT principal_id FROM session_admissions WHERE principal_id='api'")
    assert rows == [], 'the Bot Chat was never bound as an API conversation'


@pytest.mark.asyncio
async def test_streamed_peer_dm_and_peer_run_ride_the_same_door(peer):
    resp = await peer.cli.post(f"/api/sessions/{peer.chat}/chat/stream", json={"message": "ping", "author": AUTHOR})
    assert resp.status == 200, await resp.text()
    assert resp.content_type == "text/event-stream"
    names = [line[7:] for line in (await resp.text()).splitlines() if line.startswith('event: ')]
    assert names[0] == 'run.started' and 'assistant.completed' in names and names[-1] == 'done'

    resp = await peer.cli.post("/v1/runs", json={"input": "fail", "session_id": peer.chat, "author": AUTHOR})
    run_id = (await resp.json())['run_id']
    for _ in range(100):
        status = await (await peer.cli.get(f"/v1/runs/{run_id}")).json()
        if status['status'] in ('completed', 'failed', 'cancelled'):
            break
        await asyncio.sleep(0.05)
    assert status['status'] == 'failed' and status['delivery_id'], status
    assert [text for _, _, text, _ in peer.executed] == ['ping', 'fail']
    # A turn into any other session still runs as an ordinary API admission.
    resp = await peer.cli.post("/api/sessions/scratch/chat", json={"message": "elsewhere"})
    assert resp.status == 200 and [text for _, _, text, _ in peer.executed][-1] == 'elsewhere'
    assert peer.authority.db._read_all("SELECT 1 FROM session_admissions WHERE principal_id='api'")
