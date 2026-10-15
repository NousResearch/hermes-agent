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
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None)
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


async def _peer_minted(headless):
    created = await headless.cli.post("/api/sessions", json={"title": "Bot Chat", "source": "bot_peer_dm"})
    assert created.status == 201, await created.text()
    session_id = (await created.json())["session"]["id"]
    assert headless.authority.db.get_session(session_id)['user_id'] == 'api-peer'
    return session_id


def _local_owner(authority):
    """The Desktop/CLI's own connection: an interactive ticket redeemed on the loopback socket
    (``gateway/run_api.py``), the identity ``hermes chat``, the Desktop and ``bot_relay.deliver`` use."""
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {
        'user_id': 'uid:1000', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': sorted(_PURPOSE_CAPABILITIES['interactive']),
        'native_bootstrap': True, 'profile_scope': 'profile'}, operator=True)


async def _rpc(connection, method, **params):
    response = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in response, (method, response)
    return response['result']


@pytest.mark.asyncio
async def test_peer_minted_bot_chat_is_listed_for_the_local_owner(headless):
    session_id = await _peer_minted(headless)
    local = _local_owner(headless.authority)
    # The Bots roster's forever-chat registry and the TUI/Desktop pickers.
    by_title = await _rpc(local, 'session.list', title='Bot Chat')
    assert [row['session_id'] for row in by_title['sessions']] == [session_id]
    live = await _rpc(local, 'session.list')
    assert session_id in {row['id'] for row in live['sessions']}, live


@pytest.mark.asyncio
async def test_peer_minted_bot_chat_resumes_by_title_and_admits_a_local_turn(headless, monkeypatch):
    session_id = await _peer_minted(headless)
    local = _local_owner(headless.authority)
    # ``hermes chat -c "Bot Chat"``: the owner resolves the title.
    resumed = await _rpc(local, 'session.resume', title='Bot Chat')
    assert resumed['session_id'] == session_id
    monkeypatch.setattr(headless.authority, '_schedule', lambda ref: None)  # admission only, no agent
    await _rpc(local, 'prompt.submit', session_id=session_id, input_id='local-1', text='hello')
    rows = headless.authority.db._read_all(
        "SELECT principal_id, target_session_id FROM session_admissions WHERE principal_id='uid:1000'")
    assert [tuple(row) for row in rows] == [('uid:1000', session_id)]


@pytest.mark.asyncio
async def test_local_bot_relay_delivery_lands_in_the_peer_minted_bot_chat(headless):
    from pathlib import Path
    from gateway.session_authorities import served_profile_name
    from gateway.session_bot import deliver
    from tests.waiting import async_wait_until
    session_id = await _peer_minted(headless)
    profile = served_profile_name(Path(headless.authority.db.db_path).parent.resolve())
    # ``bot_relay.deliver``'s handler (``AuthorityConnection.bot_deliver``) on the local connection.
    record = await deliver(_local_owner(headless.authority),
                           dict(id='a' * 32, profile=profile, message='relayed', author=AUTHOR))
    assert record['session_id'] == session_id, record
    await async_wait_until(lambda: headless.executed, timeout=10, interval=.05)
    assert [(sid, text) for sid, _, text in headless.executed] == [(session_id, 'relayed')]
    titled = headless.authority.db._read_all("SELECT id FROM sessions WHERE title LIKE 'Bot Chat%'")
    assert [row[0] for row in titled] == [session_id], 'a second Bot Chat was minted'


@pytest.mark.asyncio
async def test_peer_minted_bot_chat_is_the_local_rosters_canonical_session(headless):
    from pathlib import Path
    from gateway.session_group_controls import _profiles
    session_id = await _peer_minted(headless)
    # ``profiles.list``'s roster row (the Desktop Bots roster's canonical-session discovery).
    home = Path(headless.authority.db.db_path).parent
    roster = _profiles(headless.authority, _local_owner(headless.authority).actor, home, {})
    assert (roster['profiles'][0]['canonical_session'] or {}).get('id') == session_id, roster['profiles'][0]
