"""A headless gateway's peer-created "Bot Chat" answers a peer DM as an ordinary API turn.

``hermes peer dm`` creates the remote canonical chat through ``POST /api/sessions``
(``source: bot_peer_dm``) when it is missing. On a headless gateway no desktop/CLI lane ever
made that chat a bound local session, so the authority FIFO door (``admit_peer_turn``) cannot
admit it — the API row is not adoptable by the submit-only peer principal. Such a turn must
fall back to running as an API turn (#139370), not answer 502 "admission refused: not_found".
"""
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
async def headless(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    from hermes_state import SessionDB
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    # A headless gateway still owns its launch home's session authority; what it lacks is any
    # desktop/CLI lane that ever turned "Bot Chat" into a bound local session.
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = authority.db
    runner.adapters[Platform.API_SERVER] = adapter
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    executed = []

    async def execute(authority, ref, row):
        text = row['payload']['text']
        executed.append((ref.session_id, row['principal_id'], text,
                         row['payload'].get('local_automation_v1', {}).get('turn_author')))
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    app = web.Application()
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    try:
        async with TestClient(TestServer(app)) as cli:
            yield SimpleNamespace(cli=cli, authority=authority, executed=executed, home=tmp_path)
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store._db.close()


@pytest.mark.asyncio
async def test_headless_peer_created_bot_chat_answers_as_an_api_turn(headless):
    # The peer client's own creation path: a titled API row no authority FIFO owns.
    created = await headless.cli.post("/api/sessions", json={"title": "Bot Chat", "source": "bot_peer_dm"})
    assert created.status == 201, await created.text()
    session_id = (await created.json())["session"]["id"]

    resp = await headless.cli.post(f"/api/sessions/{session_id}/chat",
                                   json={"message": "ping", "author": AUTHOR})
    body = await resp.json()
    assert resp.status == 200, body
    assert body['message']['content'] == 'pong'
    assert [text for _, _, text, _ in headless.executed] == ['ping']
    # It ran as this chat's own API admission — no FIFO delivery was pinned or refused.
    assert headless.authority.db._read_all("SELECT 1 FROM session_admissions WHERE principal_id='api'")
    assert not list((headless.home / 'runtime' / 'bot_live_delivery').glob('*.json'))


@pytest.mark.asyncio
async def test_repeat_headless_peer_dms_keep_answering_as_api_turns(headless):
    """The fallback lives in the chat handler's own handoff, so a repeat DM keeps working and the
    titled row stays the API chat it was created as (nothing adopted it into local policy)."""
    created = await headless.cli.post("/api/sessions", json={"title": "Bot Chat", "source": "bot_peer_dm"})
    session_id = (await created.json())["session"]["id"]
    first = await headless.cli.post(f"/api/sessions/{session_id}/chat",
                                    json={"message": "one", "author": AUTHOR})
    second = await headless.cli.post(f"/api/sessions/{session_id}/chat",
                                     json={"message": "two", "author": AUTHOR})
    assert first.status == 200 and second.status == 200
    assert [text for _, _, text, _ in headless.executed] == ['one', 'two']
    # The titled row stays the API chat it was created as: its API binding, never a local chat_id.
    assert not str(headless.authority.db.get_session(session_id)['chat_id'] or '').startswith('local-')
