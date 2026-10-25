"""An agent can hand a turn into one of the owner's desktop (local) chats through the API edge.

The turn enters that chat's authority FIFO (the door Bot Chat deliveries use), labelled with the
sending bot as its author, never as the owner. An API conversation is left to the API path; a
message without a bot author is refused by name.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions


@pytest_asyncio.fixture
async def desk(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    runner.session_authority = authority
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    chat = create_local_session(authority, owner, {'request_id': 'desk', 'source': 'gui', 'cwd': str(tmp_path),
                                                    'model': 'fixture', 'toolsets': []})
    title_new_session(authority, chat, 'Clyde Rebuild')
    seen = []

    async def execute(authority, ref, row):
        seen.append(row)
        return 'pong'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        yield SimpleNamespace(authority=authority, runner=runner, chat=chat.session_id, seen=seen, home=tmp_path)
    finally:
        store._db.close()


async def _settled(desk):
    for _ in range(200):
        await asyncio.sleep(0.02)
        rows = list_session_admissions(desk.authority.db, session_id=desk.chat, pending_only=False)
        if rows and all(r['status'] == 'terminal' for r in rows):
            return rows
    raise AssertionError('never settled')


@pytest.mark.asyncio
async def test_bot_author_handoff_lands_in_the_desktop_chat_as_the_bot(desk):
    from gateway.platforms.api_server_bot_chat import admit_peer_turn
    author = {'id': 'bot:picasso', 'name': 'Picasso', 'is_bot': True}
    admitted = await admit_peer_turn(desk.runner, desk.chat, 'Handoff: please review the diff', author)
    assert admitted is not None
    _home, record = admitted
    assert record['status'] in ('queued', 'claimed', 'settled'), record
    rows = await _settled(desk)
    payload = rows[-1]['payload']
    assert payload['text'] == 'Handoff: please review the diff'
    assert payload['local_automation_v1']['turn_author'] == author
    assert rows[-1]['principal_id'].startswith('automation:')  # never the owner's own prompt
    assert desk.seen and desk.seen[-1]['admission_id'] == rows[-1]['admission_id']


@pytest.mark.asyncio
async def test_handoff_without_a_bot_author_is_refused_by_name(desk):
    from gateway.platforms.api_server_bot_chat import admit_peer_turn
    for author in (None, {'id': 'uid:1000', 'name': 'Casey', 'is_bot': False}, {'id': 'casey', 'name': 'x', 'is_bot': True}):
        _home, record = await admit_peer_turn(desk.runner, desk.chat, 'hello', author)
        assert record['status'] == 'failed' and record['reason'] == 'handoff_author_required', (author, record)
    assert list_session_admissions(desk.authority.db, session_id=desk.chat, pending_only=False) == []


@pytest.mark.asyncio
async def test_non_local_or_unknown_session_is_left_to_the_api_path(desk):
    from gateway.platforms.api_server_bot_chat import admit_peer_turn
    author = {'id': 'bot:picasso', 'name': 'Picasso', 'is_bot': True}
    assert await admit_peer_turn(desk.runner, 'no-such-session', 'hello', author) is None
