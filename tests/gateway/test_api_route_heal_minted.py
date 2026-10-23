"""An API route the stale-route heal (#54878) handed to an empty minted session goes back to its caller."""
from datetime import datetime
import json

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_api import bind_api_session
from hermes_state_runtime import RuntimeStoreError


def _mint_on_route(owner, caller):
    """What the heal leaves: a fresh session row owning the caller's API route."""
    store = owner.runner.session_store
    route = store._generate_session_key(SessionSource(platform=Platform.API_SERVER, chat_id=caller,
                                                      user_id='api', chat_type='dm'))
    owner.db.create_session('minted', source='api_server')
    now = datetime.now()
    minted = SessionEntry(route, 'minted', now, now, platform=Platform.API_SERVER)
    owner.db._execute_write(lambda conn: conn.execute(
        "INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,0) "
        'ON CONFLICT(scope,session_key) DO UPDATE SET entry_json=excluded.entry_json',
        (route, json.dumps(minted.to_dict()))))
    with store._lock:
        store._ensure_loaded_locked()
        store._entries[route] = minted
    return route


@pytest.mark.parametrize('bound_before', [False, True])
def test_bind_takes_the_route_back_from_an_empty_minted_session(owner, bound_before):
    if bound_before:
        bind_api_session(owner, 'api-stable')
    route = _mint_on_route(owner, 'api-stable')

    bind_api_session(owner, 'api-stable')

    with owner.runner.session_store._lock:
        assert owner.runner.session_store._entries[route].session_id == 'api-stable'


def test_a_finalized_bound_row_keeps_its_route_across_later_runs(owner):
    """The live #54878 shape: the bound row is finalized, then its caller runs twice more."""
    ref = bind_api_session(owner, 'api-stable')
    source = owner.sessions['api-stable'].source
    owner.db.end_session('api-stable', 'cli_close')

    for turn in ('two', 'three'):
        assert bind_api_session(owner, 'api-stable') == ref
        assert owner.runner.session_store.get_or_create_session(source).session_id == 'api-stable'
        owner.db.append_message('api-stable', 'user', turn)


@pytest.mark.asyncio
async def test_the_next_admitted_turn_reopens_a_finalized_api_row(tmp_path, monkeypatch):
    from gateway import session_api_turn, session_finite
    from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    monkeypatch.setattr(session_api_turn, 'check_api_turn', lambda authority, ref, payload: None)
    ended = []

    async def execute(authority, ref, row):
        ended.append(db.get_session('s')['end_reason'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    db.end_session('s', 'cli_close')

    with db:
        authority._schedule = lambda ref: None
        await _submit(authority, 'turn')
        del authority._schedule
        authority._schedule(REF)
        await authority.sessions['s'].task

    assert ended == [None]


def test_bind_still_refuses_a_route_held_by_a_used_session(owner):
    _mint_on_route(owner, 'api-stable')
    owner.db.append_message('minted', 'user', 'a conversation of its own')

    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_api_session(owner, 'api-stable')
