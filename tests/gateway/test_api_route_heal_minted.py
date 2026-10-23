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


def test_bind_still_refuses_a_route_held_by_a_used_session(owner):
    _mint_on_route(owner, 'api-stable')
    owner.db.append_message('minted', 'user', 'a conversation of its own')

    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_api_session(owner, 'api-stable')
