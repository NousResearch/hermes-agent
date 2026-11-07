"""A profile store that moved with its home keeps its local sessions.

A profile id is the owning home's absolute path and it is stamped into every local creation receipt
and history binding. A ``state.db`` copied to another machine (a Hermes Cloud migration) or a moved
home then held rows naming the old path, and ``owned_receipt`` refused each one with
``profile_mismatch``: the Bot Chat refused every ``hermes peer dm`` / ``peer run`` and none of those
chats resumed.
"""
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionActor, SessionRef
from hermes_state_local import POLICY_PREFIX

_CAPS = frozenset({'session:create', 'session:read', 'session:submit'})


@pytest.fixture
def authority_for(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})

    async def build(profile_id, instance_id):
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
        return await initialize_session_authority(runner, profile_id=profile_id, instance_id=instance_id)
    return build


def _create(authority, request_id):
    from gateway.session_local import create_local_session
    actor = SessionActor('owner', authority.profile_id, _CAPS, 'socket')
    params = {'request_id': request_id, 'source': 'gui', 'cwd': '/', 'model': 'frozen', 'toolsets': []}
    return create_local_session(authority, actor, params)


@pytest.mark.asyncio
async def test_local_session_survives_its_home_moving(tmp_path, authority_for):
    old_home, new_home = str(tmp_path / 'mac' / 'work'), str(tmp_path / 'cloud' / 'work')
    ref = _create(await authority_for(old_home, 'mac'), 'bot-chat')

    # The same store, now served from another path: the cold boot restores the session.
    moved = await authority_for(new_home, 'cloud')
    assert ref.session_id in moved.sessions
    moved.authorize(SessionActor('owner', new_home, _CAPS, 'peer'), SessionRef(new_home, ref.session_id), 'session:submit')


@pytest.mark.asyncio
async def test_rehomed_receipt_never_restores_under_another_profiles_route(tmp_path, authority_for):
    """The store owns its rows whatever path stamped them; the route namespace still binds the profile:
    a receipt whose route names another profile's namespace is refused, as before the move."""
    from gateway.session_local_recovery import owned_receipt
    authority = await authority_for(str(tmp_path / 'home'), 'i')
    ref = _create(authority, 'mine')
    key = POLICY_PREFIX + ref.session_id
    receipt = json.loads(authority.db.get_meta(key))
    assert owned_receipt(authority, ref.session_id)[0]['route'] == receipt['route']
    authority.db.set_meta(key, json.dumps(dict(receipt, route='agent:sibling:local:dm:' + ref.session_id)))
    with pytest.raises(ValueError, match='route mismatch'):
        owned_receipt(authority, ref.session_id)

