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
from hermes_agent.hermes_state_local import POLICY_PREFIX
from hermes_agent.hermes_state_runtime import RuntimeStoreError

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
    authority.db.set_meta(key, json.dumps(dict(receipt, route='agent:sibling:local:dm:' + ref.session_id)))
    with pytest.raises(ValueError, match='route mismatch'):
        owned_receipt(authority, ref.session_id)



@pytest.mark.asyncio
async def test_picker_offers_an_ended_chat_only_when_resume_would_take_it(tmp_path, authority_for):
    """The cold picker judges an ended chat by the checks a resume runs: a moved store's chat is
    offered, one whose route names another profile's namespace is not (resume refuses it)."""
    from gateway.session_listing import _stored_closed
    old_home, new_home = str(tmp_path / 'mac' / 'work'), str(tmp_path / 'cloud' / 'work')
    first = await authority_for(old_home, 'mac')
    kept, foreign = _create(first, 'kept'), _create(first, 'foreign')
    for ref in (kept, foreign):
        receipt = json.loads(first.db.get_meta(POLICY_PREFIX + ref.session_id))
        first.db.end_session(receipt['entry']['session_id'], 'closed')
    receipt = json.loads(first.db.get_meta(POLICY_PREFIX + foreign.session_id))
    first.db.set_meta(POLICY_PREFIX + foreign.session_id,
                      json.dumps(dict(receipt, route='agent:sibling:local:dm:' + foreign.session_id)))

    moved = await authority_for(new_home, 'cloud')
    actor = SessionActor('owner', new_home, _CAPS, 'peer')
    assert _stored_closed(moved, actor, kept.session_id)
    assert not _stored_closed(moved, actor, foreign.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', [
    lambda receipt: receipt['policy'].update(model=''),
    lambda receipt: receipt.update(lineage=['not-the-root', receipt['entry']['session_id']]),
], ids=['policy', 'lineage'])
async def test_picker_leaves_out_a_chat_whose_policy_or_lineage_resume_refuses(tmp_path, authority_for, damage):
    from gateway.session_listing import _stored_closed
    from gateway.session_local_recovery import restore_local_session
    authority = await authority_for(str(tmp_path / 'home'), 'i')
    ref = _create(authority, 'ended')
    key = POLICY_PREFIX + ref.session_id
    receipt = json.loads(authority.db.get_meta(key))
    authority.db.end_session(receipt['entry']['session_id'], 'closed')
    actor = SessionActor('owner', authority.profile_id, _CAPS, 'peer')
    assert _stored_closed(authority, actor, ref.session_id)
    damage(receipt)
    authority.db.set_meta(key, json.dumps(receipt))
    authority.sessions.pop(ref.session_id, None)
    with pytest.raises(RuntimeStoreError):
        restore_local_session(authority, ref.session_id)
    assert not _stored_closed(authority, actor, ref.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('corrupt', [False, True], ids=['valid', 'corrupt-lineage'])
async def test_picker_judges_a_pending_compression_catch_up_as_restore_would(tmp_path, authority_for, corrupt):
    """An older writer compacted the chat past its receipt: restore adopts the step, then checks the
    lineage. The picker judges that same projected receipt, without writing it."""
    import time
    from gateway.session_listing import _stored_closed
    from gateway.session_local_recovery import restorable_receipt
    authority = await authority_for(str(tmp_path / 'home'), 'i')
    ref = _create(authority, 'compacted')
    key = POLICY_PREFIX + ref.session_id
    receipt = json.loads(authority.db.get_meta(key))
    tip = receipt['entry']['session_id']

    def compact(conn):
        parent = conn.execute('SELECT * FROM sessions WHERE id=?', (tip,)).fetchone()
        conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (time.time(), tip))
        conn.execute("""INSERT INTO sessions(id,source,model,parent_session_id,cwd,profile_name,user_id,session_key,
            chat_id,chat_type,origin_json,started_at,ended_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ('child', 'gui', parent['model'], tip, parent['cwd'], parent['profile_name'], parent['user_id'],
             parent['session_key'], parent['chat_id'], parent['chat_type'], parent['origin_json'], time.time(), time.time()))
    authority.db._execute_write(compact)
    if corrupt:
        authority.db.set_meta(key, json.dumps(dict(receipt, lineage=['not-the-root', tip])))
    authority.sessions.pop(ref.session_id, None)
    actor = SessionActor('owner', authority.profile_id, _CAPS, 'peer')

    if corrupt:
        assert not _stored_closed(authority, actor, ref.session_id)
    else:
        assert restorable_receipt(authority, ref.session_id)[1].session_id == 'child'
        assert _stored_closed(authority, actor, ref.session_id)
    assert json.loads(authority.db.get_meta(key))['entry']['session_id'] == tip, 'the listing wrote the catch-up'
