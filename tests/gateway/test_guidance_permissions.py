"""The canonical guidance dispatcher independently enforces control and read rights."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_guidance import SessionGuidance
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
async def test_guidance_control_and_read_are_independently_authorized(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    viewers = []
    try:
        db.create_session('s', source='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='owner', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='owner'))
        live = authority.sessions['s'] = LiveSession(None, 'route')
        live.guidance = SessionGuidance('owner:epoch', 's')
        live.guidance.start({'admission_id': 'a', 'generation': 1})
        target = live.guidance.snapshot('a')
        params = {'session_id': 's', 'runtime_instance_id': target['runtimeInstanceId'],
                  'expected_turn_id': target['turnId'], 'message_id': 'm'}
        for capabilities in ([], ['session:read'], ['session:control']):
            viewer = AuthorityConnection(authority, object(), {'user_id': 'human', 'capabilities': capabilities})
            viewers.append(viewer)
            admitted = await viewer.dispatch({'id': 1, 'method': 'session.guidance.admit',
                                              'params': {**params, 'text': 'instruction'}})
            if 'session:control' in capabilities:
                assert admitted['result']['state'] == 'pending_incorporation'
            else:
                assert admitted['error']['message'] == 'permission_denied'
            status = await viewer.dispatch({'id': 2, 'method': 'session.guidance.status', 'params': params})
            if 'session:read' in capabilities:
                assert status['result']['disposition'] == 'unknown_message'
            else:
                assert status['error']['message'] == 'permission_denied'
        assert len(live.guidance.by_message) == 1
    finally:
        for viewer in viewers:
            await viewer.close()
        db.close()
