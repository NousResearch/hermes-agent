"""The authority answers the Desktop's approval replay reads for its own sessions.

The Desktop replays ``approval.pending`` on every ``session.info`` of the routed session and acks
each card with ``approval.received``. Unanswered by the authority, both fell through to the legacy
sidecar, which holds no authority session and answered ``4001 session not found``; the renderer
reads that as a reaped runtime and re-resumes the chat mid-turn, racing a stored-page read against
the live reply (core e2e behind-window-send: the reply painted twice in the sending window).
"""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_approval_replay_reads_are_served_from_the_owner_projection(tmp_path):
    with storage_handle(tmp_path) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', persistence=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(None, 'route')
        live.controls.pending['p1'] = ('route', {
            'kind': 'approval', 'prompt_id': 'p1', 'execution_generation': 1,
            'command': 'rm -rf build', 'description': 'recursive delete', 'choices': ['once', 'deny', 'always']})
        live.controls.pending['c1'] = (object(), {'kind': 'clarify', 'prompt_id': 'c1', 'execution_generation': 1,
                                                  'question': 'which?', 'choices': [], 'multi_select': False})
        viewer = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        try:
            pending = await viewer.dispatch({'id': 1, 'method': 'approval.pending', 'params': {'session_id': 's'}})
            assert pending.get('result') == {'approvals': [{
                'request_id': 'p1', 'command': 'rm -rf build', 'description': 'recursive delete',
                'choices': ['once', 'deny', 'always'], 'allow_permanent': True}]}, pending
            received = await viewer.dispatch({'id': 2, 'method': 'approval.received',
                                              'params': {'session_id': 's', 'request_id': 'p1'}})
            assert received.get('result') == {'acknowledged': False}, received
        finally:
            await viewer.close()
