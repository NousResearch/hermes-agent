"""Compaction callbacks retire with the turn; worker frames retain the authority fence."""
from types import SimpleNamespace

from agent.conversation_compression import COMPACTION_STATUS
from gateway import local_viewer_events
from gateway.config import Platform
from gateway.session_compaction_events import publish_worker_progress
from tests.gateway.test_local_viewer_events_turn import _turn


def test_local_turn_restores_callbacks_and_retires_unfinished_operation():
    published, legacy = [], []
    previous = lambda *args: legacy.append(args)
    agent = SimpleNamespace(status_callback=previous, compaction_callback=None, _interrupt_requested=True)
    saved = local_viewer_events.wire(_turn(Platform.LOCAL, published), agent)
    late = agent.status_callback
    late('lifecycle', COMPACTION_STATUS)
    local_viewer_events.restore(agent, saved)
    late('lifecycle', COMPACTION_STATUS)
    assert agent.status_callback is previous and agent.compaction_callback is None
    assert [p['status'] for kind, p in published if kind == 'context.compaction'] == ['in_progress', 'cancelled']
    assert legacy == [('lifecycle', COMPACTION_STATUS)] * 2


def test_worker_lifecycle_reaches_only_its_active_generation(tmp_path):
    from gateway.session_authority import LiveSession, SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='acp')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False))
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='r', payload={'text': 'go'})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        authority._stamp_claim(SimpleNamespace(session_id='s'), live, row)
        frames = []
        live.event_stream.observers.add(frames.append)
        frame = {'type': 'compaction', 'update': {'compactionId': 'operation', 'status': 'in_progress'}}
        assert publish_worker_progress(authority, 's', row['generation'], frame)
        assert len(frames) == 1 and frames[0]['params']['type'] == 'context.compaction'
        authority.executing.pop('s')
        assert publish_worker_progress(authority, 's', row['generation'], frame)
        assert len(frames) == 1
        assert not publish_worker_progress(authority, 's', row['generation'],
            {'type': 'compaction', 'update': {**frame['update'], 'summary': 'private'}})
    finally:
        db.close()
