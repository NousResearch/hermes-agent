"""A delegated child's own transcript, before and after an owner-epoch takeover.

The child runner shares its caller's turn context but writes its OWN session's rows: live
it writes through the same store append; once a later owner epoch takes over, the write is
refused and the child transcript stays empty — the port's append keeps the fence.
"""
import threading
from types import SimpleNamespace
from contextlib import closing

import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB
from hermes_state_errors import TurnFenceSupersededError
from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence
from agent.session_persistence import _db_flush_write
from tools import async_delegation as ad


@pytest.mark.parametrize('takeover', [False, True])
def test_delegated_child_transcript(tmp_path, monkeypatch, takeover):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ad._reset_for_tests()
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        db.create_session('child', 'tool', parent_session_id='s', model_config={'_delegate_from': 's'})
        epoch = rt.begin_runtime_epoch(db, instance_id='one')
        rt.admit_session_input(db, epoch=epoch, principal_id='u', session_id='s',
                               request_id='r', payload={})
        row = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assert isinstance(row, dict)
        ready, go, done = threading.Event(), threading.Event(), threading.Event()
        outcome = []

        def runner():
            ready.set()
            assert go.wait(10)
            agent = SimpleNamespace(_session_db=db, session_id='child')
            rows = [{'role': 'assistant', 'content': 'delegated transcript'}]
            try:
                _db_flush_write(agent, rows, rows, rows)
                outcome.append('written')
            except TurnFenceSupersededError:
                outcome.append('refused')
            finally:
                done.set()
            return {'status': 'completed'}

        with bind_profile_scope(tmp_path), bind_turn('s', row['admission_id'], claim_fence(row),
                                                     store=db.db_path):
            ad.dispatch_async_delegation(goal='probe', context=None, toolsets=None,
                                         role='leaf', model=None, session_key='',
                                         parent_session_id='s', runner=runner)
        assert ready.wait(10)
        if takeover:
            rt.begin_runtime_epoch(db, instance_id='two')
        go.set()
        assert done.wait(10)
        assert outcome == (['refused'] if takeover else ['written'])
        assert [m['content'] for m in db.get_messages('child')] == \
            ([] if takeover else ['delegated transcript'])
        # Let durable completion finalize before this store closes.
        from tests.waiting import wait_until
        wait_until(lambda: ad.active_count() == 0, timeout=10)
    ad._reset_for_tests()
