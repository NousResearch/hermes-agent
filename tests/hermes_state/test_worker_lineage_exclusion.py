"""A physical worker excludes its whole FIFO lineage; an explicit reset revokes orphan unknowns."""
import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_physical_worker_excludes_logical_fifo_and_second_registration(tmp_path):
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        logical = _local_reset(db, epoch, 'logical', 'physical')
        rt.register_worker_execution(db, epoch=epoch, execution_id='standalone-worker', session_id='physical',
                                     generation=0, kind='compute', adoption_secret='secret')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=logical,
                               request_id='later', payload={'text': 'later'})
        assert rt.claim_session_input(db, epoch=epoch, session_id=logical) is None
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.register_worker_execution(db, epoch=epoch, execution_id='second-worker', session_id=logical,
                generation=db.get_session(logical)['runtime_generation'], kind='compute', adoption_secret='secret')


def test_reset_revokes_only_orphan_unknown_worker_and_unblocks_fifo(tmp_path):
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        logical = _local_reset(db, epoch, 'logical', 'physical')
        generation = db.get_session(logical)['runtime_generation']
        rt.register_worker_execution(db, epoch=epoch, execution_id='standalone-worker', session_id=logical,
                                     generation=generation, kind='compute', adoption_secret='secret')
        epoch = rt.begin_runtime_epoch(db, instance_id='restarted')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=logical,
                               request_id='later', payload={'text': 'later'})
        with pytest.raises(rt.RuntimeStoreError, match='unknown_execution'):
            rt.claim_session_input(db, epoch=epoch, session_id=logical)
        receipt = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=logical,
            request_id='explicit-reset', expected_revision=db.get_session(logical)['runtime_revision'],
            expected_generation=db.get_session(logical)['runtime_generation'],
            operation='reset', payload={})
        assert receipt['target_session_id'] != 'physical'
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
        assert rt.claim_session_input(db, epoch=epoch, session_id=logical)['request_id'] == 'later'
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.adopt_worker_execution(db, epoch=epoch, execution_id='standalone-worker',
                session_id=logical, generation=generation, adoption_secret='secret')
