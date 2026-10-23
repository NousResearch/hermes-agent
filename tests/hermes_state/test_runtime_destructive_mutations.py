"""Deletion cannot erase an admitted input or an adopted worker."""
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.mark.parametrize('obligation', ['queued', 'started', 'unknown', 'worker'])
def test_delete_refuses_outstanding_descendant_obligations(tmp_path, obligation):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.create_session('child', source='test', parent_session_id='s', model_config={'_delegate_from': 's'})
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        if obligation == 'worker':
            rt.register_worker_execution(db, epoch=epoch, execution_id='w', session_id='child',
                generation=0, kind='child', adoption_secret='private')
        else:
            rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='child', request_id='input', payload={'text': 'keep'})
            if obligation != 'queued':
                rt.claim_session_input(db, epoch=epoch, session_id='child')
            if obligation == 'unknown':
                epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
                rt.recover_session_inputs(db, epoch=epoch)
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='delete',
                    expected_revision=0, expected_generation=0, operation='delete', payload={})
        before = db.get_session('child')
        with pytest.raises(rt.RuntimeStoreError, match='session_busy|unknown_execution'):
            rt.mutate_runtime_session(db, **args)
        assert db.get_session('child') == before
        assert db.get_session('s') is not None


def test_delete_receipt_survives_removal_and_fences_generation(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.create_session('branch', source='test', parent_session_id='s')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='delete',
                    expected_revision=0, expected_generation=1, operation='delete', payload={})
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.mutate_runtime_session(db, **args)
        args['expected_generation'] = 0
        receipt = rt.mutate_runtime_session(db, **args)
        assert receipt['deleted_ids'] == ['s']
        assert db.get_session('s') is None
        assert db.get_session('branch')['parent_session_id'] is None
        assert rt.mutate_runtime_session(db, **args) == receipt
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.mutate_runtime_session(db, **(args | {'expected_generation': 1}))


def test_delete_of_compressed_logical_root_retires_every_physical_continuation(tmp_path):
    """Canonical admissions bind to the compression root; deleting it must remove the whole
    chain (transcript, rows, routing) for native sessions too, or the next message on the same
    route resolves to the surviving child and re-admits the 'deleted' conversation with history."""
    import json
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('root', source='telegram', session_key='agent:telegram:dm:1')
        db.append_message('root', 'user', 'before compression')
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='child', source='telegram',
                                     messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
        db.append_message('child', 'user', 'after compression')
        db.save_gateway_routing_entry('agent:telegram:dm:1', json.dumps({'session_id': 'child', 'session_key': 'agent:telegram:dm:1'}))
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        receipt = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                                            expected_revision=0, expected_generation=0, operation='delete', payload={})
        assert set(receipt['deleted_ids']) == {'root', 'child'}
        assert db.get_session('child') is None
        assert db.get_messages_as_conversation('child') == []
        assert db.load_gateway_routing_entries() == {}


def _compressed(db, parent, child, model_config=None):
    assert db.try_acquire_compression_lock(parent, 'holder')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child, source='cli',
            model_config=model_config, messages=[{'role': 'user', 'content': f'history of {child}'}],
            compression_lock_holder='holder')
    finally:
        db.release_compression_lock(parent, 'holder')


def _delegate_topology(db, shape):
    """root → (delegate | inherited fork) → its compression continuation(s); returns the tip.
    A real rotation copies the fork's model_config (marker names the ORIGINAL parent); a
    canonical compress publishes the continuation with no fork marker at all."""
    db.create_session('root', source='telegram')
    if shape == 'inherited-branch':
        cfg = {'_branched_from': 'ancestor'}
        db.create_session('ancestor', source='telegram')
        db._execute_write(lambda c: c.execute(
            "UPDATE sessions SET parent_session_id='ancestor', model_config=? WHERE id='root'", ('{"_branched_from": "ancestor"}',)))
        _compressed(db, 'root', 'tip', cfg)
        return 'tip', {'root', 'tip'}
    db.create_session('delegate', source='cli', parent_session_id='root', model_config={'_delegate_from': 'root'})
    _compressed(db, 'delegate', 'tip', {'_delegate_from': 'root'} if shape == 'inherited-delegate' else None)
    _compressed(db, 'tip', 'tip2', None)
    return 'tip2', {'root', 'delegate', 'tip', 'tip2'}


@pytest.mark.parametrize('shape', ['inherited-delegate', 'unmarked-delegate', 'inherited-branch'])
def test_queued_work_on_a_delegate_or_fork_continuation_blocks_the_delete(tmp_path, shape):
    """The descendant walk used to skip continuations whose model_config carries a fork marker
    (inherited from the fork, or absent after a canonical compress): the delete then succeeded,
    leaving the continuation and its queued admission orphaned under a deleted parent."""
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        tip, _ = _delegate_topology(db, shape)
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=tip, request_id='q', payload={'text': 'keep'})
        before = [dict(r) for r in db._read_all('SELECT id, parent_session_id FROM sessions ORDER BY id')]
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                                      expected_revision=0, expected_generation=0, operation='delete', payload={})
        assert [dict(r) for r in db._read_all('SELECT id, parent_session_id FROM sessions ORDER BY id')] == before


@pytest.mark.parametrize('shape', ['inherited-delegate', 'unmarked-delegate', 'inherited-branch'])
def test_idle_delete_takes_every_delegate_and_fork_continuation(tmp_path, shape):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        _, chain = _delegate_topology(db, shape)
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        receipt = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                                            expected_revision=0, expected_generation=0, operation='delete', payload={})
        assert set(receipt['deleted_ids']) == chain
        assert all(db.get_session(sid) is None for sid in chain)
        if shape == 'inherited-branch':
            assert db.get_session('ancestor') is not None
