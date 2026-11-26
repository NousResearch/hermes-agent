"""Deletion cannot erase an admitted input or an adopted worker."""
import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input, recover_session_inputs
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message, get_messages_as_conversation
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.workers._executions import register_worker_execution


@pytest.mark.parametrize('obligation', ['queued', 'started', 'unknown', 'worker'])
def test_delete_refuses_outstanding_descendant_obligations(store, obligation):
    holder = store.holder()
    create_session(holder, 's', 'test')
    create_session(holder, 'child', 'test', parent_session_id='s', model_config={'_delegate_from': 's'})
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    if obligation == 'worker':
        register_worker_execution(holder, epoch=epoch, execution_id='w', session_id='child',
            generation=0, kind='child', adoption_secret='private')
    else:
        admit_session_input(holder, epoch=epoch, principal_id='human', session_id='child', request_id='input', payload={'text': 'keep'})
        if obligation != 'queued':
            claim_session_input(holder, epoch=epoch, session_id='child')
        if obligation == 'unknown':
            epoch = begin_runtime_epoch(holder, instance_id='replacement')
            recover_session_inputs(holder, epoch=epoch)
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='delete',
                expected_revision=0, expected_generation=0, operation='delete', payload={})
    before = get_session(holder, 'child')
    with pytest.raises(AuthorityRefusal, match='session_busy|unknown_execution'):
        mutate_runtime_session(holder, **args)
    assert get_session(holder, 'child') == before
    assert get_session(holder, 's') is not None


def test_delete_receipt_survives_removal_and_fences_generation(store):
    holder = store.holder()
    create_session(holder, 's', 'test')
    create_session(holder, 'branch', 'test', parent_session_id='s')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='delete',
                expected_revision=0, expected_generation=1, operation='delete', payload={})
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        mutate_runtime_session(holder, **args)
    args['expected_generation'] = 0
    receipt, _, _ = mutate_runtime_session(holder, **args)
    assert receipt['deleted_ids'] == ['s']
    assert get_session(holder, 's') is None
    assert get_session(holder, 'branch')['parent_session_id'] is None
    assert mutate_runtime_session(holder, **args)[0] == receipt
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        mutate_runtime_session(holder, **(args | {'expected_generation': 1}))


def test_delete_of_compressed_logical_root_retires_every_physical_continuation(store):
    """Canonical admissions bind to the compression root; deleting it must remove the whole
    chain (transcript, rows, routing) for native sessions too, or the next message on the same
    route resolves to the surviving child and re-admits the 'deleted' conversation with history."""
    import json
    holder = store.holder()
    create_session(holder, 'root', 'telegram', session_key='agent:telegram:dm:1')
    append_message(holder, 'root', 'user', 'before compression')
    assert store.coordination.try_acquire_compression_lock(session_id='root', holder='holder')
    publish_compression_child(holder, parent_session_id='root', child_session_id='child', source='telegram',
                              messages=[{'role': 'user', 'content': 'summary'}], compression_lock_holder='holder')
    append_message(holder, 'child', 'user', 'after compression')
    store.routing.upsert_routing_entry('agent:telegram:dm:1', json.dumps({'session_id': 'child', 'session_key': 'agent:telegram:dm:1'}))
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    receipt, _, _ = mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                                           expected_revision=0, expected_generation=0, operation='delete', payload={})
    assert set(receipt['deleted_ids']) == {'root', 'child'}
    assert get_session(holder, 'child') is None
    assert get_messages_as_conversation(holder, 'child') == []
    assert store.routing.load_routing_entries() == {}


def _compressed(store, parent, child, model_config=None):
    assert store.coordination.try_acquire_compression_lock(session_id=parent, holder='holder')
    try:
        publish_compression_child(store.holder(), parent_session_id=parent, child_session_id=child, source='cli',
            model_config=model_config, messages=[{'role': 'user', 'content': f'history of {child}'}],
            compression_lock_holder='holder')
    finally:
        store.coordination.release_compression_lock(session_id=parent, holder='holder')


def _delegate_topology(store, shape):
    """root → (delegate | inherited fork) → its compression continuation(s); returns the tip.
    A real rotation copies the fork's model_config (marker names the ORIGINAL parent); a
    canonical compress publishes the continuation with no fork marker at all."""
    holder = store.holder()
    create_session(holder, 'root', 'telegram')
    if shape == 'inherited-branch':
        cfg = {'_branched_from': 'ancestor'}
        create_session(holder, 'ancestor', 'telegram')
        holder._execute_write(lambda c: c.execute(
            "UPDATE sessions SET parent_session_id='ancestor', model_config=? WHERE id='root'", ('{"_branched_from": "ancestor"}',)))
        _compressed(store, 'root', 'tip', cfg)
        return 'tip', {'root', 'tip'}
    create_session(holder, 'delegate', 'cli', parent_session_id='root', model_config={'_delegate_from': 'root'})
    _compressed(store, 'delegate', 'tip', {'_delegate_from': 'root'} if shape == 'inherited-delegate' else None)
    _compressed(store, 'tip', 'tip2', None)
    return 'tip2', {'root', 'delegate', 'tip', 'tip2'}


@pytest.mark.parametrize('shape', ['inherited-delegate', 'unmarked-delegate', 'inherited-branch'])
def test_queued_work_on_a_delegate_or_fork_continuation_blocks_the_delete(store, shape):
    """The descendant walk used to skip continuations whose model_config carries a fork marker
    (inherited from the fork, or absent after a canonical compress): the delete then succeeded,
    leaving the continuation and its queued admission orphaned under a deleted parent."""
    holder = store.holder()
    tip, _ = _delegate_topology(store, shape)
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    admit_session_input(holder, epoch=epoch, principal_id='human', session_id=tip, request_id='q', payload={'text': 'keep'})
    before = [dict(r) for r in holder._read_all('SELECT id, parent_session_id FROM sessions ORDER BY id')]
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                               expected_revision=0, expected_generation=0, operation='delete', payload={})
    assert [dict(r) for r in holder._read_all('SELECT id, parent_session_id FROM sessions ORDER BY id')] == before


@pytest.mark.parametrize('shape', ['inherited-delegate', 'unmarked-delegate', 'inherited-branch'])
def test_idle_delete_takes_every_delegate_and_fork_continuation(store, shape):
    holder = store.holder()
    _, chain = _delegate_topology(store, shape)
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    receipt, _, _ = mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='root', request_id='delete',
                                           expected_revision=0, expected_generation=0, operation='delete', payload={})
    assert set(receipt['deleted_ids']) == chain
    assert all(get_session(holder, sid) is None for sid in chain)
    if shape == 'inherited-branch':
        assert get_session(holder, 'ancestor') is not None
