import sqlite3

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, get_session_admission, recover_session_inputs,
    resolve_unknown_session_input, settle_session_input,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import get_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.workers._executions import adopt_worker_execution, register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution


def append(holder, *, epoch, sequence, content, **assignment):
    return mutate_worker_execution(holder, epoch=epoch, **assignment, sequence=sequence,
        operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': content}]})


@pytest.mark.parametrize('recover_first', [False, True])
def test_explicit_worker_adoption_preserves_claim_without_reexecution(backend, scope_a, recover_first):
    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    accepted, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'work'})
    claim = claim_session_input(holder, epoch=epoch, session_id='s')
    assignment = dict(execution_id='compute', session_id='s', generation=claim['generation'])
    register_worker_execution(holder, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
    new_epoch = begin_runtime_epoch(holder, instance_id='replacement')
    if recover_first:
        recover_session_inputs(holder, epoch=new_epoch)
    adopt_worker_execution(holder, epoch=new_epoch, **assignment, adoption_secret='private')
    recover_session_inputs(holder, epoch=new_epoch)
    restored = get_session_admission(holder, admission_id=accepted['admission_id'])
    assert restored['status'] == 'started' and restored['owner_epoch'] == new_epoch
    assert restored['generation'] == claim['generation']
    assert claim_session_input(holder, epoch=new_epoch, session_id='s') is None
    message_args = dict(epoch=new_epoch, **assignment, sequence=1, content='committed')
    receipt = append(holder, **message_args)
    result = settle_session_input(holder, epoch=new_epoch, admission_id=accepted['admission_id'], generation=claim['generation'], outcome='completed')
    assert result['status'] == 'terminal'
    before = holder._conn.total_changes
    assert append(holder, **message_args) == receipt
    assert holder._conn.total_changes == before
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        append(holder, epoch=new_epoch, **assignment, sequence=2, content='late')


@pytest.mark.parametrize('refuse_revoke', [False, True])
def test_unknown_resolution_atomically_revokes_worker_before_unblocking(backend, scope_a, refuse_revoke):
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    first, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s', request_id='first', payload={})
    claim = claim_session_input(holder, epoch=epoch, session_id='s')
    assignment = dict(execution_id='worker', session_id='s', generation=claim['generation'])
    register_worker_execution(holder, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
    follower, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s', request_id='next', payload={})
    epoch = begin_runtime_epoch(holder, instance_id='replacement')
    recover_session_inputs(holder, epoch=epoch)
    args = dict(epoch=epoch, admission_id=first['admission_id'], generation=claim['generation'])
    before = get_session(holder, 's')
    if refuse_revoke:
        holder._execute_write(lambda c: c.execute("CREATE TRIGGER refuse_revoke BEFORE UPDATE OF status ON worker_executions BEGIN SELECT RAISE(ABORT, 'revoke refused'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='revoke refused'):
            resolve_unknown_session_input(holder, **args)
        assert get_session_admission(holder, admission_id=first['admission_id'])['status'] == 'unknown'
        assert get_session(holder, 's') == before
        holder._execute_write(lambda c: c.execute('DROP TRIGGER refuse_revoke'))
    result = resolve_unknown_session_input(holder, **args)
    assert result['status'] == 'terminal' and result['outcome'] == 'interrupted'
    backend.reopen()
    holder = store.holder()
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        adopt_worker_execution(holder, epoch=epoch, **assignment, adoption_secret='private')
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        append(holder, epoch=epoch, **assignment, sequence=1, content='late')
    assert get_messages(holder, 's') == []
    assert claim_session_input(holder, epoch=epoch, session_id='s')['admission_id'] == follower['admission_id']
