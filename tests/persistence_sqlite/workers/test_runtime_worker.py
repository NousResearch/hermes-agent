import pytest
import sqlite3

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input, recover_session_inputs
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from persistence.sqlite.workers import _executions
from persistence.sqlite.workers._executions import adopt_worker_execution, register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution


def setup_worker(store):
    assert callable(getattr(_executions, 'register_worker_execution', None)), 'canonical worker receipts missing'
    holder = store.holder()
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    assignment = dict(execution_id='worker', session_id='s', generation=0)
    register_worker_execution(holder, epoch=epoch, **assignment, kind='compute', adoption_secret='private-fixture')
    return holder, epoch, assignment


def append(holder, *, epoch, execution_id, session_id, generation, sequence, content, role='assistant'):
    """The worker's production text write: one ``transcript.append`` receipt."""
    return mutate_worker_execution(holder, epoch=epoch, execution_id=execution_id, session_id=session_id,
        generation=generation, sequence=sequence, operation='transcript.append',
        payload={'messages': [{'role': role, 'content': content}]})


def finish(holder, *, epoch, execution_id, session_id, generation, sequence):
    return mutate_worker_execution(holder, epoch=epoch, execution_id=execution_id, session_id=session_id,
        generation=generation, sequence=sequence, operation='execution.finish', payload={})


def test_worker_receipts_require_exact_sequence_assignment_and_adoption(backend, scope_a):
    store = backend.open(scope_a)
    holder, epoch, assignment = setup_worker(store)
    assert claim_session_input(holder, epoch=epoch, session_id='s') is None
    args = dict(epoch=epoch, **assignment, sequence=1, content='kept output')
    result = append(holder, **args)
    assert append(holder, **args) == result
    assert len(get_messages(holder, 's')) == 1
    for change, reason in [({'content': 'different'}, 'admission_conflict'), ({'sequence': 3}, 'invalid_params'), ({'session_id': 'other'}, 'permission_denied'), ({'generation': 1}, 'stale_generation')]:
        with pytest.raises(AuthorityRefusal, match=reason):
            append(holder, **(args | change))
    new_epoch = begin_runtime_epoch(holder, instance_id='replacement')
    recover_session_inputs(holder, epoch=new_epoch)
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        append(holder, **args)
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        append(holder, **(args | {'epoch': new_epoch}))
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        adopt_worker_execution(holder, epoch=new_epoch, **assignment, adoption_secret='wrong')
    adopted = adopt_worker_execution(holder, epoch=new_epoch, **assignment, adoption_secret='private-fixture')
    assert adopted['owner_epoch'] == new_epoch and 'adoption_digest' not in adopted
    assert append(holder, **(args | {'epoch': new_epoch})) == result
    append(holder, **(args | {'epoch': new_epoch, 'sequence': 2, 'content': 'next'}))
    finished = finish(holder, epoch=new_epoch, **assignment, sequence=3)
    backend.reopen()
    holder = store.holder()
    before = holder._conn.total_changes
    assert finish(holder, epoch=new_epoch, **assignment, sequence=3) == finished
    assert append(holder, **(args | {'epoch': new_epoch})) == result
    assert holder._conn.total_changes == before
    assert finished == {'status': 'terminal'}
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        append(holder, **(args | {'epoch': new_epoch, 'content': 'changed'}))
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        adopt_worker_execution(holder, epoch=new_epoch, **assignment, adoption_secret='private-fixture')
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        append(holder, **(args | {'epoch': new_epoch, 'sequence': 4}))
    assert [r['content'] for r in get_messages(holder, 's')] == ['kept output', 'next']
    admit_session_input(holder, epoch=new_epoch, principal_id='human', session_id='s', request_id='next', payload={})
    assert claim_session_input(holder, epoch=new_epoch, session_id='s')['generation'] > assignment['generation']
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        append(holder, **(args | {'epoch': new_epoch}))
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        finish(holder, epoch=new_epoch, **assignment, sequence=3)


@pytest.mark.parametrize('content', ['once', 'a' + chr(0xD800) + 'b'])
def test_worker_mutation_and_receipt_roll_back_together(backend, scope_a, content):
    store = backend.open(scope_a)
    holder, epoch, assignment = setup_worker(store)
    holder._conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON worker_receipts BEGIN SELECT RAISE(ABORT, 'receipt fixture'); END")
    create_session(holder, 'direct', 'test')
    append_message(holder, 'direct', 'assistant', content)
    expected_content = get_messages(holder, 'direct')[0]['content']
    args = dict(epoch=epoch, **assignment, sequence=1, content=content)
    with pytest.raises(sqlite3.IntegrityError, match='receipt fixture'):
        append(holder, **args)
    assert get_messages(holder, 's') == []
    assert get_session(holder, 's')['message_count'] == 0
    assert holder._conn.execute("SELECT last_sequence FROM worker_executions WHERE execution_id='worker'").fetchone()[0] == 0
    holder._conn.execute('DROP TRIGGER reject_receipt')
    receipt = append(holder, **args)
    backend.reopen()
    holder = store.holder()
    assert append(holder, **args) == receipt
    assert get_messages(holder, 's')[0]['content'] == expected_content
    if content != expected_content:
        with pytest.raises(AuthorityRefusal, match='admission_conflict'):
            append(holder, **(args | {'content': expected_content}))
    assert len(get_messages(holder, 's')) == get_session(holder, 's')['message_count'] == 1


@pytest.mark.parametrize('closed', [False, True])
def test_worker_append_preserves_compression_write_guards(backend, scope_a, closed):
    from persistence.sqlite._state.errors import CompressionSessionClosedError

    store = backend.open(scope_a)
    holder, epoch, assignment = setup_worker(store)
    args = dict(epoch=epoch, **assignment, sequence=1, content='before closure')
    receipt = append(holder, **args)
    assert store.coordination.try_acquire_compression_lock(session_id='s', holder='compressor')
    if closed:
        end_session(holder, 's', end_reason='compression')
        create_session(holder, 'tip', 'test', parent_session_id='s')
    before = get_session(holder, 's')
    messages = get_messages(holder, 's')
    next_args = args | {'sequence': 2, 'content': 'new output'}
    if closed:
        with pytest.raises(CompressionSessionClosedError):
            append_message(holder, 's', 'assistant', 'ordinary')
        with pytest.raises(CompressionSessionClosedError):
            append(holder, **next_args)
        assert get_session(holder, 's') == before
        assert get_messages(holder, 's') == messages
        assert holder._conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 1
        assert holder._conn.execute('SELECT last_sequence FROM worker_executions').fetchone()[0] == 1
    else:
        append_message(holder, 's', 'assistant', 'ordinary')
        append(holder, **next_args)
        assert [r['content'] for r in get_messages(holder, 's')] == ['before closure', 'ordinary', 'new output']
    backend.reopen()
    holder = store.holder()
    assert append(holder, **args) == receipt
