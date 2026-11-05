"""Owner-reserved delegated children of a worker execution (#141793): the owner mints every child,
caps them, scopes their writes to their own row, and retires them with their parent."""
import os
from functools import partial

import psutil
import pytest

from persistence.core.errors import AuthorityRefusal

from hermes_cli.process_identity import worker_birth
from persistence.sqlite.authority._admission_store import recover_session_inputs
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import get_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.workers._executions import adopt_worker_execution, register_worker_execution
from persistence.sqlite.workers._gateway import reserve_child_execution
from persistence.sqlite.workers._receipt import apply_worker_receipt, mutate_worker_execution


def _reserve(holder, epoch, sequence, *, limits=(2, 1), refused=None, execution_id='worker'):
    handler = partial(reserve_child_execution, epoch=epoch, execution_id=execution_id,
                      identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=limits, refused=refused)
    return apply_worker_receipt(holder, epoch=epoch, execution_id=execution_id, session_id='parent', generation=0,
                                sequence=sequence, operation='child.reserve',
                                payload={'secret': f'child-secret-{sequence:08d}'}, handlers={'child.reserve': handler})


def _child_append(holder, epoch, child, sequence, content, session_id=None):
    return mutate_worker_execution(holder, epoch=epoch, execution_id=child['execution_id'],
        session_id=session_id or child['session_id'], generation=0, sequence=sequence,
        operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': content}]})


@pytest.fixture
def ledger(backend, scope_a):
    holder = backend.open(scope_a).holder()
    create_session(holder, 'parent', 'kanban', profile_name='coder')
    create_session(holder, 'other', 'cli')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    register_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                              kind='compute', adoption_secret='private-fixture')
    return holder, epoch


def test_owner_mints_caps_and_scopes_children_and_retires_them_with_the_parent(ledger):
    holder, epoch = ledger
    first = _reserve(holder, epoch, 1)['child']
    second = _reserve(holder, epoch, 2)['child']
    # Concurrency and draining caps are receipts, so the parent's journal never sticks.
    assert _reserve(holder, epoch, 3) == {'refused': 'delegation_capacity'}
    assert _reserve(holder, epoch, 4, refused='runtime_draining') == {'refused': 'runtime_draining'}
    row = get_session(holder, first['session_id'])
    assert (row['source'], row['parent_session_id']) == ('subagent', 'parent')
    with holder._read_ctx() as conn:
        kinds = conn.execute("SELECT kind,status FROM worker_executions WHERE execution_id!='worker'").fetchall()
    assert [tuple(r) for r in kinds] == [('child', 'running')] * 2
    # A child writes its own row only.
    _child_append(holder, epoch, first, 1, 'CHILD_OUTPUT')
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        _child_append(holder, epoch, first, 2, 'escape', session_id='parent')
    assert [m['content'] for m in get_messages(holder, first['session_id'])] == ['CHILD_OUTPUT']
    assert get_messages(holder, 'parent') == []
    create_session(holder, first['session_id'] + '-x', 'subagent')  # unrelated row, untouched by retirement
    # The parent's finish retires every live child: it can no longer write and its row is ended.
    mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                            sequence=5, operation='execution.finish', payload={})
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        _child_append(holder, epoch, second, 1, 'late')
    with holder._read_ctx() as conn:
        assert {r[0] for r in conn.execute("SELECT status FROM worker_executions")} == {'terminal'}
    assert get_session(holder, second['session_id'])['end_reason'] == 'agent_close'
    assert get_session(holder, first['session_id'] + '-x')['ended_at'] is None


def test_owner_restart_retires_in_flight_children_instead_of_recovering_them(ledger):
    holder, epoch = ledger
    adopt_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                           adoption_secret='private-fixture')
    assert _reserve(holder, epoch, 1, limits=(1, 1), execution_id='worker')['child']
    with holder._read_ctx() as conn:
        child_eid, child_sid = conn.execute("SELECT execution_id,session_id FROM worker_executions WHERE kind='child'").fetchone()
    assert _reserve(holder, epoch, 2, limits=(1, 1)) == {'refused': 'delegation_capacity'}
    replacement = begin_runtime_epoch(holder, instance_id='restart')
    recover_session_inputs(holder, epoch=replacement)
    with holder._read_ctx() as conn:
        states = dict(conn.execute('SELECT execution_id,status FROM worker_executions').fetchall())
    # The root is the recoverable unknown; its child died with the worker and is never adoptable.
    assert states == {'worker': 'unknown', child_eid: 'terminal'}
    assert get_session(holder, child_sid)['end_reason'] == 'agent_close'
    with pytest.raises(AuthorityRefusal, match='stale_generation|stale_epoch'):
        _child_append(holder, replacement, {'execution_id': child_eid, 'session_id': child_sid}, 1, 'late')


def test_depth_cap_profile_and_an_early_child_finish_settle_the_child_row(ledger):
    holder, epoch = ledger
    child = _reserve(holder, epoch, 1)['child']
    assert get_session(holder, child['session_id'])['profile_name'] == 'coder'
    # The child is already at max_spawn_depth=1: the owner refuses its own child.request.
    nested = partial(reserve_child_execution, epoch=epoch, execution_id=child['execution_id'],
                     identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                     principal='local', limits=(2, 1), refused=None)
    assert apply_worker_receipt(holder, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                                generation=0, sequence=1, operation='child.reserve',
                                payload={'secret': 'nested-secret-00000001'},
                                handlers={'child.reserve': nested}) == {'refused': 'delegation_depth_exceeded'}
    # The child's constructor facts land beside the owner's lineage marker.
    mutate_worker_execution(holder, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                            generation=0, sequence=2, operation='session.create',
                            payload={'source': 'subagent', 'model_config': {'max_iterations': 7}})
    import json
    assert json.loads(get_session(holder, child['session_id'])['model_config']) == {
        '_delegate_from': 'parent', 'max_iterations': 7}
    # An early close finishes the child itself (failed constructor, refused sibling): its row ends
    # then, while the parent is still live, since parent retirement only selects live children.
    mutate_worker_execution(holder, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                            generation=0, sequence=3, operation='execution.finish', payload={})
    assert get_session(holder, child['session_id'])['end_reason'] == 'agent_close'
    assert get_session(holder, 'parent')['ended_at'] is None


@pytest.mark.asyncio
async def test_child_reserve_verifies_the_parent_claim_and_refuses_while_draining(backend, scope_a):
    """The request path, not just the handler: a forged parent secret never reaches the ledger,
    and a draining owner answers with a refusal receipt the parent's journal can move past."""
    import subprocess
    import sys
    from types import SimpleNamespace
    from gateway.session_worker import worker_request
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, 'parent', 'kanban')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)  # noqa: ASYNC220 -- returns at once; the test only needs a live pid
    draining = []

    def admission_open():
        if draining:
            raise AuthorityRefusal('runtime_draining')
    authority = SimpleNamespace(epoch=begin_runtime_epoch(holder, instance_id='owner'), profile_id='p', persistence=store,
                                _require_admission_open=admission_open, wake_after_worker=lambda session_id: None,
                                authorize=lambda actor, ref, capability: None,
                                sessions={}, logical_owner=lambda sid: sid)
    connection = SimpleNamespace(authority=authority, actor=SimpleNamespace(
        subject='human', profile_id='p', capabilities={'session:submit', 'worker:adopt'}))
    scope = {'profile_id': 'p', 'session_id': 'parent', 'execution_id': 'compute', 'generation': 0,
             'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)), 'secret': 'w' * 32}
    try:
        epoch = (await worker_request(connection, None, scope | {'kind': 'compute'}, operation='register'))['owner_epoch']

        def reserve(sequence, **override):
            return worker_request(connection, None, scope | override | {
                'epoch': epoch, 'sequence': sequence, 'operation': 'child.reserve',
                'payload': {'secret': f'child-secret-{sequence:08d}'}}, operation='persist')
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            await reserve(1, secret='f' * 32)
        draining.append(True)
        assert await reserve(1) == {'refused': 'runtime_draining'}
        draining.clear()
        assert 'child' in await reserve(2)
        with holder._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM worker_executions WHERE kind='child'").fetchone()[0] == 1
    finally:
        process.stdin.close()
        process.wait(timeout=5)


def test_a_child_store_closed_before_any_turn_finishes_its_execution_and_ends_its_row(ledger, tmp_path):
    """The constructor-failure / refused-sibling path: the child's store closes having written
    nothing, and that close alone settles the reservation (the parent is still live)."""
    from agent.runtime_session_child import reserve_child_store
    from agent.runtime_session_store import RuntimeSessionStore
    holder, epoch = ledger
    reserve = partial(reserve_child_execution, epoch=epoch, execution_id='worker',
                      identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=(2, 1), refused=None)

    def rpc(method, **params):
        params = {k: params[k] for k in ('epoch', 'execution_id', 'session_id', 'generation', 'sequence', 'operation', 'payload')}
        if params['operation'] == 'child.reserve':
            return apply_worker_receipt(holder, **params, handlers={'child.reserve': reserve})
        return mutate_worker_execution(holder, **params)
    parent = RuntimeSessionStore(rpc, dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0),
                                 tmp_path / 'outbox')
    try:
        child = reserve_child_store(parent)
        child.close()
        with holder._read_ctx() as conn:
            assert conn.execute("SELECT status FROM worker_executions WHERE session_id=?",
                                (child.reserved_session_id,)).fetchone()[0] == 'terminal'
        assert get_session(holder, child.reserved_session_id)['end_reason'] == 'agent_close'
    finally:
        parent.close()
