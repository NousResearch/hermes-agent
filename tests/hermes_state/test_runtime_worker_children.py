"""Owner-reserved delegated children of a worker execution (#141793): the owner mints every child,
caps them, scopes their writes to their own row, and retires them with their parent."""
import os
from functools import partial

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from gateway.session_worker_child import reserve_child_execution
from hermes_state import SessionDB
import hermes_state_runtime as rt


def _reserve(db, epoch, sequence, *, limits=(2, 1), refused=None, execution_id='worker'):
    handler = partial(reserve_child_execution, epoch=epoch, execution_id=execution_id,
                      identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=limits, refused=refused)
    return rt.apply_worker_receipt(db, epoch=epoch, execution_id=execution_id, session_id='parent', generation=0,
                                   sequence=sequence, operation='child.reserve',
                                   payload={'secret': f'child-secret-{sequence:08d}'}, handlers={'child.reserve': handler})


def _child_append(db, epoch, child, sequence, content, session_id=None):
    return rt.mutate_worker_execution(db, epoch=epoch, execution_id=child['execution_id'],
        session_id=session_id or child['session_id'], generation=0, sequence=sequence,
        operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': content}]})


@pytest.fixture
def ledger(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('parent', source='kanban', profile_name='coder')
    db.create_session('other', source='cli')
    epoch = rt.begin_runtime_epoch(db, instance_id='boot')
    rt.register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                                 kind='compute', adoption_secret='private-fixture')
    yield db, epoch
    db.close()


def test_owner_mints_caps_and_scopes_children_and_retires_them_with_the_parent(ledger):
    db, epoch = ledger
    first = _reserve(db, epoch, 1)['child']
    second = _reserve(db, epoch, 2)['child']
    # Concurrency and draining caps are receipts, so the parent's journal never sticks.
    assert _reserve(db, epoch, 3) == {'refused': 'delegation_capacity'}
    assert _reserve(db, epoch, 4, refused='runtime_draining') == {'refused': 'runtime_draining'}
    row = db.get_session(first['session_id'])
    assert (row['source'], row['parent_session_id']) == ('subagent', 'parent')
    with db._read_ctx() as conn:
        kinds = conn.execute("SELECT kind,status FROM worker_executions WHERE execution_id!='worker'").fetchall()
    assert [tuple(r) for r in kinds] == [('child', 'running')] * 2
    # A child writes its own row only.
    _child_append(db, epoch, first, 1, 'CHILD_OUTPUT')
    with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
        _child_append(db, epoch, first, 2, 'escape', session_id='parent')
    assert [m['content'] for m in db.get_messages(first['session_id'])] == ['CHILD_OUTPUT']
    assert db.get_messages('parent') == []
    db.create_session(first['session_id'] + '-x', source='subagent')  # unrelated row, untouched by retirement
    # The parent's finish retires every live child: it can no longer write and its row is ended.
    rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                               sequence=5, operation='execution.finish', payload={})
    with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
        _child_append(db, epoch, second, 1, 'late')
    with db._read_ctx() as conn:
        assert {r[0] for r in conn.execute("SELECT status FROM worker_executions")} == {'terminal'}
    assert db.get_session(second['session_id'])['end_reason'] == 'agent_close'
    assert db.get_session(first['session_id'] + '-x')['ended_at'] is None


def test_owner_restart_retires_in_flight_children_instead_of_recovering_them(ledger):
    db, epoch = ledger
    rt.adopt_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                              adoption_secret='private-fixture')
    assert _reserve(db, epoch, 1, limits=(1, 1), execution_id='worker')['child']
    with db._read_ctx() as conn:
        child_eid, child_sid = conn.execute("SELECT execution_id,session_id FROM worker_executions WHERE kind='child'").fetchone()
    assert _reserve(db, epoch, 2, limits=(1, 1)) == {'refused': 'delegation_capacity'}
    replacement = rt.begin_runtime_epoch(db, instance_id='restart')
    rt.recover_session_inputs(db, epoch=replacement)
    with db._read_ctx() as conn:
        states = dict(conn.execute('SELECT execution_id,status FROM worker_executions').fetchall())
    # The root is the recoverable unknown; its child died with the worker and is never adoptable.
    assert states == {'worker': 'unknown', child_eid: 'terminal'}
    assert db.get_session(child_sid)['end_reason'] == 'agent_close'
    with pytest.raises(rt.RuntimeStoreError, match='stale_generation|stale_epoch'):
        _child_append(db, replacement, {'execution_id': child_eid, 'session_id': child_sid}, 1, 'late')


def test_depth_cap_profile_and_an_early_child_finish_settle_the_child_row(ledger):
    db, epoch = ledger
    child = _reserve(db, epoch, 1)['child']
    assert db.get_session(child['session_id'])['profile_name'] == 'coder'
    # The child is already at max_spawn_depth=1: the owner refuses its own child.request.
    nested = partial(reserve_child_execution, epoch=epoch, execution_id=child['execution_id'],
                     identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                     principal='local', limits=(2, 1), refused=None)
    assert rt.apply_worker_receipt(db, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                                   generation=0, sequence=1, operation='child.reserve',
                                   payload={'secret': 'nested-secret-00000001'},
                                   handlers={'child.reserve': nested}) == {'refused': 'delegation_depth_exceeded'}
    # The child's constructor facts land beside the owner's lineage marker.
    rt.mutate_worker_execution(db, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                               generation=0, sequence=2, operation='session.create',
                               payload={'source': 'subagent', 'model_config': {'max_iterations': 7}})
    import json
    assert json.loads(db.get_session(child['session_id'])['model_config']) == {
        '_delegate_from': 'parent', 'max_iterations': 7}
    # An early close finishes the child itself (failed constructor, refused sibling): its row ends
    # then, while the parent is still live, since parent retirement only selects live children.
    rt.mutate_worker_execution(db, epoch=epoch, execution_id=child['execution_id'], session_id=child['session_id'],
                               generation=0, sequence=3, operation='execution.finish', payload={})
    assert db.get_session(child['session_id'])['end_reason'] == 'agent_close'
    assert db.get_session('parent')['ended_at'] is None


@pytest.mark.asyncio
async def test_child_reserve_verifies_the_parent_claim_and_refuses_while_draining(tmp_path):
    """The request path, not just the handler: a forged parent secret never reaches the ledger,
    and a draining owner answers with a refusal receipt the parent's journal can move past."""
    import subprocess
    import sys
    from types import SimpleNamespace
    from gateway.session_worker import worker_request
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('parent', source='kanban')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)  # noqa: ASYNC220 -- returns at once; the test only needs a live pid
    draining = []

    def admission_open():
        if draining:
            raise rt.RuntimeStoreError('runtime_draining')
    authority = SimpleNamespace(db=db, epoch=rt.begin_runtime_epoch(db, instance_id='owner'), profile_id='p',
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
        with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
            await reserve(1, secret='f' * 32)
        draining.append(True)
        assert await reserve(1) == {'refused': 'runtime_draining'}
        draining.clear()
        assert 'child' in await reserve(2)
        with db._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM worker_executions WHERE kind='child'").fetchone()[0] == 1
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


def test_a_child_store_closed_before_any_turn_finishes_its_execution_and_ends_its_row(ledger, tmp_path):
    """The constructor-failure / refused-sibling path: the child's store closes having written
    nothing, and that close alone settles the reservation (the parent is still live)."""
    from agent.runtime_session_child import reserve_child_store
    from agent.runtime_session_store import RuntimeSessionStore
    db, epoch = ledger
    reserve = partial(reserve_child_execution, epoch=epoch, execution_id='worker',
                      identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=(2, 1), refused=None)

    def rpc(method, **params):
        params = {k: params[k] for k in ('epoch', 'execution_id', 'session_id', 'generation', 'sequence', 'operation', 'payload')}
        if params['operation'] == 'child.reserve':
            return rt.apply_worker_receipt(db, **params, handlers={'child.reserve': reserve})
        return rt.mutate_worker_execution(db, **params)
    parent = RuntimeSessionStore(rpc, dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0),
                                 tmp_path / 'outbox')
    try:
        child = reserve_child_store(parent)
        child.close()
        with db._read_ctx() as conn:
            assert conn.execute("SELECT status FROM worker_executions WHERE session_id=?",
                                (child.reserved_session_id,)).fetchone()[0] == 'terminal'
        assert db.get_session(child.reserved_session_id)['end_reason'] == 'agent_close'
    finally:
        parent.close()


def test_fresh_child_reservation_refuses_existing_local_history(ledger):
    db, epoch = ledger
    db.append_message('other', 'user', 'preserve unrelated history')
    before = [tuple(row) for row in db._conn.execute("SELECT * FROM sessions WHERE id='other'")]
    messages = [tuple(row) for row in db._conn.execute("SELECT * FROM messages WHERE session_id='other'")]
    handler = partial(reserve_child_execution, epoch=epoch, execution_id='worker',
                      identity={'profile_id': 'home', 'pid': os.getpid(), 'birth': worker_birth(psutil.Process())},
                      principal='local', limits=(2, 1), refused=None, reserved_session_id='other')
    args = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                sequence=1, operation='child.reserve', payload={'secret': 'child-secret-00000001'},
                handlers={'child.reserve': handler}, lineage_ids=('other',))
    result = rt.apply_worker_receipt(db, **args)
    assert result == {'refused': 'session_id_collision'}
    assert rt.apply_worker_receipt(db, **args) == result
    assert [tuple(row) for row in db._conn.execute("SELECT * FROM sessions WHERE id='other'")] == before
    assert [tuple(row) for row in db._conn.execute("SELECT * FROM messages WHERE session_id='other'")] == messages
    assert db._conn.execute("SELECT COUNT(*) FROM worker_executions WHERE kind='child'").fetchone()[0] == 0


@pytest.fixture
def managed_finish(tmp_path, monkeypatch):
    from pathlib import Path
    from functools import partial
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    profile = home / 'profiles' / 'science'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    local, sibling = SessionDB(home / 'state.db'), SessionDB(profile / 'state.db')
    local.create_session('parent', 'cli')
    epoch = rt.begin_runtime_epoch(local, instance_id='finish-fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0)
    rt.register_worker_execution(local, **scope, kind='compute', adoption_secret='fixture')
    def reserve(sid, seq):
        handler = partial(reserve_child_execution, epoch=epoch, execution_id='worker',
                          identity={'profile_id': 'default', 'pid': 1, 'birth': 'fixture'},
                          principal='local', limits=(4, 1), refused=None, reserved_session_id=sid)
        return rt.apply_worker_receipt(local, **scope, sequence=seq, operation='child.reserve',
            payload={'secret': f'child-secret-{seq:08d}'}, handlers={'child.reserve': handler}, lineage_ids=(sid,))
    yield local, sibling, scope, reserve
    local.close()
    sibling.close()


def _finish_snapshot(db):
    return {table: [tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('sessions', 'messages', 'worker_executions', 'worker_receipts')}


def test_finish_refuses_foreign_owned_active_child(managed_finish):
    from hermes_state_lineage import SessionLineageAmbiguousError
    local, sibling, scope, reserve = managed_finish
    assert reserve('child', 1)['child']['session_id'] == 'child'
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('child','cli',1)")
    sibling._conn.commit()
    before = _finish_snapshot(local), _finish_snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        rt.mutate_worker_execution(local, **scope, sequence=2, operation='execution.finish', payload={})
    assert (_finish_snapshot(local), _finish_snapshot(sibling)) == before


def test_finish_multi_child_and_exact_receipt_replay(managed_finish):
    local, sibling, scope, reserve = managed_finish
    reserve('one', 1)
    reserve('two', 2)
    args = dict(scope, sequence=3, operation='execution.finish', payload={})
    assert rt.mutate_worker_execution(local, **args) == {'status': 'terminal'}
    assert all(local.get_session(sid)['end_reason'] == 'agent_close' for sid in ('one', 'two'))
    before = _finish_snapshot(local), _finish_snapshot(sibling)
    # A foreign copy appearing after the committed result cannot invalidate exact replay.
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('one','cli',1)")
    sibling._conn.commit()
    after_external = _finish_snapshot(local), _finish_snapshot(sibling)
    assert rt.mutate_worker_execution(local, **args) == {'status': 'terminal'}
    assert (_finish_snapshot(local), _finish_snapshot(sibling)) == after_external
    assert before[0] == after_external[0]


def test_finish_rolls_back_all_children_and_receipt_on_failure(managed_finish):
    import sqlite3
    local, sibling, scope, reserve = managed_finish
    reserve('one', 1)
    reserve('two', 2)
    local._conn.execute("CREATE TRIGGER fail_child_end BEFORE UPDATE OF ended_at ON sessions "
                        "WHEN NEW.id='two' BEGIN SELECT RAISE(ABORT, 'child end fault'); END")
    local._conn.commit()
    before = _finish_snapshot(local), _finish_snapshot(sibling)
    with pytest.raises(sqlite3.IntegrityError, match='child end fault'):
        rt.mutate_worker_execution(local, **scope, sequence=3, operation='execution.finish', payload={})
    assert (_finish_snapshot(local), _finish_snapshot(sibling)) == before


def test_finish_rechecks_newly_active_descendant_before_any_write(managed_finish, monkeypatch):
    from hermes_state_lineage import SessionLineageAmbiguousError
    local, sibling, scope, reserve = managed_finish
    reserve('one', 1)
    second = reserve('two', 2)['child']['execution_id']
    local._conn.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id=?", (second,))
    local._conn.commit()
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('two','cli',1)")
    sibling._conn.commit()
    original = local._execute_write
    calls = []
    def make_late_active(fn, **kwargs):
        if not calls:
            local._conn.execute("UPDATE worker_executions SET status='running' WHERE execution_id=?", (second,))
            local._conn.commit()
        calls.append(True)
        return original(fn, **kwargs)
    monkeypatch.setattr(local, '_execute_write', make_late_active)
    with pytest.raises(SessionLineageAmbiguousError):
        rt.mutate_worker_execution(local, **scope, sequence=3, operation='execution.finish', payload={})
    assert local.get_session('one')['ended_at'] is None
    assert local.get_session('two')['ended_at'] is None
    assert local._conn.execute("SELECT count(*) FROM worker_receipts WHERE execution_id='worker'").fetchone()[0] == 2
    assert local._conn.execute("SELECT status FROM worker_executions WHERE execution_id='worker'").fetchone()[0] == 'running'


@pytest.fixture
def admitted_finish(tmp_path, monkeypatch):
    from pathlib import Path
    from functools import partial
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    profile = home / 'profiles' / 'science'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    local, sibling = SessionDB(home / 'state.db'), SessionDB(profile / 'state.db')
    local.create_session('parent', 'cli')
    epoch = rt.begin_runtime_epoch(local, instance_id='admission-owner')
    rt.admit_session_input(local, epoch=epoch, principal_id='owner', session_id='parent',
                           request_id='input', payload={'text': 'hello'})
    admission = rt.claim_session_input(local, epoch=epoch, session_id='parent')
    execution = 'admission-worker:' + admission['admission_id']
    scope = dict(epoch=epoch, execution_id=execution, session_id='parent', generation=admission['generation'])
    rt.register_worker_execution(local, **scope, kind='compute', adoption_secret='fixture')
    handler = partial(reserve_child_execution, epoch=epoch, execution_id=execution,
                      identity={'profile_id': 'default', 'pid': 1, 'birth': 'fixture'},
                      principal='local', limits=(4, 1), refused=None, reserved_session_id='admitted-child')
    assert rt.apply_worker_receipt(local, **scope, sequence=1, operation='child.reserve',
        payload={'secret': 'child-secret-00000001'}, handlers={'child.reserve': handler},
        lineage_ids=('admitted-child',))['child']['session_id'] == 'admitted-child'
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('admitted-child','cli',1)")
    sibling._conn.commit()
    yield local, sibling, scope, admission
    local.close()
    sibling.close()


def _full_finish_snapshot(db):
    return {table: [tuple(row) for row in db._conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')]
            for table in ('sessions', 'messages', 'worker_executions', 'worker_receipts', 'session_admissions', 'state_meta')}


@pytest.mark.parametrize('action', ['settle', 'resolve_unknown', 'lose'])
def test_admission_terminal_paths_fence_descendant(admitted_finish, action):
    from types import SimpleNamespace
    from hermes_state_lineage import SessionLineageAmbiguousError
    from gateway.session_worker_reservation import lose_admission_worker
    local, sibling, scope, admission = admitted_finish
    if action == 'resolve_unknown':
        local._conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission['admission_id'],))
        local._conn.commit()
    before = _full_finish_snapshot(local), _full_finish_snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        if action == 'settle':
            rt.settle_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                    generation=scope['generation'], outcome='completed')
        elif action == 'resolve_unknown':
            rt.resolve_unknown_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                             generation=scope['generation'])
        else:
            lose_admission_worker(SimpleNamespace(db=local), admission, scope)
    assert (_full_finish_snapshot(local), _full_finish_snapshot(sibling)) == before


def test_restart_recovery_fences_foreign_duplicate_stale_child(managed_finish):
    from hermes_state_lineage import SessionLineageAmbiguousError
    local, sibling, _scope, reserve = managed_finish
    reserve('restart-child', 1)
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('restart-child','cli',1)")
    sibling._conn.commit()
    next_epoch = rt.begin_runtime_epoch(local, instance_id='replacement')
    before = _full_finish_snapshot(local), _full_finish_snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        rt.recover_session_inputs(local, epoch=next_epoch)
    assert (_full_finish_snapshot(local), _full_finish_snapshot(sibling)) == before


@pytest.mark.parametrize('foreign_owner', [True, False])
def test_reset_fences_foreign_duplicate_orphan_descendant(tmp_path, monkeypatch, foreign_owner):
    from pathlib import Path
    from functools import partial
    from hermes_state_lineage import SessionLineageAmbiguousError
    from tests.gateway.test_unknown_resolution_physical_worker import _local_reset
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    profile = home / 'profiles' / 'science'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    with SessionDB(home / 'state.db') as local, SessionDB(profile / 'state.db') as sibling:
        epoch = rt.begin_runtime_epoch(local, instance_id='reset-owner')
        logical = _local_reset(local, epoch, 'logical', 'physical')
        generation = local.get_session(logical)['runtime_generation']
        scope = dict(epoch=epoch, execution_id='worker', session_id=logical, generation=generation)
        rt.register_worker_execution(local, **scope, kind='compute', adoption_secret='fixture')
        handler = partial(reserve_child_execution, epoch=epoch, execution_id='worker',
                          identity={'profile_id': 'default', 'pid': 1, 'birth': 'fixture'},
                          principal='local', limits=(4, 1), refused=None, reserved_session_id='reset-child')
        rt.apply_worker_receipt(local, **scope, sequence=1, operation='child.reserve',
            payload={'secret': 'child-secret-00000001'}, handlers={'child.reserve': handler}, lineage_ids=('reset-child',))
        local._conn.execute("UPDATE worker_executions SET status='unknown' WHERE execution_id='worker'")
        local._conn.commit()
        if foreign_owner:
            sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('reset-child','cli',1)")
            sibling._conn.commit()
        before = _full_finish_snapshot(local), _full_finish_snapshot(sibling)
        row = local.get_session(logical)
        args = dict(epoch=epoch, principal_id='human', session_id=logical, request_id='reset-second',
                    expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                    operation='reset', payload={})
        if foreign_owner:
            with pytest.raises(SessionLineageAmbiguousError):
                rt.mutate_runtime_session(local, **args)
            assert (_full_finish_snapshot(local), _full_finish_snapshot(sibling)) == before
        else:
            result = rt.mutate_runtime_session(local, **args)
            assert result['operation'] == 'reset'
            assert local.get_session('reset-child')['end_reason'] == 'agent_close'
            assert rt.mutate_runtime_session(local, **args) == result
            assert _full_finish_snapshot(sibling) == before[1]


@pytest.mark.parametrize('action', ['finish', 'recover'])
def test_terminal_paths_allow_already_ended_local_child(managed_finish, action):
    local, sibling, scope, reserve = managed_finish
    reserve('ended-child', 1)
    local._conn.execute("UPDATE sessions SET ended_at=10,end_reason='adopted_by_profile' WHERE id='ended-child'")
    local._conn.commit()
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('ended-child','cli',1)")
    sibling._conn.commit()
    before = tuple(local._conn.execute("SELECT * FROM sessions WHERE id='ended-child'").fetchone())
    foreign_before = _full_finish_snapshot(sibling)
    if action == 'finish':
        assert rt.mutate_worker_execution(local, **scope, sequence=2, operation='execution.finish', payload={}) == {'status': 'terminal'}
    else:
        next_epoch = rt.begin_runtime_epoch(local, instance_id='replacement')
        assert rt.recover_session_inputs(local, epoch=next_epoch) == 0
    assert tuple(local._conn.execute("SELECT * FROM sessions WHERE id='ended-child'").fetchone()) == before
    assert _full_finish_snapshot(sibling) == foreign_before
    assert local._conn.execute("SELECT status FROM worker_executions WHERE session_id='ended-child'").fetchone()[0] == 'terminal'


@pytest.mark.parametrize('action', ['settle', 'resolve_unknown', 'lose'])
def test_admission_terminal_paths_allow_already_ended_local_child(admitted_finish, action):
    from types import SimpleNamespace
    from gateway.session_worker_reservation import lose_admission_worker
    local, sibling, scope, admission = admitted_finish
    local._conn.execute("UPDATE sessions SET ended_at=10,end_reason='adopted_by_profile' WHERE id='admitted-child'")
    if action == 'resolve_unknown':
        local._conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission['admission_id'],))
    local._conn.commit()
    before = tuple(local._conn.execute("SELECT * FROM sessions WHERE id='admitted-child'").fetchone())
    foreign_before = _full_finish_snapshot(sibling)
    if action == 'settle':
        result = rt.settle_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                         generation=scope['generation'], outcome='completed')
        assert result['status'] == 'terminal'
    elif action == 'resolve_unknown':
        result = rt.resolve_unknown_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                                  generation=scope['generation'])
        assert result['status'] == 'terminal'
    else:
        lose_admission_worker(SimpleNamespace(db=local), admission, scope)
    assert tuple(local._conn.execute("SELECT * FROM sessions WHERE id='admitted-child'").fetchone()) == before
    assert _full_finish_snapshot(sibling) == foreign_before
    assert local._conn.execute("SELECT status FROM worker_executions WHERE session_id='admitted-child'").fetchone()[0] == 'terminal'


@pytest.mark.parametrize('action', ['settle', 'resolve_unknown', 'lose'])
def test_admission_terminal_paths_retire_owned_live_child(admitted_finish, action):
    from types import SimpleNamespace
    from gateway.session_worker_reservation import lose_admission_worker
    local, sibling, scope, admission = admitted_finish
    sibling._conn.execute("DELETE FROM sessions WHERE id='admitted-child'")
    sibling._conn.commit()
    if action == 'resolve_unknown':
        local._conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission['admission_id'],))
        local._conn.commit()
    if action == 'settle':
        result = rt.settle_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                         generation=scope['generation'], outcome='completed')
        assert result['status'] == 'terminal'
    elif action == 'resolve_unknown':
        result = rt.resolve_unknown_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                                  generation=scope['generation'])
        assert result['status'] == 'terminal'
        # Replay of the committed resolution does not rediscover the retired child.
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('admitted-child','cli',1)")
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('parent','cli',1)")
        sibling._conn.commit()
        before = _full_finish_snapshot(local), _full_finish_snapshot(sibling)
        assert rt.resolve_unknown_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                                generation=scope['generation']) == result
        assert (_full_finish_snapshot(local), _full_finish_snapshot(sibling)) == before
    else:
        lose_admission_worker(SimpleNamespace(db=local), admission, scope)
    assert local.get_session('admitted-child')['end_reason'] == 'agent_close'
    assert local._conn.execute("SELECT status FROM worker_executions WHERE session_id='admitted-child'").fetchone()[0] == 'terminal'


@pytest.mark.parametrize('action', ['settle', 'resolve_unknown', 'lose'])
def test_terminal_stale_generation_precedes_foreign_descendant_discovery(admitted_finish, action):
    from types import SimpleNamespace
    from gateway.session_worker_reservation import lose_admission_worker
    local, sibling, scope, admission = admitted_finish
    if action == 'resolve_unknown':
        local._conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission['admission_id'],))
        local._conn.commit()
    before = _full_finish_snapshot(local), _full_finish_snapshot(sibling)
    with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
        if action == 'settle':
            rt.settle_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                    generation=scope['generation'] + 1, outcome='completed')
        elif action == 'resolve_unknown':
            rt.resolve_unknown_session_input(local, epoch=scope['epoch'], admission_id=admission['admission_id'],
                                             generation=scope['generation'] + 1)
        else:
            lose_admission_worker(SimpleNamespace(db=local), admission, scope | {'generation': scope['generation'] + 1})
    assert (_full_finish_snapshot(local), _full_finish_snapshot(sibling)) == before


def test_restart_recovery_retires_owned_live_child(managed_finish):
    local, _sibling, _scope, reserve = managed_finish
    reserve('restart-child', 1)
    next_epoch = rt.begin_runtime_epoch(local, instance_id='replacement')
    assert rt.recover_session_inputs(local, epoch=next_epoch) == 0
    assert local.get_session('restart-child')['end_reason'] == 'agent_close'
    assert local._conn.execute("SELECT status FROM worker_executions WHERE execution_id='worker'").fetchone()[0] == 'unknown'
