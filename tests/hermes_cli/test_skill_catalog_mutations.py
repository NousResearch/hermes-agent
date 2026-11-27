"""Permit contract: real held gate, writer chain and operation CAS, no provider I/O."""
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from hermes_cli.skill_catalog.validation import CatalogError
from hermes_cli.skill_catalog.locks import FileLease
from tests.hermes_cli.test_skill_catalog_projection import started, review


def operation(owner, op, **changes):
    result = dict(operation_id=op, identifier=owner.source_id + ':item', action='install',
                  state='completed', outcome='installed', managed_state='installed',
                  message='', installation_revision='revision-1')
    result.update(changes)
    return result


@contextmanager
def admitted(home):
    from hermes_cli.skill_catalog.mutations import Mutations
    with started(home) as (projection, owner):
        facade = Mutations(owner, projection)
        plan = review(owner)
        admission = projection.reserve(owner, plan)
        op = uuid.uuid4().hex
        facade.bind(dict({k: v for k, v in admission.items() if k != 'generation'}, operation_id=op))
        try:
            yield projection, owner, facade, admission, op
        finally:
            facade.close()


@pytest.mark.parametrize('phase', ['queued', 'acquired'])
@pytest.mark.parametrize('method', ['acquire', 'recovery_acquire'])
def test_timed_out_acquire_never_orphans_permit(tmp_path, monkeypatch, phase, method):
    with admitted(tmp_path) as (p, _owner, facade, admission, op):
        entered, resume = threading.Event(), threading.Event()
        original = getattr(facade, '_do_' + method)
        original_wait = threading.Event.wait

        def delayed(payload):
            if phase == 'queued':
                entered.set()
                assert resume.wait(10)
            result = original(payload)
            if phase == 'acquired':
                entered.set()
                assert resume.wait(10)
            return result

        def bounded_wait(event, timeout=None):
            # Keep the real command queue and worker, accelerate only caller wait.
            return original_wait(event, 0.1 if timeout == 3.5 else timeout)

        monkeypatch.setattr(facade, '_do_' + method, delayed)
        monkeypatch.setattr(threading.Event, 'wait', bounded_wait)
        payload = dict(admission=admission, operation_id=op)
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(getattr(facade, method), payload)
            try:
                assert entered.wait(5)
                with pytest.raises(CatalogError) as caught:
                    request.result(5)
                assert caught.value.code == 'outcome_unknown'
            finally:
                resume.set()
        # FIFO barrier: all late acquisition/cleanup has finished by this reply.
        facade.settlement(op)
        try:
            assert facade._permit is None, 'timed-out caller left an orphan permit'
            monkeypatch.setattr(facade, '_do_' + method, original)
            next_permit = facade.acquire(payload)
            facade.release(dict(permit_id=next_permit['permit_id']))
        finally:
            if facade._permit is not None:
                facade.release(dict(permit_id=facade._permit['id']))
        assert p.read()['reservations'][admission['reservation_id']]['state'] in ('bound', 'unknown')


@pytest.mark.parametrize('method', ['acquire', 'recovery_acquire'])
def test_cancelled_queued_acquire_never_reaches_projection(tmp_path, monkeypatch, method):
    with admitted(tmp_path) as (_p, _owner, facade, admission, op):
        entered, resume = threading.Event(), threading.Event()
        original_settlement = facade._do_settlement
        original_acquire = getattr(facade, '_do_' + method)
        original_wait = threading.Event.wait
        acquisitions = []

        def blocking(payload):
            entered.set()
            assert resume.wait(10)
            return original_settlement(payload)

        def acquire(payload):
            acquisitions.append(payload)
            return original_acquire(payload)

        monkeypatch.setattr(facade, '_do_settlement', blocking)
        monkeypatch.setattr(facade, '_do_' + method, acquire)
        payload = dict(admission=admission, operation_id=op)
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocker = pool.submit(facade.settlement, op)
            try:
                assert entered.wait(5)
                monkeypatch.setattr(threading.Event, 'wait', lambda event, timeout=None:
                    original_wait(event, 0.1 if timeout == 3.5 else timeout))
                with pytest.raises(CatalogError) as caught:
                    getattr(facade, method)(payload)
                assert caught.value.code == 'outcome_unknown'
            finally:
                resume.set()
            blocker.result(5)
        monkeypatch.setattr(facade, '_do_settlement', original_settlement)
        facade.settlement(op)
        assert acquisitions == []
        assert facade._permit is None
        monkeypatch.setattr(facade, '_do_' + method, original_acquire)
        permit = facade.acquire(payload)
        facade.release(dict(permit_id=permit['permit_id']))


@pytest.mark.parametrize('method', ['acquire', 'recovery_acquire'])
def test_cancel_cleanup_publication_failure_keeps_worker_alive(tmp_path, monkeypatch, method):
    from hermes_cli.skill_catalog import projection as storage
    from tests.waiting import wait_until

    with admitted(tmp_path) as (p, owner, facade, admission, op):
        # Bind is durable before acquire. A failed unknown-state publication must
        # retain that unresolved row byte-for-byte, never invent a terminal result.
        before = (p.directory / 'claims.json').read_bytes()
        entered, resume = threading.Event(), threading.Event()
        original = getattr(facade, '_do_' + method)
        original_wait, original_put = threading.Event.wait, facade._queue.put_nowait
        stage = storage._stage
        commands = []

        def record(command):
            commands.append(command)
            return original_put(command)

        def delayed(payload):
            result = original(payload)
            entered.set()
            assert resume.wait(10)
            return result

        def failed_stage(*args):
            raise OSError('injected disk publication failure')

        monkeypatch.setattr(facade._queue, 'put_nowait', record)
        monkeypatch.setattr(facade, '_do_' + method, delayed)
        monkeypatch.setattr(threading.Event, 'wait', lambda event, timeout=None:
            original_wait(event, 0.1 if timeout == 3.5 else timeout))
        payload = dict(admission=admission, operation_id=op)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(getattr(facade, method), payload)
                assert entered.wait(5)
                with pytest.raises(CatalogError) as caught:
                    pending.result(5)
                assert caught.value.code == 'outcome_unknown'
                command, = commands
                # Only the abandoned caller gets the accelerated wait. Restored
                # FIFO/reconciliation commands retain their real production budget.
                monkeypatch.setattr(threading.Event, 'wait', original_wait)
                monkeypatch.setattr(storage, '_stage', failed_stage)
                resume.set()
                assert wait_until(lambda: command.done.is_set() or not facade._thread.is_alive(), timeout=5)
                assert facade._thread.is_alive(), 'cancel cleanup publication error killed permit owner'
                assert command.done.is_set(), 'abandoned command was not notified'
                assert isinstance(command.error, CatalogError)
                assert command.error.code == 'ownership_state_unavailable'
                assert facade._permit is None
                assert (p.directory / 'claims.json').read_bytes() == before
                assert 'Catalog abandoned acquire cleanup failed: CatalogError' in (
                    tmp_path / 'logs/skill-catalog.log').read_text()
            monkeypatch.setattr(storage, '_stage', stage)
            monkeypatch.setattr(facade, '_do_' + method, original)
            assert p.read()['reservations'][admission['reservation_id']]['state'] == 'bound'
            assert facade.settlement(op) == 'reconciling'  # real next FIFO command
            acquired = getattr(facade, method)(payload)
            assert facade.validate(dict(permit_id=acquired['permit_id']))
            facade._send('release', dict(permit_id=acquired['permit_id']))
            result = operation(owner, op, state='failed', outcome='refused',
                               managed_state='not_installed', installation_revision=None)
            owner.provider = type('Provider', (), {'operation': lambda self, request: dict(ok=True, data=result)})()
            owner.ready = True
            facade.reconcile(operation_id=op)
            assert facade.settlement(op) == 'settled'
            assert p.read()['reservations'] == {}
        finally:
            resume.set()
            monkeypatch.setattr(storage, '_stage', stage)
            # The baseline dies here. Avoid masking its behavioral failure at teardown.
            if not facade._thread.is_alive():
                facade._closed = True


def test_commit_derives_target_holds_both_locks_and_idempotently_settles(tmp_path):
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        with facade.commit(admission, op) as commit:
            assert commit.target_relative_path == 'new'
            assert commit.expected_installation_revision is None
            with pytest.raises(CatalogError):
                FileLease(p.directory / 'mutation.lock', deadline=time.monotonic()).acquire()
            target = tmp_path / 'skills/new'
            target.mkdir(parents=True)
            (target / 'SKILL.md').write_text('approved')
            result = operation(owner, op)
            first = commit.settle(result)
            assert commit.settle(result) == first
        doc = p.read()
        assert doc['targets']['new']['installation_revision'] == 'revision-1'
        assert not doc['reservations']
        assert facade.settle(dict(reservation_id=admission['reservation_id'], operation_id=op,
                                  expected_path_revision=0, result=result)) == first
        assert p.read() == doc
        with pytest.raises(CatalogError):
            facade.settle(dict(reservation_id=admission['reservation_id'], operation_id=op,
                              expected_path_revision=0, result=dict(result, message='changed')))


@pytest.mark.parametrize('state', ['completed', 'pending'])
def test_permit_cannot_settle_another_bound_operation(tmp_path, state):
    from hermes_cli.skill_catalog.mutations import Mutations
    with started(tmp_path) as (p, owner):
        facade = Mutations(owner, p)
        a = p.reserve(owner, review(owner, target='a'))
        b = p.reserve(owner, review(owner, target='b'))
        opa, opb = uuid.uuid4().hex, uuid.uuid4().hex
        for admission, op in ((a, opa), (b, opb)):
            facade.bind({k: v for k, v in admission.items() if k != 'generation'}
                        | {'operation_id': op})
        permit = facade.acquire(dict(admission=a, operation_id=opa))
        (tmp_path / 'skills/b').mkdir(parents=True)
        before = (p.directory / 'claims.json').read_bytes()
        try:
            with pytest.raises(CatalogError) as caught:
                facade.settle(dict(permit_id=permit['permit_id'], reservation_id=b['reservation_id'],
                    operation_id=opb, expected_path_revision=0,
                    result=operation(owner, opb, state=state,
                                     outcome=None if state == 'pending' else 'installed')))
            assert caught.value.code == 'ownership_conflict'
            assert (p.directory / 'claims.json').read_bytes() == before
        finally:
            facade.release(dict(permit_id=permit['permit_id']))
            facade.close()
        doc = p.read()
        assert doc['targets'] == {} and doc['path_revisions'] == {}
        assert doc['reservations'][a['reservation_id']]['state'] == 'unknown'
        assert doc['reservations'][b['reservation_id']]['state'] == 'bound'
        assert all(row['local_result'] is None for row in doc['reservations'].values())


def test_pending_local_settlement_then_terminal_outside_hold(tmp_path):
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        with facade.commit(admission, op) as commit:
            (tmp_path / 'skills/new').mkdir(parents=True)
            commit.settle(operation(owner, op, state='pending', outcome=None))
        doc = p.read()
        assert doc['targets']['new']['installation_revision'] == 'revision-1'
        assert doc['reservations'][admission['reservation_id']]['operation_id'] == op
        assert facade.settlement(op) == 'reconciling'
        result = operation(owner, op)
        facade.settle(dict(reservation_id=admission['reservation_id'], operation_id=op,
                          expected_path_revision=0, result=result))
        assert facade.settlement(op) == 'settled'
        assert not p.read()['reservations']


def test_exception_leaves_unknown_and_late_stale_generation_cannot_settle(tmp_path):
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        with pytest.raises(RuntimeError):
            with facade.commit(admission, op):
                raise RuntimeError('before effect')
        assert p.read()['reservations'][admission['reservation_id']]['state'] == 'unknown'
        (tmp_path / 'skills/new').mkdir(parents=True)
        payload = dict(reservation_id=admission['reservation_id'], operation_id=op,
                       expected_path_revision=0, result=operation(owner, op))
        revision = facade.settle(payload)
        assert facade.settle(payload) == revision
        before = (p.directory / 'claims.json').read_bytes()
        # Cached exact replay has no other row-generation or target-prestate fence.
        owner.generation = uuid.uuid4().hex
        with pytest.raises(CatalogError) as caught:
            facade.settle(payload)
        assert caught.value.code == 'ownership_conflict'
        assert (p.directory / 'claims.json').read_bytes() == before


def test_terminal_settlement_rejects_only_payload_expected_revision(tmp_path):
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        (tmp_path / 'skills/new').mkdir(parents=True)
        payload = dict(reservation_id=admission['reservation_id'], operation_id=op,
                       expected_path_revision=0, result=operation(owner, op))
        before = (p.directory / 'claims.json').read_bytes()
        with pytest.raises(CatalogError) as caught:
            facade.settle(dict(payload, expected_path_revision=1))
        assert caught.value.code == 'ownership_conflict'
        assert (p.directory / 'claims.json').read_bytes() == before
        # Same pre-state, correct CAS: reaches publication without a permit fence.
        assert facade.settle(payload) == 1
        assert p.read()['targets']['new']['installation_revision'] == 'revision-1'


def test_prestate_recheck_and_revocation_prevent_writes(tmp_path):
    with admitted(tmp_path) as (_p, owner, facade, admission, op):
        (tmp_path / 'skills/new').mkdir(parents=True)
        with pytest.raises(CatalogError) as caught:
            with facade.commit(admission, op):
                pytest.fail('target appeared after reservation')
        assert caught.value.code == 'review_changed'
        (tmp_path / 'skills/new').rmdir()
        owner.authorized = lambda: False
        with pytest.raises(CatalogError):
            with facade.commit(admission, op):
                pytest.fail('consent revoked')


def test_terminal_journal_failed_settlement_keeps_unknown_until_read_only_reconcile(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import projection as storage
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        result = operation(owner, op)
        # The provider's durable terminal journal is read through the ordinary worker.
        owner.provider = type('Provider', (), {'operation': lambda self, request: {'ok': True, 'data': result}})()
        owner.ready = True
        original = storage._stage
        owner.provider.operation = lambda request: (_ for _ in ()).throw(RuntimeError('offline'))
        def fail_stage(*args):
            raise OSError('disk failure')
        with pytest.raises(CatalogError):
            with facade.commit(admission, op) as commit:
                (tmp_path / 'skills/new').mkdir(parents=True)
                monkeypatch.setattr(storage, '_stage', fail_stage)
                try:
                    commit.settle(result)
                finally:
                    monkeypatch.setattr(storage, '_stage', original)
        doc = p.read()
        assert doc['reservations'][admission['reservation_id']]['state'] == 'unknown'
        assert 'new' not in doc['targets']
        owner.provider.operation = lambda request: {'ok': True, 'data': result}
        facade.reconcile(operation_id=op)
        doc = p.read()
        assert not doc['reservations']
        assert doc['targets']['new']['installation_revision'] == 'revision-1'
        assert (tmp_path / 'skills/new').is_dir()


def test_context_facade_native_no_live_and_exact_binding(tmp_path):
    from hermes_cli.plugins_context import PluginContext
    from hermes_cli.plugins_manifest import PluginManifest
    ctx = PluginContext(PluginManifest(name='missing', version='1'), manager=None)
    assert ctx.skill_catalog_mutations.admit_native({}, {}) is None


@pytest.mark.parametrize('action', ['install', 'uninstall'])
def test_terminal_journal_after_pending_save_failure_is_reconciled(tmp_path, monkeypatch, action):
    from hermes_cli.skill_catalog import projection as storage
    with admitted(tmp_path) as (p, owner, facade, admission, op):
        if action == 'uninstall':
            with p.transaction() as doc:
                doc['reservations'][admission['reservation_id']]['action'] = 'uninstall'
        pending = operation(owner, op, action=action, state='pending', outcome=None,
                            managed_state='removed' if action == 'uninstall' else 'installed',
                            installation_revision=None if action == 'uninstall' else 'revision-1')
        result = dict(pending, state='completed', outcome='uninstalled' if action == 'uninstall' else 'installed')
        owner.provider = type('Provider', (), {'operation': lambda self, request: {'ok': True, 'data': pending}})()
        owner.ready = True
        with facade.commit(admission, op) as commit:
            if action == 'install':
                (tmp_path / 'skills/new').mkdir(parents=True)
            commit.settle(pending)
        owner.provider.operation = lambda request: {'ok': True, 'data': result}
        payload = dict(reservation_id=admission['reservation_id'], operation_id=op,
                       expected_path_revision=0, result=result)
        def failure(*args):
            raise OSError('atomic replace failed')
        with monkeypatch.context() as scoped:
            scoped.setattr(storage, '_stage', failure)
            with pytest.raises(CatalogError):
                facade.settle(payload)
        assert facade.settlement(op) == 'reconciling'
        facade.reconcile(operation_id=op)
        assert facade.settlement(op) == 'settled'
        assert not p.read()['reservations']
        assert ('new' in p.read()['targets']) == (action == 'install')
