"""Permit contract: real held gate, writer chain and operation CAS, no provider I/O."""
import uuid
import time
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
        owner.generation = uuid.uuid4().hex
        with pytest.raises(CatalogError):
            facade.settle(dict(reservation_id=admission['reservation_id'], operation_id=op,
                              expected_path_revision=0, result=operation(owner, op)))


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
