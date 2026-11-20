"""Deferred global applications retain an original-request repair link outside the consumed pick slot."""
import json

import pytest

from persistence import FenceStale
from persistence.authority import runtime as rt
from tests.contracts.authority.test_mutation_commands_contract import _commit, _local


def _defer(store, epoch, sid, request_id='original', model='picked'):
    payload = {'model': model, 'scope': 'global'}
    row = store.sessions.get(sid)
    call = dict(principal_id='p', session_id=sid, request_id=request_id, command=rt.parse_mutation('model', payload),
                expected_revision=row.runtime_revision, expected_generation=row.runtime_generation)
    key, digest = rt.mutation_receipt_key('p', sid, request_id, operation='model', payload=payload,
                                          expected_revision=row.runtime_revision, expected_generation=row.runtime_generation)
    accepted = store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id=sid,
                                              pick=rt.DeferredPick('model', 'p', request_id, payload))
    return call, accepted


def _apply(store, epoch, sid, model, request_id):
    policy = {'model': model, 'cwd': '/w', 'config_json': json.dumps({'model': {
        'default': model, 'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_mode': 'responses'}})}
    return _commit(store, epoch, sid, rt.ModelPick(model, scope='global'), {'policy': policy},
                   request_id='deferred-pick:' + request_id, expected_generation=store.sessions.get(sid).runtime_generation)


def test_application_link_survives_local_picks_and_reopen_and_completes_only_its_generation(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id='deferred-owner')
    sid = _local(store)
    original, accepted = _defer(store, epoch, sid)
    assert store.runtime.deferred_global_model_result(sid) is None
    applied = _apply(store, epoch, sid, 'picked', 'original')
    assert store.runtime.deferred_pick(sid) is None
    identity = rt.PinnedIdentity('p', 'original')
    assert store.runtime.deferred_global_model_result(sid, request=identity) == applied.result
    for other in (rt.PinnedIdentity('other', 'original'), rt.PinnedIdentity('p', 'deferred-pick:original')):
        assert store.runtime.deferred_global_model_result(sid, request=other) is None
    # Neither immutable receipt is changed by querying the durable relationship.
    assert store.runtime.prepare_mutation(epoch, **original).result == accepted
    assert store.runtime.mutate(epoch, **original, owner=None).result == accepted
    _commit(store, epoch, sid, rt.ModelPick('local'), {'policy': {'model': 'local', 'cwd': '/w',
            'config_json': '{"model":{"provider":"custom"}}'}}, request_id='local', expected_generation=1)
    if backend.capabilities.reopen_durability:
        backend.reopen()
    assert store.runtime.deferred_global_model_result(sid, request=identity) == applied.result
    assert store.runtime.is_latest_global_model_selection(session_id=sid, generation=1)
    store.runtime.finish_deferred_global_model_publication(epoch, session_id=sid, generation=2)
    with pytest.raises(FenceStale):
        store.runtime.finish_deferred_global_model_publication(epoch + 1, session_id=sid, generation=1)
    assert store.runtime.deferred_global_model_result(sid) == applied.result
    newer_call, newer_accepted = _defer(store, epoch, sid, 'newer', 'current')
    newer = _apply(store, epoch, sid, 'current', 'newer')
    # Completion of an older publication can never erase the newer one's durable repair link.
    store.runtime.finish_deferred_global_model_publication(epoch, session_id=sid, generation=1)
    assert store.runtime.deferred_global_model_result(sid) == newer.result
    assert store.runtime.deferred_global_model_result(sid, request=identity) is None
    assert store.runtime.mutate(epoch, **original, owner=None).result == accepted
    assert store.runtime.mutate(epoch, **newer_call, owner=None).result == newer_accepted
    store.runtime.finish_deferred_global_model_publication(epoch, session_id=sid, generation=3)
    assert store.runtime.deferred_global_model_result(sid) is None
    assert store.runtime.has_mutation_receipt('p', sid, 'deferred-pick:original')
    assert store.runtime.has_mutation_receipt('p', sid, 'deferred-pick:newer')
    assert store.sessions.get(sid).runtime_generation == 3


@pytest.mark.parametrize('replacement', ['deferred', 'local', 'global'])
def test_failed_application_rolls_back_the_receipt_fence_and_link_and_a_superseded_request_has_no_link(backend, scope_a, replacement):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id='deferred-owner')
    sid = _local(store)
    original, accepted = _defer(store, epoch, sid)
    slot = store.runtime.deferred_pick(sid)
    # Failure after the policy generation starts changing must roll back the whole application.
    with pytest.raises(json.JSONDecodeError):
        _commit(store, epoch, sid, rt.ModelPick('picked', scope='global'),
                {'policy': {'model': 'picked', 'config_json': 'invalid', 'cwd': '/w'}},
                request_id='deferred-pick:original', expected_generation=0)
    assert store.sessions.get(sid).runtime_generation == 0
    assert not store.runtime.has_mutation_receipt('p', sid, 'deferred-pick:original')
    assert not store.runtime.is_latest_global_model_selection(session_id=sid, generation=1)
    assert store.runtime.deferred_global_model_result(sid) is None
    assert store.runtime.deferred_pick(sid) == slot
    if replacement == 'deferred':
        _defer(store, epoch, sid, 'newer', 'current')
        newer = _apply(store, epoch, sid, 'current', 'newer')
        assert store.runtime.deferred_global_model_result(sid) == newer.result
    else:
        _commit(store, epoch, sid, rt.ModelPick('current', scope='global' if replacement == 'global' else None),
                {'policy': {'model': 'current', 'config_json': '{"model":{"provider":"custom"}}', 'cwd': '/w'}},
                request_id='newer', expected_generation=0)
        assert store.runtime.deferred_global_model_result(sid) is None
    assert store.runtime.deferred_global_model_result(sid, request=rt.PinnedIdentity('p', 'original')) is None
    assert store.runtime.mutate(epoch, **original, owner=None).result == accepted
    assert store.sessions.get(sid).runtime_generation == 1
