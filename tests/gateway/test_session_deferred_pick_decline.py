"""Declining a guarded deferred model pick must drop it, not re-ask on every attach."""
import pytest

from persistence.authority.runtime import DeferredPick, mutation_receipt_key
from tests.fixtures.storage_read import session_row
from tests.gateway.test_session_mutation_model_busy import _owner


def _ask(authority, owner, ref, token):
    payload = {'model': 'switched'}
    key, digest = mutation_receipt_key(
        owner.actor.subject, ref.session_id, 'queued-pick', operation='model', payload=payload,
        expected_revision=1, expected_generation=1)
    runtime = authority.persistence.runtime
    runtime.defer_model_pick(
        authority.epoch, key=key, digest=digest, session_id=ref.session_id,
        pick=DeferredPick(operation='model', principal_id=owner.actor.subject, request_id='queued-pick',
                          payload=payload))
    slot = runtime.deferred_pick(ref.session_id)
    runtime.mark_deferred_pick_asked(ref.session_id, stored=slot.stored, confirm=token)
    return runtime.deferred_pick(ref.session_id)


@pytest.mark.asyncio
async def test_declining_a_guarded_deferred_pick_drops_it_so_attach_does_not_reask(tmp_path, monkeypatch):
    """A decline is the same as a refusal: the slot is forgotten, so the next attach does not re-ask."""
    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, lambda event: 'ok')
    try:
        asked = _ask(authority, owner, ref, 'tok')
        assert asked.pick.get('awaiting_confirm') == 'tok'
        snap = session_row(authority.persistence, ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'decline-1',
            'expected_revision': snap['runtime_revision'], 'expected_generation': snap['runtime_generation'],
            'operation': 'model', 'payload': {'decline': 'tok'}}})
        assert reply.get('result', {}).get('status') == 'declined', reply
        assert authority.persistence.runtime.deferred_pick(ref.session_id) is None
        live = authority.sessions[ref.session_id]
        from gateway.session_deferred_pick import reask_on_attach
        await reask_on_attach(authority, ref, live)
        assert live.reask_pick is False
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_decline_for_a_different_token_keeps_the_newer_ask(tmp_path, monkeypatch):
    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, lambda event: 'ok')
    try:
        _ask(authority, owner, ref, 'fresh')
        snap = session_row(authority.persistence, ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'decline-stale',
            'expected_revision': snap['runtime_revision'], 'expected_generation': snap['runtime_generation'],
            'operation': 'model', 'payload': {'decline': 'stale'}}})
        assert reply.get('result', {}).get('status') == 'declined', reply
        assert authority.persistence.runtime.deferred_pick(ref.session_id).pick.get('awaiting_confirm') == 'fresh'
        live = authority.sessions[ref.session_id]
        from gateway.session_deferred_pick import reask_on_attach
        await reask_on_attach(authority, ref, live)
        assert live.reask_pick is True
    finally:
        await owner.close()
        store.close_all_db_handles()
