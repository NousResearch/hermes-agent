"""Webhook capability contracts: CAS, source lifetime and owner admission gate.

Gateway regressions own presentation/wire behavior; these exercise the transactional
repository through its declared door, with no gateway SQL or callback connection.
"""
from dataclasses import replace

import pytest

from persistence import AuthorityRefusal
from persistence.authority.admissions import GateRefusal
from persistence.authority.webhooks import StoreIncarnation, WebhookLedger, WebhookRecord
from persistence.conversation.sessions import NewSession
from tests.contracts.conftest import require_capability


def obligation(origin_id, *, ledger=WebhookLedger.FINAL):
    return WebhookRecord(ledger, origin_id, 0, dict(origin_id=origin_id, delivery_id='b' * 64,
        producer_home='producer', target='bot-chat', target_home='recipient', target_pin=None,
        route='fixture', route_digest='route', message='private report', message_digest='digest',
        category='result', author=None, state='pending', uncertain=False))


def producer(backend, scope):
    require_capability(backend, 'webhook_delivery_fences')
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id='webhook-source', source='webhook'))
    epoch = store.runtime.begin_epoch(instance_id='webhook-contract')
    row, _ = store.admissions.admit(epoch=epoch, principal_id='webhook', session_id='webhook-source',
                                    request_id='event', payload={'text': 'source input'})
    claimed = store.admissions.claim(epoch=epoch, session_id='webhook-source')
    store.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=claimed.generation,
                            outcome='completed', result={'result': {'final_response': 'private report'}})
    return store, epoch, obligation(row.admission_id)


def test_publication_cas_and_epoch_never_rebind_a_retained_obligation(backend, scope_a):
    store, epoch, proposed = producer(backend, scope_a)
    saved = store.webhook_deliveries.publish(epoch=epoch, record=proposed)
    assert saved.revision == 1 and saved.value['message'] == 'private report'
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        store.webhook_deliveries.publish(epoch=epoch, record=proposed)
    changed = replace(saved, value={**saved.value, 'target_home': 'replacement'})
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        store.webhook_deliveries.publish(epoch=epoch, record=changed)
    store.runtime.begin_epoch(instance_id='successor')
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        store.webhook_deliveries.publish(epoch=epoch, record=saved, claim=True)
    assert store.webhook_deliveries.get(saved.ledger, saved.origin_id) == saved


def test_publication_and_predial_claim_refuse_missing_source_or_root(backend, scope_a):
    store, epoch, proposed = producer(backend, scope_a)
    missing = replace(proposed, origin_id='a' * 32, value={**proposed.value, 'origin_id': 'a' * 32})
    refused = store.webhook_deliveries.publish(epoch=epoch, record=missing)
    assert refused.value['state'] == 'refused' and refused.value['message'] == ''
    saved = store.webhook_deliveries.publish(epoch=epoch, record=proposed)
    refused = store.webhook_deliveries.publish(epoch=epoch, record=saved, claim=True,
                                              native={'webhook_root': 'missing-root'})
    assert refused.value['state'] == 'refused' and refused.value['reason'] == 'source_unauthorized'
    assert refused.value['source_retired_at'] == 0


def test_owner_gate_is_rechecked_inside_publication(backend, scope_a):
    store, epoch, proposed = producer(backend, scope_a)
    gates = []
    def retiring():
        gates.append('checked')
        return GateRefusal('runtime_draining')
    with pytest.raises(AuthorityRefusal, match='runtime_draining'):
        store.webhook_deliveries.publish(epoch=epoch, record=proposed, gate=retiring)
    assert gates == ['checked']
    assert store.webhook_deliveries.get(proposed.ledger, proposed.origin_id) is None


def test_bounded_candidates_and_inspection_are_profile_scoped_payload_free(backend, scope_a, scope_b):
    store, epoch, proposed = producer(backend, scope_a)
    for origin_id in ('a' * 64, 'b' * 64):
        item = obligation(origin_id, ledger=WebhookLedger.ORIGIN)
        store.webhook_deliveries.publish(epoch=epoch, record=item)
    first = store.webhook_deliveries.candidates(WebhookLedger.ORIGIN, limit=1)
    second = store.webhook_deliveries.candidates(WebhookLedger.ORIGIN, after=first[0], limit=1)
    assert first == ('a' * 64,) and second == ('b' * 64,)
    assert backend.open(scope_b).webhook_deliveries.candidates(WebhookLedger.ORIGIN) == ()
    page = store.webhook_deliveries.inspection(routes=('fixture',), limit=1)
    assert len(page) == 1 and page[0]['origin_id'] == first[0]
    assert not {'message', 'native', 'target_home', 'target_pin'} & set(page[0])
    assert store.webhook_deliveries.inspection(routes=('other-server',)) == ()


@pytest.mark.parametrize('patch', [dict(directory=[1]), dict(directory=[1, 0]),
    dict(store={'file': [1, 2], 'instance': 3}), dict(extra='untrusted')])
def test_incarnation_wire_rejects_malformed_proof(patch):
    pin = dict(home='recipient', directory=[1, 2], store={'file': [1, 3], 'instance': 'generation'})
    with pytest.raises(ValueError):
        StoreIncarnation.from_wire({**pin, **patch})
