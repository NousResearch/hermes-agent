"""Physical writer/retention boundaries not modeled by the memory reference.

Wire/native integration tests own profile routing. These cases call the actual
repository and typed admission extra against the same SQLite writer.
"""
from dataclasses import replace
import time

import pytest

from persistence import AuthorityRefusal
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.authority.webhooks import WebhookLedger
from persistence.conversation.sessions import NewSession
from tests.contracts.authority.test_webhook_deliveries_contract import obligation


def test_incarnation_extra_is_rechecked_in_admission_transaction(store):
    store.sessions.create(NewSession(session_id='recipient-chat', source='cli'))
    epoch = store.runtime.begin_epoch(instance_id='recipient-owner')
    expected = store.webhook_deliveries.capture_incarnation()
    store.webhook_deliveries.require_incarnation(expected)
    # Replacement after the mailbox's early check, before the FIFO write. Keep
    # inode and directory unchanged so this isolates the OPEN connection's instance.
    store.state_meta.set('store_instance_id', 'replacement-generation')
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        store.admissions.admit(epoch=epoch, principal_id='bot', session_id='recipient-chat',
                                request_id='bot:delivery', payload={'text': 'never admitted'}, incarnation=expected)
    assert store.admissions.list(session_id='recipient-chat', pending_only=False) == []
    assert store.state_meta.get('store_instance_id') == 'replacement-generation'


def test_exact_incarnation_keeps_native_fifo_identity_and_legacy_callers(store):
    store.sessions.create(NewSession(session_id='recipient-chat', source='cli'))
    epoch = store.runtime.begin_epoch(instance_id='recipient-owner')
    pin = store.webhook_deliveries.capture_incarnation()
    kwargs = dict(epoch=epoch, principal_id='bot', session_id='recipient-chat',
                  request_id='bot:delivery', payload={'text': 'one turn'})
    first, created = store.admissions.admit(**kwargs, incarnation=pin)
    replay, duplicate = store.admissions.admit(**kwargs, incarnation=pin)
    legacy, legacy_created = store.admissions.admit(**{**kwargs, 'request_id': 'legacy-native'})
    assert created and not duplicate and legacy_created
    assert first.admission_id == replay.admission_id != legacy.admission_id
    assert [r.request_id for r in store.admissions.list(session_id='recipient-chat')] == ['bot:delivery', 'legacy-native']


def test_compaction_fences_stale_observer_and_keeps_replay_digests(store, monkeypatch):
    epoch = store.runtime.begin_epoch(instance_id='producer')
    proposed = obligation('a' * 64, ledger=WebhookLedger.ORIGIN)
    proposed = replace(proposed, value={**proposed.value, 'created_at': time.time(),
                                       'native': {'source': {'platform': 'webhook'}}})
    saved = store.webhook_deliveries.publish(epoch=epoch, record=proposed)
    claimed = store.webhook_deliveries.publish(epoch=epoch, record=replace(
        saved, value={**saved.value, 'uncertain': True}), claim=True)
    # The physical handoff could still finish; expiry is not proof of non-admission.
    at = proposed.value['created_at'] + RETRY_HORIZON_S + 1
    monkeypatch.setattr(time, 'time', lambda: at)
    store.maintenance.prune_sessions(older_than_days=90, media_root=str(store.scope.profile_dir))
    compacted = store.webhook_deliveries.get(WebhookLedger.ORIGIN, proposed.origin_id)
    assert compacted.value['message'] == '' and 'native' not in compacted.value
    assert compacted.value['payload_compacted_at'] == at
    assert compacted.value['delivery_id'] == claimed.value['delivery_id']
    assert compacted.value['message_digest'] == claimed.value['message_digest']
    assert compacted.value['uncertain'] is True
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        store.webhook_deliveries.publish(epoch=epoch, record=replace(
            claimed, value={**claimed.value, 'state': 'settled'}))
    assert store.webhook_deliveries.get(WebhookLedger.ORIGIN, proposed.origin_id) == compacted
