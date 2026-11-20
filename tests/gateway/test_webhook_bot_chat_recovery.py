"""Recovery hands off committed original finals without rerunning source work."""
from tests.fixtures.storage_holder import raw_holder
from pathlib import Path
import threading

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'settled'])
async def test_ended_source_retries_unavailable_owner_and_observes_receipt(monkeypatch, status):
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    from tools import bot_live_delivery

    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, _runner, authority, row):
        sid = row['target_session_id']
        raw_holder(authority.persistence)._write_sql("UPDATE sessions SET ended_at=1 WHERE id=?", (sid,))
        receipts, calls = {}, []
        loop = threading.get_ident()
        def read(home, key):
            assert threading.get_ident() != loop
            return receipts.get(key)
        def discover(home):
            assert threading.get_ident() != loop
            return {'owner': True} if calls else None
        def send(home, owner, message, *, delivery_id, **kwargs):
            assert threading.get_ident() != loop
            calls.append((Path(home), delivery_id, message))
            receipts[delivery_id] = {'status': status, 'admission_id': 'target', 'delivery_id': delivery_id,
                'profile_home': str(home), 'session_id': 'target-session', 'message': message}
            return receipts[delivery_id]
        monkeypatch.setattr(bot_live_delivery, 'read_delivery_result', read)
        monkeypatch.setattr(bot_live_delivery, 'find_canonical_live_owner', discover)
        monkeypatch.setattr(bot_live_delivery, 'deliver_to_live_owner', send)
        await recover_webhook_finalizations(authority)
        assert not calls
        calls.append('owner available')
        await recover_webhook_finalizations(authority)
        assert len(calls) == 2
        home, key, message = calls[-1]
        assert home == authority.persistence.scope.profile_dir
        assert 'committed answer' in message and 'Webhook automation' in message
        await recover_webhook_finalizations(authority)
        assert len(calls) == 2 and receipts[key]['status'] == status
        assert len(raw_holder(authority.persistence)._read_all('SELECT * FROM session_admissions')) == 1


@pytest.mark.asyncio
async def test_changed_route_denies_recovery(monkeypatch):
    from gateway.platforms.webhook_ingress import recover_webhook_finalizations
    from tools import bot_live_delivery
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, _row):
        adapter = next(iter(runner.adapters.values()))
        adapter._routes['fixture']['prompt'] = 'changed'
        def forbidden(*args, **kwargs):
            pytest.fail('changed route must not reach native transport')
        monkeypatch.setattr(bot_live_delivery, 'read_delivery_result', forbidden)
        await recover_webhook_finalizations(authority)


def test_durable_snapshot_rejects_invalid_current_mode():
    from types import SimpleNamespace
    from gateway.platforms.webhook_delivery import route_digest
    from persistence.core.errors import AuthorityRefusal
    adapter = SimpleNamespace(_reload_dynamic_routes=lambda: None,
                              _routes={'fixture': {'deliver': 'bot-chat', 'deliver_extra': {'session_id': 'x'}}})
    with pytest.raises(AuthorityRefusal):
        route_digest(adapter, 'webhook:fixture:delivery')
