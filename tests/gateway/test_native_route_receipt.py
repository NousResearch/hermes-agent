"""Short-lived native route receipts supply a synchronous fence, not permission."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_envelope import _preflight_native_route, _validate_native_route, check_native_route
from hermes_state_runtime import RuntimeStoreError


@pytest.fixture
def route(monkeypatch):
    source = SessionSource(Platform.DISCORD, 'channel', chat_type='group', user_id='member')
    adapter = object()
    state = SimpleNamespace(direct=True, role=True, checks=0, adapter=adapter, entry='sid')
    store = SimpleNamespace(
        _generate_session_key=lambda candidate: 'route' if candidate.chat_id == 'channel' else 'other',
        lookup_by_session_key=lambda key: SimpleNamespace(session_id=state.entry) if key == 'route' else None,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False), session_store=store,
        _adapter_for_source=lambda candidate: state.adapter,
        _is_user_authorized_for_source=lambda candidate, allow_adapter_delegation=False:
            state.direct or (allow_adapter_delegation and state.role),
    )
    # The connector's asynchronous role check is the only network seam; production
    # restoration, sender validation, physical route and adapter checks remain real.
    async def reauthorize(runner, candidate, provenance):
        state.checks += 1
        return candidate.role_authorized and (state.direct or state.role)
    monkeypatch.setattr('gateway.session_ingress_context.reauthorize_roles', reauthorize)
    monkeypatch.setattr('gateway.session_ingress_context.restore_provenance', lambda *args: None)
    payload = {
        'text': 'hello',
        'native_text_v1': {
            'source': {**source.to_dict(), 'is_bot': False},
            'route': 'route', 'event': {'message_id': 'message-1'},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'provenance': {'connector': 'fixture'}, 'reauthorize': 'roles',
        },
    }
    return runner, payload, source, adapter, state


@pytest.mark.asyncio
async def test_receipt_rechecks_direct_grant_and_claim_wrapper_requires_fresh_roles(route):
    runner, payload, source, adapter, state = route
    result, receipt = await _preflight_native_route(runner, payload, 'sid', source, adapter)
    assert result[0].to_dict() == source.to_dict() and result[1] == 'route'
    assert result[0].role_authorized
    assert state.checks == 1
    assert receipt.direct_only and receipt.fresh_roles
    assert _validate_native_route(runner, payload, 'sid', source, adapter, receipt)[1] == 'route'
    assert state.checks == 1  # The whole-batch fence cannot yield to the connector.
    state.direct = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)
    assert state.checks == 1  # A formerly direct grant cannot become role reuse.
    assert (await check_native_route(runner, payload, 'sid', source, adapter))[1] == 'route'
    assert state.checks == 2  # Preclaim still awaits its own fresh role check.
    state.role = False
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await check_native_route(runner, payload, 'sid', source, adapter)
    assert state.checks == 3


@pytest.mark.asyncio
async def test_receipt_revalidates_payload_sender_route_adapter_and_role_only(route):
    runner, payload, source, adapter, state = route
    state.direct = False
    _, receipt = await _preflight_native_route(runner, payload, 'sid', source, adapter)
    assert receipt.fresh_roles and not receipt.direct_only
    assert _validate_native_route(runner, payload, 'sid', source, adapter, receipt)[1] == 'route'
    changed = deepcopy(payload)
    changed['text'] = 'tampered'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, changed, 'sid', source, adapter, receipt)
    changed = deepcopy(payload)
    changed['native_text_v1']['source']['user_id'] = 'someone-else'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, changed, 'sid', source, adapter, receipt)
    state.entry = 'new-sid'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)
    state.entry = 'sid'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, payload, 'sid', SessionSource(Platform.DISCORD, 'other'), adapter, receipt)
    state.adapter = object()
    with pytest.raises(RuntimeStoreError, match='not_found'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)
    state.adapter = adapter
    state.role = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)


@pytest.mark.asyncio
async def test_webhook_receipt_rechecks_destination_and_live_route(route):
    from gateway.platforms.webhook_delivery import route_digest

    runner, payload, source, _, state = route
    source = SessionSource(Platform.WEBHOOK, 'webhook:fixture:delivery', user_id='sender')
    adapter = SimpleNamespace(_routes={'fixture': {'deliver': 'log'}},
                              _reload_dynamic_routes=lambda: None)
    state.adapter = adapter
    runner.session_store._generate_session_key = lambda candidate: (
        'route' if candidate.chat_id == source.chat_id else 'other')
    payload['native_text_v1'].update(
        source={**source.to_dict(), 'is_bot': False}, webhook_delivery={'deliver': 'log', 'deliver_extra': {}},
        webhook_route=route_digest(adapter, source.chat_id),
    )
    payload['native_text_v1'].pop('reauthorize')
    _, receipt = await _preflight_native_route(runner, payload, 'sid', source, adapter)
    assert _validate_native_route(runner, payload, 'sid', source, adapter, receipt)[1] == 'route'
    adapter._routes['fixture'] = {'deliver': 'github_comment'}
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)


@pytest.mark.asyncio
async def test_automation_receipt_uses_existing_synchronous_owner_check(route, monkeypatch):
    runner, payload, source, adapter, state = route
    payload['native_text_v1']['automation'] = {'owner': 'sid'}
    calls = []
    def check(runner, payload, session_id, available_source, adapter):
        calls.append(session_id)
        if state.entry != session_id:
            raise RuntimeStoreError('admission_conflict')
        return source, 'route'
    monkeypatch.setattr('gateway.session_automation.check_automation_route', check)
    _, receipt = await _preflight_native_route(runner, payload, 'sid', source, adapter)
    assert not receipt.fresh_roles and not receipt.direct_only and state.checks == 0
    assert _validate_native_route(runner, payload, 'sid', source, adapter, receipt) == (source, 'route')
    state.entry = 'new-sid'
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        _validate_native_route(runner, payload, 'sid', source, adapter, receipt)
    assert calls == ['sid', 'sid', 'sid']
