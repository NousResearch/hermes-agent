"""Native admission must retain attribution, including an explicit mixed-sender sentinel."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.platforms.event import absorb_envelope_sender
from gateway.session_envelope import prepare_native, restore_native, snapshot_native
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_mixed_sender_batching import _alice, _mallory, _prepared


@pytest.fixture
def runner(monkeypatch):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, '_load_gateway_config', dict)
    adapter = object()
    return SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False),
        session_store=SimpleNamespace(_generate_session_key=lambda source: 'topic'),
        _adapter_flag=lambda *args: False,
        _adapter_profile_for_source=lambda source: None,
        _is_user_authorized_for_source=lambda *args, **kwargs: True,
        _adapter_for_source=lambda source: adapter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ordinary', 'observed', 'mixed', 'unknown', 'bot'])
async def test_native_sender_round_trip_preserves_verified_note(runner, kind):
    event = _alice('hello')
    if kind in ('observed', 'bot'):
        event.envelope_sender = replace(event.source, is_bot=kind == 'bot')
        event.source = replace(event.source, user_id=None, user_name=None)
    elif kind == 'mixed':
        absorb_envelope_sender(event, _mallory('other'))
    elif kind == 'unknown':
        event.envelope_sender = replace(event.source, user_id=None, user_name=None)
    expected = await _prepared(event)
    payload = json.loads(json.dumps(await prepare_native(runner, event)))
    restored = restore_native(payload)
    assert await _prepared(restored) == expected
    assert restored.envelope_sender == event.envelope_sender
    assert restored.source == event.source
    if kind == 'ordinary':
        # Old snapshots omit the optional field and continue to fall back to source.
        assert 'envelope_sender' not in payload['native_text_v1']['event']
    elif kind in ('mixed', 'unknown'):
        assert restored.envelope_sender is not None
        assert restored.envelope_sender.user_id is None
        absorb_envelope_sender(restored, _alice('later'))
        assert 'Gateway-verified' not in await _prepared(restored)


@pytest.mark.parametrize('bad', [
    {}, '4242', SimpleNamespace(user_id='4242'),
    replace(_alice('').source, user_id=[]),
    replace(_alice('').source, user_name=[]),
    replace(_alice('').source, is_bot='false'),
])
def test_native_sender_rejects_invalid_adapter_attribution(runner, bad):
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        snapshot_native(runner, _alice('hello', envelope_sender=bad))


@pytest.mark.asyncio
async def test_native_sender_is_frozen_before_role_reauthorization(runner, monkeypatch):
    import gateway.session_ingress_context as context
    event = _alice('hello', envelope_sender=_alice('').source)

    async def reauthorize(*args):
        event.envelope_sender.user_id = '5151'
        return False

    monkeypatch.setattr(context, 'reauthorize_roles', reauthorize)
    restored = restore_native(await prepare_native(runner, event))
    assert restored.envelope_sender.user_id == '4242'


@pytest.mark.parametrize('bad', [{}, '4242', {'user_id': '4242', 'role_authorized': True}])
def test_native_sender_rejects_malformed_snapshot(runner, bad):
    payload = snapshot_native(runner, _alice('hello'))
    payload['native_text_v1']['event']['envelope_sender'] = bad
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        restore_native(payload)
