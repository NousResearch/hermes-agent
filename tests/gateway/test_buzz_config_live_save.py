"""Real router -> atomic config save -> SAME registered runner and adapter.

Tier: manually mounted ASGI router with synthetic auth; no host lifecycle claim.
Only transport/profile/model dispatch observations are faked.
"""
import json
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from tests.hermes_cli.test_buzz_config_policy import client, URL
from tests.gateway.test_live_buzz_authorization import registered_buzz, source


@pytest.mark.asyncio
async def test_endpoint_save_to_existing_runtime(registered_buzz, tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    home, runner, adapter = registered_buzz
    identities = (id(runner), id(adapter))
    c = client()
    adapter._self_pubkey = '1' * 64
    adapter._display_name = 'TestAgent'
    channel = 'synthetic-group'
    adapter._channel_meta[channel] = {'name': 'Group', 'description': 'shared'}
    adapter._channel_names[channel] = 'Group'
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic')
    adapter.send_reaction = AsyncMock()
    verdicts = []
    async def observe_turn(event):
        verdicts.append(runner._is_user_authorized(event.source))
    adapter.set_message_handler(observe_turn)
    adapter.send_typing = AsyncMock()
    state = adapter._new_channel_state('group')
    count = 0
    async def emit(user='a' * 64, mention=False, thread=False):
        nonlocal count
        count += 1
        event = dict(id=f'save-live-{count}', pubkey=user, content='@TestAgent hi' if mention else 'ordinary',
                     created_at=count, kind=9, tags=[['h', channel]])
        if thread:
            event['tags'].append(['e', 'synthetic-root', '', 'root'])
        before = len(verdicts)
        await adapter._handle_event(channel, state, event)
        import asyncio
        tasks = tuple(adapter._session_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        return any(verdicts[before:])
    def check(user):
        s = source(user, platform='buzz', profile='routed')
        s._transport_adapter_ref = weakref.ref(adapter)
        return runner._is_user_authorized(s)
    def put(**policy):
        response = c.put(URL, json={'policy': policy})
        assert response.status_code == 200, response.text
        import yaml
        raw = yaml.safe_load((home / 'config.yaml').read_text())
        saved = raw['gateway']['platforms']['buzz']['extra']
        assert all(saved[k] == v for k, v in policy.items())
        return response.json()
    assert check('a' * 64) is True and check('b' * 64) is False
    assert await emit(mention=True) and not await emit()
    put(allowed_users=['b' * 64], allow_all_users=False, require_mention=False, thread_require_mention=True)
    assert check('a' * 64) is False and check('b' * 64) is True
    profile_calls = adapter._resolve_user_name.await_count
    assert not await emit(mention=True)
    assert adapter._resolve_user_name.await_count == profile_calls
    assert await emit('b' * 64) and not await emit('b' * 64, thread=True)
    assert await emit('b' * 64, mention=True, thread=True)
    put(require_mention=True, thread_require_mention=False)
    assert not await emit('b' * 64) and await emit('b' * 64, thread=True)
    assert await emit('b' * 64, mention=True)
    # Editing another routed home must not redirect the transport owner's reads.
    other = tmp_path / 'other'; other.mkdir()
    (other / 'config.yaml').write_text('buzz: {allow_all_users: true}\n')
    (other / '.env').write_text('BUZZ_ALLOW_ALL_USERS=true\n')
    put(allowed_users=[], allow_all_users=False)
    token = set_hermes_home_override(other)
    try:
        assert check('b' * 64) is False
        assert not await emit('b' * 64, mention=True)
    finally:
        reset_hermes_home_override(token)
    # Platform empty is not universal denial: independent grants remain unions.
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert check('b' * 64) is True
    runner.pairing_store = None
    monkeypatch.setenv('GATEWAY_ALLOWED_USERS', '*')
    assert check('b' * 64) is True
    monkeypatch.delenv('GATEWAY_ALLOWED_USERS')
    put(allow_all_users=True, require_mention=False, thread_require_mention=False)
    assert await emit('b' * 64)
    (home / '.env').write_text('BUZZ_ALLOWED_USERS=\nBUZZ_ALLOW_ALL_USERS=false\n')
    assert check('b' * 64) is False
    (home / '.env').write_text('')
    assert check('b' * 64) is True
    replacement = home / 'next.yaml'; replacement.write_text('{}'); replacement.replace(home / 'config.yaml')
    assert check('b' * 64) is False
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert not await emit('b' * 64) and not await emit('b' * 64, thread=True)
    assert await emit('b' * 64, mention=True) and await emit('b' * 64, mention=True, thread=True)
    assert (id(runner), id(adapter)) == identities


@pytest.mark.asyncio
async def test_endpoint_revocation_and_mentions_existing_sender(registered_buzz):
    """Independent vertical, does not waive the new-sender snapshot blocker."""
    import asyncio
    import yaml
    home, runner, adapter = registered_buzz
    original_ids = id(runner), id(adapter)
    c = client()
    adapter._self_pubkey = '1' * 64
    adapter._display_name = 'TestAgent'
    adapter._channel_meta['group'] = {'name': 'Group', 'description': 'shared'}
    adapter._channel_names['group'] = 'Group'
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic')
    adapter.send_reaction = AsyncMock()
    adapter.send_typing = AsyncMock()
    verdicts = []
    async def observe(event):
        verdicts.append(runner._is_user_authorized(event.source))
    adapter.set_message_handler(observe)
    state = adapter._new_channel_state('group')
    count = 0
    async def emit(thread=False, mention=False):
        nonlocal count
        count += 1
        before = len(verdicts)
        event = dict(id=f'live-existing-{count}', kind=9, created_at=count, pubkey='a' * 64,
                     content=f'@TestAgent message {count}' if mention else f'ordinary {count}', tags=[['h', 'group']])
        if thread:
            event['tags'].append(['e', 'root', '', 'root'])
        await adapter._handle_event('group', state, event)
        tasks = tuple(adapter._session_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        return any(verdicts[before:])
    def put(**policy):
        response = c.put(URL, json={'policy': policy})
        assert response.status_code == 200, response.text
        raw = yaml.safe_load((home / 'config.yaml').read_text())['gateway']['platforms']['buzz']['extra']
        assert all(raw[k] == v for k, v in policy.items())
    assert await emit(mention=True)
    put(require_mention=False, thread_require_mention=True)
    assert await emit() and not await emit(thread=True)
    assert await emit(thread=True, mention=True)
    put(require_mention=True, thread_require_mention=False)
    assert not await emit() and await emit(thread=True)
    put(allowed_users=[], allow_all_users=False)
    assert runner._is_user_authorized(source(platform='buzz')) is False
    before = adapter._resolve_user_name.await_count, adapter.send_reaction.await_count
    assert not await emit(mention=True)
    assert before == (adapter._resolve_user_name.await_count, adapter.send_reaction.await_count)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert await emit(thread=True)
    runner.pairing_store = None
    put(allow_all_users=True)
    assert await emit(thread=True)
    replacement = home / 'next.yaml'; replacement.write_text('{}'); replacement.replace(home / 'config.yaml')
    assert runner._is_user_authorized(source(platform='buzz')) is False
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert not await emit() and not await emit(thread=True)
    assert await emit(mention=True) and await emit(thread=True, mention=True)
    assert (id(runner), id(adapter)) == original_ids
