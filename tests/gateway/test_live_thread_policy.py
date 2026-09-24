"""Test-only installed-contract characterization against unchanged stock.
Real canonical raw loader -> PlatformConfig -> BuzzAdapter -> _handle_event ->
_dispatch_message -> MessageEvent. Only transport/profile/model boundaries stubbed.
"""
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
import yaml
from gateway.config import PlatformConfig
from hermes_cli.config import read_user_config_raw
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter('buzz')
SELF = '1' * 64
OTHER = '2' * 64
CHANNEL = 'synthetic-channel'

CASES = [
    ('channel_open_thread_strict', [{'require_mention': False, 'thread_require_mention': True}], [(True, False)]),
    ('channel_strict_thread_open', [{'require_mention': True, 'thread_require_mention': False}], [(False, True)]),
    ('channel_change_next_event', [{'require_mention': True, 'thread_require_mention': True}, {'require_mention': False, 'thread_require_mention': True}], [(False, False), (True, False)]),
    ('channel_deletion_next_event', [{'require_mention': False, 'thread_require_mention': True}, {'thread_require_mention': True}], [(True, False), (False, False)]),
    ('thread_change_and_deletion', [{'require_mention': True, 'thread_require_mention': True}, {'require_mention': True, 'thread_require_mention': False}, {'require_mention': True}], [(False, False), (False, True), (False, False)]),
    ('strict_reply_to_own_old_event', [{'require_mention': True, 'thread_require_mention': True}], [(False, False)]),
]

@pytest.mark.asyncio
@pytest.mark.parametrize('name,policies,expected', CASES, ids=[c[0] for c in CASES])
async def test_live_independent_policy(name, policies, expected, tmp_path, monkeypatch):
    home = tmp_path / 'synthetic-hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    config = home / 'config.yaml'
    def save(policy):
        extra = {'relay_url': 'https://synthetic.invalid', **policy}
        replacement = config.with_suffix('.next')
        replacement.write_text(yaml.safe_dump({'gateway': {'platforms': {'buzz': {'enabled': True, 'extra': extra}}}}))
        replacement.replace(config)
        raw = read_user_config_raw(config)
        assert raw['gateway']['platforms']['buzz']['extra'] == extra, 'SETUP: canonical loader must see exact synthetic policy'
        return raw['gateway']['platforms']['buzz']
    adapter = buzz.BuzzAdapter(PlatformConfig.from_dict(save(policies[0])))
    identity = id(adapter)
    adapter._self_pubkey = SELF
    adapter._display_name = 'TestAgent'
    adapter._channel_meta[CHANNEL] = {'name': 'Synthetic shared channel', 'description': 'Not a DM'}
    adapter._channel_names[CHANNEL] = 'Synthetic shared channel'
    adapter.set_authorization_check(lambda *_: True)
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic user')
    adapter.send_reaction = AsyncMock()
    adapter._message_handler = AsyncMock()  # Marks intake connected, without starting a gateway/model.
    adapter.handle_message = AsyncMock()  # Observe fully constructed real MessageEvent boundary.
    state = adapter._new_channel_state('group')
    count = 0
    async def emit(label, thread=False, mention=False, own=False):
        nonlocal count
        count += 1
        event = {'id': 'own-parent' if own else f'{name}-{count}-{label}', 'pubkey': SELF if own else OTHER,
                 'content': '@TestAgent hello' if mention else 'ordinary follow up',
                 'created_at': 1000 + count, 'kind': 9, 'tags': [['h', CHANNEL]]}
        if thread:
            event['tags'] += [['e', 'synthetic-root', '', 'root'], ['e', 'own-parent' if name == 'strict_reply_to_own_old_event' else 'other-parent', '', 'reply']]
        before = adapter.handle_message.await_count
        await adapter._handle_event(CHANNEL, state, event)
        assert state['chat_type'] == 'group', 'SETUP: must not reclassify as DM'
        if adapter.handle_message.await_count > before:
            dispatched = adapter.handle_message.await_args.args[0]
            assert dispatched.message_id == event['id'], 'SETUP: real MessageEvent identity'
        return adapter.handle_message.await_count > before
    if name == 'strict_reply_to_own_old_event':
        assert await emit('seed-self', own=True) is False
    observed = []
    controls = []
    for index, policy in enumerate(policies):
        save(policy)
        observed.append((await emit('channel'), await emit('thread', thread=True)))
        controls.append((await emit('channel-direct', mention=True), await emit('thread-direct', thread=True, mention=True)))
        assert id(adapter) == identity
    result = {'case': name, 'policies': policies, 'observed': observed, 'expected': expected, 'direct_mention_controls': controls, 'adapter_reconstructed': False}
    if os.environ.get('THREAD_OBSERVATIONS'):
        with Path(os.environ['THREAD_OBSERVATIONS']).open('a') as output:
            output.write(json.dumps(result) + '\n')
    assert all(c == (True, True) for c in controls), f'CONTROL: direct mentions must dispatch: {controls!r}'
    assert observed == expected, f'POLICY {name}: observed={observed!r}; expected={expected!r}'


@pytest.mark.asyncio
async def test_owner_scope_malformed_deletion_and_positive_controls(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    owner, routed = tmp_path / 'owner', tmp_path / 'routed'
    owner.mkdir(); routed.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(owner))
    def save(text):
        replacement = owner / 'next.yaml'
        replacement.write_text(text)
        replacement.replace(owner / 'config.yaml')
    save(yaml.safe_dump({'buzz': {'require_mention': False, 'thread_require_mention': True}}))
    adapter = buzz.BuzzAdapter(PlatformConfig())
    adapter._self_pubkey = SELF
    adapter._display_name = 'TestAgent'
    adapter._channel_meta[CHANNEL] = {'name': 'Synthetic channel', 'description': 'Group'}
    adapter._channel_names[CHANNEL] = 'Synthetic channel'
    adapter.set_authorization_check(lambda *_: True)
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic user')
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    state = adapter._new_channel_state('group')
    count = 0
    async def emit(tags=(), content='ordinary', dm=False):
        nonlocal count
        count += 1
        event = dict(id=f'positive-{count}', pubkey=OTHER, content=content,
                     kind=9, created_at=count, tags=[['h', CHANNEL], *tags])
        current = adapter._new_channel_state('dm') if dm else state
        before = adapter.handle_message.await_count
        await adapter._handle_event(CHANNEL, current, event)
        return adapter.handle_message.await_count > before
    root = [['e', 'root-id', '', 'root']]
    token = set_hermes_home_override(routed)
    try:
        # Captured transport owner survives a routed context switch.
        assert await emit()
        assert not await emit(root)
        # Supersedes the added channel-classification assertion: installed oracle
        # and POLICY-DISPOSITION item 3 require legacy replies to use thread policy.
        assert not await emit([['e', 'legacy-id']])
        assert await emit([['e', 'legacy-id']], '@TestAgent hello')
        assert adapter.handle_message.await_args.args[0].source.thread_id == 'legacy-id'
        assert await emit(root, '@TestAgent hello')
        assert await emit(root + [['p', SELF]])
        assert await emit(dm=True)
        save('gateway: [')
        assert await emit()  # Last good exact-owner policy.
        (owner / 'config.yaml').unlink()
        assert not await emit()  # Deletion restores strict default.
        assert not await emit(root)
        assert await emit(content='@TestAgent still allowed')
        assert await emit(root + [['p', SELF]])
        assert await emit(dm=True)
    finally:
        reset_hermes_home_override(token)


def test_yaml_mention_values_never_bridge_to_process_environment(monkeypatch):
    monkeypatch.delenv('BUZZ_REQUIRE_MENTION', raising=False)
    monkeypatch.delenv('BUZZ_THREAD_REQUIRE_MENTION', raising=False)
    buzz._apply_yaml_config({}, {'extra': {'require_mention': False, 'thread_require_mention': False}})
    assert 'BUZZ_REQUIRE_MENTION' not in os.environ
    assert 'BUZZ_THREAD_REQUIRE_MENTION' not in os.environ
