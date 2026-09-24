"""Installed no-loss contract, POLICY-DISPOSITION.md item 3.

The installed real-import oracle (characterization-policy-edges, supplement r2)
admits legacy positional replies by thread policy, not channel policy. Exercise
candidate constructor -> real live loader -> intake -> actual MessageEvent.
Structural own-parent control admission remains a distinct, narrower gate.
"""
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter('buzz')
SELF, USER = '1' * 64, '2' * 64
CHANNEL, ROOT, PARENT = 'legacy-channel', 'legacy-root', 'legacy-parent'


@pytest.fixture
def intake(tmp_path, monkeypatch):
    home = tmp_path / 'legacy-owner'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    def save(channel, thread):
        replacement = home / 'next.yaml'
        replacement.write_text(yaml.safe_dump({'buzz': {
            'require_mention': channel, 'thread_require_mention': thread}}))
        replacement.replace(home / 'config.yaml')
    save(True, True)
    adapter = buzz.BuzzAdapter(PlatformConfig(enabled=True))
    adapter._self_pubkey = SELF
    adapter._display_name = 'TestAgent'
    adapter._channel_names[CHANNEL] = 'Synthetic shared channel'
    adapter._channel_meta[CHANNEL] = {'name': 'Synthetic shared channel', 'description': 'Not DM'}
    adapter.set_authorization_check(lambda *_: True)
    adapter._resolve_user_name = AsyncMock(return_value='Synthetic user')
    adapter.send_reaction = AsyncMock()
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    state = adapter._new_channel_state('group')
    sequence = 0
    async def emit(tags, content='ordinary follow up', *, own_id=None):
        nonlocal sequence
        sequence += 1
        event = dict(id=own_id or f'legacy-{sequence}', kind=9,
                     pubkey=SELF if own_id else USER, created_at=sequence,
                     content=content, tags=[['h', CHANNEL], *tags])
        before = adapter.handle_message.await_count
        await adapter._handle_event(CHANNEL, state, event)
        assert state['chat_type'] == 'group'
        if adapter.handle_message.await_count > before:
            message = adapter.handle_message.await_args.args[0]
            assert message.message_id == event['id']
            return message
        return None
    return adapter, save, emit


@pytest.mark.asyncio
@pytest.mark.parametrize('channel,thread', [(True, False), (False, True)])
@pytest.mark.parametrize('tags', [
    [['e', ROOT]],
    [['e', ROOT], ['e', PARENT]],
], ids=['single-unmarked', 'two-positional'])
async def test_legacy_intake_uses_installed_thread_policy(intake, channel, thread, tags):
    adapter, save, emit = intake
    # Check the inverse on the SAME constructed adapter as well: policy stays live.
    for channel, thread in [(channel, thread), (thread, channel)]:
        save(channel, thread)
        direct = await emit(tags, '@TestAgent hello')
        assert direct is not None and direct.source.thread_id == ROOT
        p_direct = await emit(tags + [['p', SELF]])
        assert p_direct is not None and p_direct.source.thread_id == ROOT
        assert (await emit([]) is not None) is (not channel)
        assert (await emit([['e', ROOT, '', 'root']]) is not None) is (not thread)
        message = await emit(tags)
        assert (message is not None) is (not thread), 'installed legacy thread-policy contract'
        if message is not None:
            assert message.source.thread_id == ROOT


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approve session', '/deny'])
@pytest.mark.parametrize('tags', [
    [['e', ROOT]],
    [['e', ROOT], ['e', PARENT]],
], ids=['single-unmarked', 'two-positional'])
async def test_legacy_own_parent_does_not_expand_control_exemption(intake, command, tags):
    adapter, save, emit = intake
    # A permissive channel cannot accidentally admit a legacy control while its
    # thread is strict, even if the cached final positional parent is ours.
    save(False, True)
    assert await emit([], own_id=tags[-1][1]) is None
    assert await emit(tags, command) is None
    marked = [['e', ROOT, '', 'root'], ['e', tags[-1][1], '', 'reply']]
    admitted = await emit(marked, command)
    assert admitted is not None and admitted.source.thread_id == ROOT
    assert admitted.text == command
    # Direct p-mention is ordinary admission, not a new legacy control exception.
    direct = await emit(tags + [['p', SELF]], command)
    assert direct is not None and direct.source.thread_id == ROOT
