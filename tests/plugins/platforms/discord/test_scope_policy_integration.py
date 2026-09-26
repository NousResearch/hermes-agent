"""Exercise scope policy through YAML seeding and real Discord ingress methods."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config, discord
from plugins.platforms.discord.policy import resolve_scope_policy


@pytest.mark.parametrize('invalid', [
    None,
    {'guilds': {'10': {'conversation_trust': 'private'}}},
    {1: True, 'unknown': True},
    {'platform': {1: True, 'unknown': True}},
    {'platform': {'defaults': {1: True, 'unknown': True}}},
    {'guilds': {'10': {1: True, 'unknown': True}}},
])
def test_yaml_seed_preserves_platform_defaults_and_legacy_flags(tmp_path, monkeypatch, invalid):
    cfg = yaml.safe_load("""
discord:
  require_mention: true
  scope_policies:
    platform:
      defaults:
        require_mention: false
        conversation_trust: private
    guilds:
      '10':
        channels:
          '20':
            allow_humans: false
""")
    from gateway.config import Platform, load_gateway_config

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    if invalid is not None:
        cfg['discord']['scope_policies'] = invalid
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    if invalid is not None:
        with pytest.raises(ValueError, match='unknown') as error:
            load_gateway_config()
        assert type(error.value).__name__ == 'ScopePolicyValidationError'
        return
    seeded = load_gateway_config().platforms[Platform.DISCORD].extra
    assert seeded['require_mention'] is True
    policy = resolve_scope_policy(seeded['scope_policies'], '10', '20')
    assert policy.require_mention is False
    assert policy.allow_humans is False
    assert policy.conversation_trust == 'private'


@pytest.mark.asyncio
@pytest.mark.parametrize('recovered', [False, True])
@pytest.mark.parametrize('is_dm,authorized,override,expected', [
    (False, True, False, True),
    (False, True, True, False),
    (False, True, None, False),
    (False, False, False, False),
    (True, True, True, True),
    (True, False, False, False),
])
async def test_scope_dispatch_preserves_native_admission(
    monkeypatch, recovered, is_dm, authorized, override, expected,
):
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '99')
    defaults = {'allow_humans': not is_dm, 'conversation_trust': 'full_trusted'}
    if override is not None:
        defaults['require_mention'] = override
    cfg = {'discord': {'scope_policies': {
        'platform': {'defaults': defaults},
        'guilds': {'10': {'channels': {'20': {'conversation_trust': 'private'}}}},
    }}}
    extra = _apply_yaml_config(cfg, cfg['discord'])
    extra.update(require_mention=True, auto_thread=False, history_backfill=False)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='test', extra=extra))
    adapter._allowed_user_ids = adapter._get_allowed_users()
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=123, bot=True))
    adapter._ready_event.set()
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    guild = None if is_dm else SimpleNamespace(id=10, name='guild')
    if is_dm:
        channel = discord.DMChannel.__new__(discord.DMChannel)
        channel.id = 20
    else:
        channel = SimpleNamespace(id=20, parent_id=None, guild=guild, name='channel', topic=None)
    message = SimpleNamespace(
        id=1, author=SimpleNamespace(id=99 if authorized else 98, bot=False, name='user', display_name='user', roles=[]),
        guild=guild, channel=channel, content='hello', mentions=[], attachments=[],
        reference=None, message_snapshots=[], created_at=datetime.now(timezone.utc),
        type=discord.MessageType.default,
    )
    _, source_kwargs = adapter._message_event_parts(message, lambda *_: {})
    event_source = adapter._source_for_platform_event(**source_kwargs)
    assert event_source.parent_chat_id is None
    assert event_source.conversation_trust == (None if is_dm else 'private')
    dispatch = adapter._dispatch_recovered_message if recovered else adapter._dispatch_discord_message
    result = await dispatch(message)
    assert result is expected
    assert adapter.handle_message.await_count == int(expected)
    if expected:
        source = adapter.handle_message.await_args.args[0].source
        assert source.conversation_trust == (None if is_dm else 'private')
