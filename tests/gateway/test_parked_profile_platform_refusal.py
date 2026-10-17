"""A message a direct platform bot routes to a parked profile is refused out loud, never dropped.

A parked profile comes back on a rescan, so its traffic is temporarily unavailable. Relay and webhook
ingress are redelivered (no ACK, a retryable 503). A direct bot (Telegram, Discord, Slack, ...) has
already confirmed the update to its platform before the gateway sees it, so there is nothing to
withhold: the sender is told in-channel to resend, and nothing is admitted.
"""
import asyncio
import logging
import weakref

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.profile_routing import ProfileRoute
from gateway.session import SessionSource
from tests.gateway.test_parked_profile_ingress import _admissions, _close, _homes, _runner_with_beta_parked


class _Bot(BasePlatformAdapter):
    """The default profile's shared bot; ``sent`` records what went back to the chat."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token='t'), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content='', reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to))
        return SendResult(success=True, message_id='r1')

    async def get_chat_info(self, chat_id):
        return {}


async def _shared_bot_with_beta_parked(tmp_path, monkeypatch):
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(
        tmp_path, monkeypatch, homes,
        profile_routes=[ProfileRoute(name='to-beta', platform='telegram', profile='beta', chat_id='C')])
    bot = _Bot()
    bot.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = bot
    runner._wire_adapter_handlers(bot)
    return homes, runner, bot


async def _settle(bot):
    await asyncio.gather(*list(bot._background_tasks), return_exceptions=True)


def _assert_refused(bot, homes, event, caplog):
    assert [(chat, reply_to) for chat, _text, reply_to in bot.sent] == [('C', 'm1')], (
        f'the sender of a message routed to a parked profile was not told: {bot.sent}')
    assert 'temporarily unavailable' in bot.sent[0][1] and 'resend' in bot.sent[0][1], bot.sent
    assert 'beta' not in bot.sent[0][1], 'the refusal names the routed profile to the chat'
    assert event._gateway_accepted is False
    assert _admissions(homes[0][1]) == 0 and _admissions(homes[1][1]) == 0
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('beta' in m and 'telegram' in m and 'profile_parked' in m for m in warned), warned


@pytest.mark.asyncio
async def test_a_shared_bot_message_routed_to_a_parked_profile_is_refused_in_channel(tmp_path, monkeypatch, caplog):
    homes, runner, bot = await _shared_bot_with_beta_parked(tmp_path, monkeypatch)
    source = bot.build_source('C', chat_type='group', user_id='U', message_id='m1')
    event = MessageEvent(text='please do this', message_type=MessageType.TEXT, source=source, message_id='m1')
    try:
        with caplog.at_level(logging.WARNING):
            await bot.handle_message(event)
            await _settle(bot)
        _assert_refused(bot, homes, event, caplog)
    finally:
        _close(runner, homes)


@pytest.mark.asyncio
async def test_a_hand_built_source_routed_to_a_parked_profile_is_refused_at_the_runner_gate(
        tmp_path, monkeypatch, caplog):
    """Internal and voice paths build the source themselves and reach the runner's ingress gate."""
    homes, runner, bot = await _shared_bot_with_beta_parked(tmp_path, monkeypatch)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='C', chat_type='group', user_id='U', message_id='m1')
    source._transport_adapter_ref = weakref.ref(bot)
    event = MessageEvent(text='please do this', message_type=MessageType.TEXT, source=source, message_id='m1')
    try:
        with caplog.at_level(logging.WARNING):
            assert await runner._hm_admit_event(event) is None
            await _settle(bot)
        _assert_refused(bot, homes, event, caplog)
    finally:
        _close(runner, homes)
