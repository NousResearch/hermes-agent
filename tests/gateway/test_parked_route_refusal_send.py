"""A parked-profile refusal whose send fails is logged, with safe fields only.

``adapter.send`` reports most failures as ``SendResult(success=False)``, not by raising, so the
refusal must read the result: an unsent notice is a WARNING naming the failure kind, never the
message text or the platform's raw error (which can echo tokens or content).
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.parked_route_refusal import PARKED_ROUTE_NOTICE, drop_rejected_route
from gateway.platforms.base import SendResult

SECRET = 'xoxb-not-a-real-token'


class _Bot:
    name = 'Slack'

    def __init__(self, result):
        self._result, self.sent, self._background_tasks = result, [], set()
        self.gateway_runner = SimpleNamespace(
            session_runtime_descriptor={'parked_profiles': {'beta': 'test park'}},
            _thread_metadata_for_source=lambda source: None)

    async def send(self, chat_id, text, reply_to=None, metadata=None):
        self.sent.append((chat_id, text, reply_to))
        return self._result


async def _refuse(bot):
    source = SimpleNamespace(platform=Platform.SLACK, chat_id='C', message_id='m1',
                             _rejected_route_profile='beta')
    drop_rejected_route(bot, source)
    await asyncio.gather(*list(bot._background_tasks))


@pytest.mark.asyncio
async def test_an_unsuccessful_refusal_send_is_logged_without_text_or_raw_error(caplog):
    bot = _Bot(SendResult(success=False, error=f'invalid_auth for {SECRET}', error_kind='forbidden'))
    with caplog.at_level(logging.WARNING, logger='gateway.parked_route_refusal'):
        await _refuse(bot)
    assert len(bot.sent) == 1
    failures = [r.getMessage() for r in caplog.records if 'not delivered' in r.getMessage()]
    assert failures, f'a failed refusal send was not logged: {[r.getMessage() for r in caplog.records]}'
    assert 'error_kind=forbidden' in failures[0] and ' to C ' in failures[0], failures[0]
    assert all(r.levelno == logging.WARNING for r in caplog.records if 'not delivered' in r.getMessage())
    logged = caplog.text
    assert SECRET not in logged and PARKED_ROUTE_NOTICE not in logged
    assert 'was asked to resend' not in logged, 'the refusal is logged as delivered before it was sent'


@pytest.mark.asyncio
async def test_a_successful_refusal_send_logs_no_failure(caplog):
    bot = _Bot(SendResult(success=True, message_id='n1'))
    with caplog.at_level(logging.WARNING, logger='gateway.parked_route_refusal'):
        await _refuse(bot)
    assert len(bot.sent) == 1
    assert not [r for r in caplog.records if 'not delivered' in r.getMessage()]


@pytest.mark.asyncio
async def test_a_backlog_for_a_parked_profile_tells_each_chat_once_per_window(monkeypatch):
    """A relay reconnect drains a backlog of N frames for a parked profile; the chat is told once
    per (profile, platform, chat) and window, not N times. Another chat is still told."""
    from gateway import parked_route_refusal
    clock = [1000.0]
    monkeypatch.setattr(parked_route_refusal, '_monotonic', lambda: clock[0])
    bot = _Bot(SendResult(success=True))
    for _ in range(5):
        await _refuse(bot)
    other = SimpleNamespace(platform=Platform.SLACK, chat_id='D', message_id='m2', _rejected_route_profile='beta')
    drop_rejected_route(bot, other)
    await asyncio.gather(*list(bot._background_tasks))
    assert [chat for chat, _text, _reply in bot.sent] == ['C', 'D']
    clock[0] += parked_route_refusal.PARKED_NOTICE_WINDOW_S
    await _refuse(bot)
    assert [chat for chat, _text, _reply in bot.sent] == ['C', 'D', 'C']
