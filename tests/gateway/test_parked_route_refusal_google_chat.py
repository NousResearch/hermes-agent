"""A Google Chat HTTP event routed to a parked profile is answered retryably; Pub/Sub is refused in chat.

The HTTP callback awaits the dispatch before it answers, so unlike Pub/Sub (acked on hand-off) the
delivery is not yet acknowledged: Google retries a failed or non-2xx interaction event. So the
parked-profile refusal answers 503 and releases the adapter's dedup claim, and the redelivery after
the rescan is admitted exactly once.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult
from gateway.profile_routing import ProfileRoute
from tests.gateway.test_google_chat import GoogleChatAdapter, _make_chat_envelope
from tests.gateway.test_parked_profile_ingress import (
    _admissions, _close, _homes, _runner_with_beta_parked, _serve_beta_again,
)
from tests.waiting import async_wait_until

MESSAGE, SPACE, THREAD = 'spaces/S/messages/M.M', 'spaces/S', 'spaces/S/threads/T'


async def _shared_chat_bot_with_beta_parked(tmp_path, monkeypatch):
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(
        tmp_path, monkeypatch, homes,
        profile_routes=[ProfileRoute(name='to-beta', platform='google_chat', profile='beta', chat_id=SPACE)])
    monkeypatch.setattr(runner, '_is_user_authorized', lambda *_a, **_k: True)  # allowlists are not under test
    bot = GoogleChatAdapter(PlatformConfig(enabled=True))
    bot._thread_count_store._path = tmp_path / 'google_chat_thread_counts.json'
    bot.gateway_runner = runner
    runner.adapters[Platform('google_chat')] = bot
    runner._wire_adapter_handlers(bot)
    bot.send = AsyncMock(return_value=SendResult(success=True, message_id='notice'))
    bot.verify_http_event_request = lambda header: (True, '')
    return homes, runner, bot


async def _post(bot, envelope):
    """The real platform-event callback, as Google's HTTP delivery reaches it."""
    request = SimpleNamespace(match_info={'platform': 'google_chat'}, app={'google_chat_adapter': bot},
                              headers={}, json=AsyncMock(return_value=envelope))
    response = await object.__new__(APIServerAdapter)._handle_platform_event_callback(request)
    await asyncio.gather(*list(bot._background_tasks), return_exceptions=True)
    return response


@pytest.mark.asyncio
async def test_an_http_event_for_a_parked_profile_is_retried_and_admitted_once(tmp_path, monkeypatch):
    homes, runner, bot = await _shared_chat_bot_with_beta_parked(tmp_path, monkeypatch)
    beta = homes[1][1]
    try:
        refused = await _post(bot, _make_chat_envelope())
        assert refused.status == 503, (
            f'not admitted but acknowledged with {refused.status}: Google will not retry it '
            f'(dedup holds it: {bot._dedup.contains(MESSAGE)})')
        assert json.loads(refused.text)['error']['code'] == 'profile_parked', refused.text
        assert refused.headers.get('Retry-After')
        assert not bot._dedup.contains(MESSAGE), 'the unadmitted delivery keeps its dedup claim'
        assert bot.send.await_count == 0, 'Google retries it: no in-chat notice as well'
        assert _admissions(beta) == 0
        await _serve_beta_again(runner, beta.resolve())
        retried = await _post(bot, _make_chat_envelope())
        assert retried.status == 200, retried.text
        await async_wait_until(lambda: _admissions(beta) == 1, what='the redelivery admitted')
        assert (await _post(bot, _make_chat_envelope())).status == 200
        assert _admissions(beta) == 1, 'the event was admitted more than once'
        # The redelivery reads the DM thread as the first delivery did (top-level, not a side thread).
        assert bot._thread_count_store.get(SPACE, THREAD) == 1
    finally:
        _close(runner, homes)


@pytest.mark.asyncio
async def test_a_pubsub_event_for_a_parked_profile_is_still_refused_in_chat(tmp_path, monkeypatch):
    """Pub/Sub acks on hand-off, so ``_dispatch_message`` (what it submits) tells the sender instead."""
    homes, runner, bot = await _shared_chat_bot_with_beta_parked(tmp_path, monkeypatch)
    try:
        await bot._dispatch_message(*bot._prepare_inbound(_make_chat_envelope(), 'google.workspace.chat.message.v1.created'))
        await asyncio.gather(*list(bot._background_tasks), return_exceptions=True)
        assert bot.send.await_count == 1
        assert bot._dedup.contains(MESSAGE)
        assert all(_admissions(home) == 0 for _, home in homes)
    finally:
        _close(runner, homes)
