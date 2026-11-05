"""A parked-profile refusal on Discord is an outage notice: it never hides the message from backfill.

Discord's missed-message backfill re-dispatches an input that has no substantive bot reply, and skips
the replies it recognises as outage notices (``_is_down_notice_content``). The refusal replies to the
input it refuses, so unless it reads as an outage notice, the never-admitted message counts as
answered and is not picked up after the profile is served again.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.parked_route_refusal import drop_rejected_route
from tests.gateway import test_discord_missed_message_backfill as discord_tests

adapter = discord_tests.adapter


def _replying_channel(adapter, monkeypatch):
    """``(channel, message)``: the real ``adapter.send`` puts its reply to *message* in the history."""
    channel = discord_tests.FakeChannel()
    message = discord_tests.make_message(channel=channel)

    async def send(*, content, reference, **_kwargs):
        reply = SimpleNamespace(id=100 + len(channel._history_messages), content=content,
                                author=adapter._client.user, reference=SimpleNamespace(message_id=message.id))
        channel._history_messages.append(reply)
        return reply

    channel.send = send
    monkeypatch.setattr(adapter, '_resolve_channel', AsyncMock(return_value=channel))
    monkeypatch.setattr(adapter, '_reply_reference_for_send', lambda *a: SimpleNamespace(message_id=message.id))
    return channel, message


@pytest.mark.asyncio
async def test_a_parked_profile_refusal_leaves_the_message_to_backfill(adapter, monkeypatch):
    channel, message = _replying_channel(adapter, monkeypatch)
    assert await adapter._should_backfill_discord_message(message)
    adapter.gateway_runner = SimpleNamespace(
        session_runtime_descriptor={'parked_profiles': {'beta': 'test park'}},
        _thread_metadata_for_source=lambda source: None)
    source = adapter.build_source('123', chat_type='group', user_id='42', message_id='1')
    source._rejected_route_profile = 'beta'
    drop_rejected_route(adapter, source)
    await asyncio.gather(*list(adapter._background_tasks))
    assert len(channel._history_messages) == 1, 'the refusal was not sent'
    assert not adapter._discord_message_is_persistently_complete('1')
    assert await adapter._should_backfill_discord_message(message), (
        f'the refusal reads as an answer, so the never-admitted message is not backfilled: '
        f'{channel._history_messages[0].content!r}')


@pytest.mark.asyncio
async def test_a_real_final_answer_still_completes_the_message(adapter, monkeypatch):
    channel, message = _replying_channel(adapter, monkeypatch)
    result = await adapter.send('123', 'Here is the summary you asked for.', reply_to='1', metadata={'notify': True})
    assert result.success and len(channel._history_messages) == 1
    assert not await adapter._should_backfill_discord_message(message)
