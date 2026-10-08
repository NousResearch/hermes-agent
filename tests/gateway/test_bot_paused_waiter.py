"""The legacy Bot watcher survives a recoverable shared FIFO pause."""
import asyncio

import pytest

from tests.gateway.test_session_bot_retry import KEY

pytest_plugins = ['tests.gateway.test_session_bot_retry']


@pytest.mark.asyncio
async def test_paused_bot_follower_rearms_until_its_durable_completion(bot, monkeypatch):
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from tools.bot_live_delivery import read_delivery_result

    authority = bot.authority
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    authority._pause(SessionRef(authority.profile_id, bot.chat), 'session_busy')
    async with asyncio.timeout(5):
        while (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['status'] != 'queued':
            await asyncio.sleep(.01)
    record = (await asyncio.to_thread(read_delivery_result, bot.home, KEY))
    assert record['status'] == 'queued'
    schedule(SessionRef(authority.profile_id, bot.chat))
    await asyncio.wait_for(authority.sessions[bot.chat].task, 5)
    async with asyncio.timeout(5):
        while (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['status'] != 'settled':
            await asyncio.sleep(.01)
    record = (await asyncio.to_thread(read_delivery_result, bot.home, KEY))
    assert record['status'] == 'settled' and record['reply'] == 'pong'
