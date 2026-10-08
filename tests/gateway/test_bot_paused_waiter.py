"""The legacy Bot watcher survives a recoverable shared FIFO pause."""
import asyncio

import pytest

from tests.gateway.test_session_bot_retry import bot, KEY


@pytest.mark.asyncio
async def test_paused_bot_follower_rearms_until_its_durable_completion(bot, monkeypatch):
    from gateway.session_bot import deliver
    from gateway.session_contract import SessionRef
    from tools.bot_live_delivery import _read, _root

    authority = bot.authority
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    accepted = await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    authority._pause(SessionRef(authority.profile_id, bot.chat), 'session_busy')
    async with asyncio.timeout(5):
        while _read(_root(bot.home) / f'{KEY}.json')['status'] != 'queued':
            await asyncio.sleep(.01)
    record = _read(_root(bot.home) / f'{KEY}.json')
    assert record['status'] == 'queued'
    schedule(SessionRef(authority.profile_id, bot.chat))
    await asyncio.wait_for(authority.sessions[bot.chat].task, 5)
    async with asyncio.timeout(5):
        while _read(_root(bot.home) / f'{KEY}.json')['status'] != 'settled':
            await asyncio.sleep(.01)
    record = _read(_root(bot.home) / f'{KEY}.json')
    assert record['status'] == 'settled' and record['reply'] == 'pong'
