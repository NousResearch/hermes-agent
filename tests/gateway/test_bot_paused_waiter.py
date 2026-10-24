"""The legacy Bot watcher survives a recoverable shared FIFO pause."""
import asyncio

import pytest

from tests.gateway.test_session_bot_retry import KEY
from tests.waiting import async_wait_until

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
    for _ in range(500):  # bounded poll: a slow runner stretches the rounds, never fails early
        if (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['status'] == 'queued':
            break
        await asyncio.sleep(.01)
    record = (await asyncio.to_thread(read_delivery_result, bot.home, KEY))
    assert record['status'] == 'queued'
    schedule(SessionRef(authority.profile_id, bot.chat))
    await asyncio.wait_for(authority.sessions[bot.chat].task, 5)
    for _ in range(500):
        if (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['status'] == 'settled':
            break
        await asyncio.sleep(.01)
    record = (await asyncio.to_thread(read_delivery_result, bot.home, KEY))
    assert record['status'] == 'settled' and record['reply'] == 'pong'


@pytest.mark.asyncio
async def test_paused_watcher_never_publishes_final_failed_while_retry_is_due(bot, monkeypatch):
    """The pause wakes the watcher; the resumed admission then fails transiently before the watcher
    takes the mailbox lock. Its receipt must read pending, never the final ``failed``, while the
    one retry is still owed (a poller would otherwise return that failure as the answer)."""
    from gateway import session_bot
    from gateway.session_bot_mailbox import mailbox_lock
    from gateway.session_contract import SessionRef
    from tests.gateway.test_session_bot_retry import _admissions

    authority = bot.authority
    written = []
    write = session_bot.write_receipt

    async def recording_write(home, record, **kwargs):
        written.append(record.get('status'))
        return await write(home, record, **kwargs)
    monkeypatch.setattr(session_bot, 'write_receipt', recording_write)
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    await session_bot.deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    ref = SessionRef(authority.profile_id, bot.chat)
    gate, held = asyncio.Event(), asyncio.Event()

    async def hold_mailbox():
        async with mailbox_lock(authority):
            held.set()
            await gate.wait()
    holder = asyncio.create_task(hold_mailbox())
    await held.wait()
    authority._pause(ref, 'session_busy')
    # Let the pause reach the watcher: it is parked either on its reply future (kept by a
    # session_busy pause) or, if the pause woke it, on the held mailbox lock.
    _key, _task, future = authority._bot_reply_watchers[KEY]
    await async_wait_until(lambda: not future.done() or getattr(authority._bot_mailbox_lock, '_waiters', None),
                           timeout=5, what='the watcher parked after the pause')
    monkeypatch.setattr(authority, '_schedule', schedule)
    schedule(ref)
    await asyncio.wait_for(authority.sessions[bot.chat].task, 5)
    gate.set()
    await holder
    await async_wait_until(lambda: len(_admissions(bot)) >= 2 and all(r['status'] == 'terminal' for r in _admissions(bot)),
                           timeout=5, what='the retried admission settled')
    for task in list(getattr(authority, '_bot_receipt_tasks', ())):
        await asyncio.gather(task, return_exceptions=True)
    assert 'failed' not in written, written
    assert written[-1] == 'settled'
