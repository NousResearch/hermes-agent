"""Startup cancellation cannot release ownership before mailbox I/O physically ends."""
import asyncio
import threading

import pytest

pytest_plugins = ('tests.gateway.test_session_bot_retry',)


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_count', [1, 3])
async def test_cancelled_startup_recovery_retains_owner_until_receipt_writer_finishes(bot, monkeypatch, cancel_count):
    from gateway import run_runtime, session_bot_mailbox
    from gateway.session_bot import deliver, recover_bot_deliveries
    from tests.gateway.test_session_bot_retry import KEY, _settled

    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    await _settled(bot, 1)
    watchers = list(getattr(bot.authority, '_bot_receipt_tasks', ()))
    if watchers:
        await asyncio.gather(*watchers)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = session_bot_mailbox._write_receipt

    def held_write(*args):
        entered.set()
        try:
            assert release.wait(10), 'test did not release receipt writer'
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(session_bot_mailbox, '_write_receipt', held_write)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 0)
    recovery = asyncio.create_task(recover_bot_deliveries(bot.authority))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(cancel_count):
            recovery.cancel()
            await asyncio.sleep(0)
        assert not finished.is_set()
        # This is the cleanup path of a cancelled serve_profile_runtime startup.
        assert await run_runtime._retire_profile_authority(bot.authority) is False
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        results = await asyncio.gather(recovery, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
    assert await run_runtime._retire_profile_authority(bot.authority) is True
