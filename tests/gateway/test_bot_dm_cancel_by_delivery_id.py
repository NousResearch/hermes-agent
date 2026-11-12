"""A ``message_agent`` sender's cancel by delivery id reaches the delivery's one automatic retry.

``gateway/session_bot_cancel.py`` decides on the owner, under the mailbox lock that admits the
retry (``gateway/session_bot.py::_maybe_retry``), so a cancel landing after the first attempt
failed but before its retry ran withdraws that retry for good instead of reporting "already
finished" while the retry goes on to run.
"""
import asyncio

import pytest

from hermes_state_runtime import RuntimeStoreError
from tests.gateway import test_session_bot_retry as retry_suite
from tests.gateway.test_session_bot_retry import KEY, _admissions, _settle_quietly

bot = retry_suite.bot  # the retry suite's owner fixture: a real authority and Bot Chat

AUTHOR = {'id': 'bot:scribe', 'name': 'scribe', 'is_bot': True}


@pytest.mark.asyncio
async def test_cancel_after_a_transient_failure_withdraws_the_retry_for_good(bot, monkeypatch):
    from gateway.session_bot import deliver, recover_bot_deliveries
    from gateway.session_bot_cancel import cancel
    from tools.bot_live_delivery import read_delivery_result
    admit = bot.authority.admit_automation
    blocked, full = [], [True]

    async def full_for_the_retry(adapter, event, identity):
        if identity.endswith(':retry') and full[0]:
            blocked.append(identity)  # the retry is owed but not admitted yet: the cancel's window
            raise RuntimeStoreError('queue_full')
        return await admit(adapter, event, identity)

    monkeypatch.setattr(bot.authority, 'admit_automation', full_for_the_retry)
    bot.errors[:] = ['Error code: 429 - rate limit exceeded']
    await deliver(bot.connection, dict(id=KEY, profile='default', message='ping', author=AUTHOR))
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: blocked, timeout=10, what='the owner to owe the retry')
    await _settle_quietly(bot)

    stranger = await asyncio.gather(cancel(bot.connection, dict(id=KEY, profile='default', author_id='bot:other')),
                                    return_exceptions=True)
    answer = await cancel(bot.connection, dict(id=KEY, profile='default', author_id='bot:scribe'))
    # The queue has room again: a wake or an owner restart would admit the owed retry now.
    full[0] = False
    await recover_bot_deliveries(bot.authority)
    await _settle_quietly(bot)

    assert isinstance(stranger[0], RuntimeStoreError) and stranger[0].reason == 'permission_denied'
    assert answer == {'delivery_id': KEY, 'status': 'cancelled', 'receipt_status': 'cancelled'}
    assert [row['request_id'] for row in _admissions(bot)] == ['bot:' + KEY]  # the retry never ran
    assert (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['status'] == 'cancelled'
