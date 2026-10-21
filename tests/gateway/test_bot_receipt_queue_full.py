"""A Bot Chat delivery the FIFO refuses ``queue_full`` withdraws its pre-admission pin, so the
same-id retry admits once the backlog drains instead of answering ``unknown_execution`` forever."""
import contextlib

import pytest

from gateway.session_contract import Principal, SessionRef
from hermes_state_runtime import (ADMISSION_QUEUE_MAX, RuntimeStoreError, admit_session_input,
                                  list_session_admissions)

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'b' * 32


@pytest.mark.asyncio
async def test_queue_full_refusal_withdraws_the_pin_and_the_retry_admits(bot):
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result
    authority = bot.authority
    authority._schedule = lambda ref: None  # hold the backlog: nothing drains until we cancel it
    fillers = [admit_session_input(authority.db, epoch=authority.epoch, principal_id='uid:1000',
               session_id=bot.chat, request_id=f'fill{i}', payload={'text': f'f{i}'})
               for i in range(ADMISSION_QUEUE_MAX)]
    with pytest.raises(RuntimeStoreError, match='queue_full'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    assert read_delivery_result(bot.home, KEY) is None
    control = Principal('uid:1000', 'default', frozenset({'session:submit', 'session:read', 'session:control'}), 'native')
    for row in fillers:
        with contextlib.suppress(RuntimeStoreError):
            await authority.cancel_queued(control, SessionRef('default', bot.chat), row['admission_id'])
    retried = await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    assert 'error' not in retried and retried['admission_id']
    assert [r['request_id'] for r in list_session_admissions(authority.db, session_id=bot.chat)] == ['bot:' + KEY]
