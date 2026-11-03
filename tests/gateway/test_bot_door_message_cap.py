"""The shared Bot door caps a relayed bot DM like main's relay edge (16,200), never a trusted
producer's report: a cron bot-chat digest over that size is admitted whole."""
import asyncio

import pytest

from tests.waiting import async_wait_until

from gateway.session_contract import Principal
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'e' * 32


@pytest.mark.asyncio
async def test_long_cron_report_is_admitted_and_relayed_dm_stays_capped(bot):
    from cron.scheduler_delivery import bot_chat_message
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result
    report = '\n'.join(f'- item {i}: status ok, details recorded' for i in range(480))
    message = bot_chat_message({'id': 'daily', 'name': 'Daily report'}, report)
    assert len(message) > 16200
    await deliver(bot.connection, dict(id=KEY, profile='default', message=message))
    await async_wait_until(lambda: (read_delivery_result(bot.home, KEY) or {}).get('status') == 'settled',
                           timeout=10, what='the cron report to settle')
    assert (await asyncio.to_thread(read_delivery_result, bot.home, KEY))['message'] == message

    relayed = Principal('uid:1000', 'default', frozenset({'session:submit', 'session:read'}), 'relay')
    author = {'id': 'bot:conn/peer', 'name': 'peer', 'is_bot': True}
    connection = type(bot.connection)(authority=bot.authority, actor=relayed)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await deliver(connection, dict(id='f' * 32, profile='default', message='x' * 16201, author=author))
    ok = await deliver(connection, dict(id='a' * 32, profile='default', message='y' * 16000, author=author))
    assert ok['admission_id']
    rows = list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)
    assert sorted(r['request_id'] for r in rows) == ['bot:' + 'a' * 32, 'bot:' + KEY]
