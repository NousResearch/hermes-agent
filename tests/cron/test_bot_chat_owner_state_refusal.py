"""A cron Bot Chat delivery the authority refuses for its own state (a drain for restart or update,
compression busy, a full FIFO) committed nothing under its delivery id. It is parked and a later
tick admits it exactly once; it is never booked as a failed delivery, which dropped the page for
good (#141890). Real authority and JSON-RPC dispatch with the client's error mapping; only the
socket is skipped."""
import asyncio
import json

import pytest
import pytest_asyncio

from hermes_state_runtime import list_session_admissions
from tests.gateway.test_bot_receipt_refusal import _refuse, _rpc_connection

pytest_plugins = ['tests.gateway.test_session_bot_retry']


@pytest_asyncio.fixture
async def wired(bot, monkeypatch):
    """Route cron's ``bot_relay.deliver`` into the fixture authority through its RPC dispatch."""
    from cron import jobs, scheduler_delivery
    from gateway import config
    from hermes_cli.gateway_client import rpc_error
    from tools import bot_live_delivery as mailbox

    loop = asyncio.get_running_loop()
    connection = _rpc_connection(bot)
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)

    def authority_delivery(home, params):
        request = {'id': 1, 'method': 'bot_relay.deliver', 'params': params}
        reply = asyncio.run_coroutine_threadsafe(connection.dispatch(request), loop).result()
        if 'error' in reply:
            raise rpc_error(reply['error'])
        return reply['result']

    monkeypatch.setattr(mailbox, 'authority_delivery', authority_delivery)
    monkeypatch.setattr(mailbox, 'find_canonical_live_owner', lambda home: dict(
        profile_home=str(bot.home.resolve()), session_id='', lease_id='fixture', live_session_id=''))
    monkeypatch.delenv('_HERMES_CRON_EXTERNAL_WORKER', raising=False)
    monkeypatch.setattr(scheduler_delivery._sched, 'load_config', dict)
    monkeypatch.setattr(config, 'load_gateway_config', lambda: None)
    monkeypatch.setattr(jobs, 'update_job', lambda key, values: None)
    return bot


def _pending(bot):
    return [json.loads(path.read_text()) for path in (bot.home / 'cron' / 'bot_chat_pending').glob('*.json')]


@pytest.mark.parametrize('reason', ['runtime_draining', 'session_busy', 'queue_full'])
@pytest.mark.asyncio
async def test_a_report_refused_for_owner_state_is_parked_then_admitted_once(wired, monkeypatch, reason):
    from cron.bot_chat_legacy import drain_legacy_pending
    from cron.scheduler_delivery import _deliver_result

    bot = wired
    job = dict(id='tea', name='Tea', execution_id='run-1', deliver='bot-chat')
    refusing = _refuse(monkeypatch, reason)
    assert await asyncio.to_thread(_deliver_result, job, 'Reminder: the tea is ready') is None
    receipt = job['last_delivery_queued']['bot-chat:(own)']
    assert receipt['status'] == 'queued', job
    [record] = _pending(bot)
    assert (record['id'], record['status']) == (receipt['delivery_id'], 'queued'), record
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []

    await asyncio.to_thread(drain_legacy_pending)  # still refusing: stays parked, nothing admitted
    assert [r['status'] for r in _pending(bot)] == ['queued']
    refusing.clear()
    await asyncio.to_thread(drain_legacy_pending)
    await asyncio.to_thread(drain_legacy_pending)
    [record] = _pending(bot)
    assert record['status'] == 'transferred' and record['receipt_status'] != 'failed', record
    rows = list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)
    assert [row['request_id'] for row in rows] == ['bot:' + receipt['delivery_id']]
    assert 'Reminder: the tea is ready' in json.dumps(rows[0]['payload'])


@pytest.mark.asyncio
async def test_a_final_refusal_is_still_a_failed_delivery(wired, monkeypatch):
    """A refusal about the delivery itself is final: booked failed, nothing parked or resent."""
    from cron.scheduler_delivery import _deliver_result

    bot = wired
    job = dict(id='tea', name='Tea', execution_id='run-1', deliver='bot-chat')
    _refuse(monkeypatch, 'permission_denied')
    error = await asyncio.to_thread(_deliver_result, job, 'Reminder: the tea is ready')
    assert error and 'permission_denied' in error, error
    assert _pending(bot) == []
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []
