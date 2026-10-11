"""A retired profile authority refuses every verb that writes the home it released, from a stale
connection: its ownership is released, but until a successor advances the runtime epoch its epoch
would still pass the store's stale-epoch fence, and the Bot relay files carry no epoch at all.
Table-driven over the in-class sites of SWEEP_retirement-fence.md."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway import run_runtime, session_bot, session_cron
from gateway.session_contract import Principal, Submission
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_profile_retire_stops_turns import REF, _Agent, _authority

CRON = Principal('cron-owner', 'owned', frozenset({'session:submit', 'session:control'}), 'cron-ticker')


async def _cron_cancel(authority, receipt):
    await session_cron.operation(authority, 'cancel', {'session_id': 's', 'admission_id': receipt.admission_id})


def _relay(operation, params):
    async def call(authority, receipt):
        session_bot.relay_operation(SimpleNamespace(authority=authority, actor=CRON), operation, params)
    return call


def _outbox(home):
    box = home / 'bot_relay' / 'outbox'
    box.mkdir(parents=True)
    (box / 'e1.json').write_text(json.dumps({'id': 'e1', 'created_at': 9e12, 'target_handle': 'x'}))


async def _cancel_queued(authority, receipt):
    await authority.cancel_queued(CRON, REF, receipt.admission_id)


async def _submit_more(authority, receipt):
    await authority.submit(CRON, Submission(request_id='cron:j:2', ref=REF, payload={'text': ''}, intent='queue'))


async def _worker_adopt(authority, receipt):
    from gateway.session_worker import worker_request
    connection = SimpleNamespace(authority=authority, actor=CRON)
    await worker_request(connection, REF, {}, operation='adopt')


SITES = {
    'authority.cancel_queued': (_cancel_queued, None, lambda db, home: list_session_admissions(
        db, session_id='s', pending_only=False)[0]['status'] == 'queued'),
    'authority.submit': (_submit_more, None, lambda db, home: len(list_session_admissions(
        db, session_id='s', pending_only=False)) == 1),
    'worker.adopt': (_worker_adopt, None, lambda db, home: list_session_admissions(
        db, session_id='s', pending_only=False)[0]['status'] == 'queued'),
    'cron.cancel': (_cron_cancel, None, lambda db, home: list_session_admissions(
        db, session_id='s', pending_only=False)[0]['status'] == 'queued'),
    'bot_relay.outbox.drain': (_relay('outbox', {}), _outbox,
                               lambda db, home: (home / 'bot_relay' / 'outbox' / 'e1.json').exists()),
    'bot_relay.reply': (_relay('reply', {'id': 'e1', 'reply': 'hi'}), None,
                        lambda db, home: not (home / 'bot_relay').exists()),
    'bot_relay.roster.sync': (_relay('roster', {'agents': []}), None,
                              lambda db, home: not (home / 'bot_relay').exists()),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('site', sorted(SITES))
async def test_retired_authority_refuses_writes_before_successor_epoch(tmp_path, monkeypatch, site):
    call, arrange, untouched = SITES[site]
    db, authority, _execute, _returned = _authority(tmp_path, _Agent())
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # the row stays queued for the next owner
    try:
        with db:
            receipt = await authority.submit(CRON, Submission(request_id='cron:j:1', ref=REF, payload={'text': ''}, intent='queue'))
            if arrange is not None:
                arrange(tmp_path)
            assert await run_runtime._retire_profile_authority(authority, timeout=1.0) is True
            with pytest.raises(RuntimeStoreError, match='runtime_retired'):
                await call(authority, receipt)
            assert untouched(db, tmp_path)
    finally:
        await asyncio.sleep(.05)
        db.close()
