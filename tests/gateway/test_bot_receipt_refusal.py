"""A Bot Chat delivery the FIFO definitely refused (nothing committed, e.g. ``permission_denied``)
turns its pre-admission pin into a terminal ``failed`` receipt the bulk scan reads without a
warning and never re-sends (#139138). A pin with no such proof stays a malformed record."""
import logging

import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'e' * 32


def _refuse(monkeypatch, reason):
    """Refuse at the snapshot check, before any FIFO write, until the returned switch is cleared."""
    from gateway import session_automation
    real, refusing = session_automation.snapshot_automation, [True]
    def snapshot(*args, **kwargs):
        if refusing:
            raise RuntimeStoreError(reason)
        return real(*args, **kwargs)
    monkeypatch.setattr(session_automation, 'snapshot_automation', snapshot)
    return refusing


def _scan_warnings(caplog, key):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and key in r.getMessage()]


@pytest.mark.asyncio
async def test_refused_delivery_leaves_a_terminal_receipt_the_scan_accepts(bot, monkeypatch, caplog):
    from gateway.session_bot import deliver, recover_bot_deliveries
    from gateway.session_bot_mailbox import scan_receipts
    from tools.bot_live_delivery import read_delivery_result
    params = dict(id=KEY, profile='default', message='ping')
    refusing = _refuse(monkeypatch, 'permission_denied')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await deliver(bot.connection, params)
    receipt = read_delivery_result(bot.home, KEY)
    assert (receipt['status'], receipt['reason']) == ('failed', 'agent_blocked'), receipt
    assert 'admission_id' not in receipt and 'permission' not in receipt['error']
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        _, records = await scan_receipts(bot.home)
        await recover_bot_deliveries(bot.authority)
    assert [record['delivery_id'] for _, record in records] == [KEY]
    assert not _scan_warnings(caplog, KEY)
    # The same id answers its refusal receipt; it is never re-sent, not even once the refusal is gone.
    refusing.clear()
    again = await deliver(bot.connection, params)
    assert (again['status'], again['reason']) == ('failed', 'agent_blocked'), again
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []


@pytest.mark.parametrize('reason', ['invalid_params', 'admission_conflict'])
@pytest.mark.asyncio
async def test_a_final_refusal_answers_every_resend_with_its_receipt(bot, monkeypatch, reason):
    """A refusal about the delivery itself is final for that id, like ``permission_denied``."""
    from gateway.session_bot import deliver
    params = dict(id=KEY, profile='default', message='ping')
    refusing = _refuse(monkeypatch, reason)
    with pytest.raises(RuntimeStoreError, match=reason):
        await deliver(bot.connection, params)
    refusing.clear()
    again = await deliver(bot.connection, params)
    assert (again['status'], again['refused'], again['reason']) == ('failed', True, 'agent_blocked'), again
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []


@pytest.mark.parametrize('reason', ['runtime_draining', 'storage_unavailable', 'stale_epoch'])
@pytest.mark.asyncio
async def test_a_transient_refusal_leaves_the_same_id_resendable(bot, monkeypatch, reason):
    """An owner-state refusal (a drain for restart/update, storage, a fenced epoch) says nothing
    about the delivery: no receipt is kept, and the same-id resend once it clears admits once."""
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import read_delivery_result
    params = dict(id=KEY, profile='default', message='ping')
    refusing = _refuse(monkeypatch, reason)
    with pytest.raises(RuntimeStoreError, match=reason):
        await deliver(bot.connection, params)
    assert read_delivery_result(bot.home, KEY) is None
    refusing.clear()
    again = await deliver(bot.connection, params)
    assert not again.get('refused') and again['admission_id'], again
    rows = list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)
    assert [row['request_id'] for row in rows] == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_a_cron_report_parked_across_a_drain_is_delivered_on_the_next_tick(bot, monkeypatch):
    """``cron/bot_chat_legacy`` keeps a record queued when the door raises and replays the same id
    next tick: a drain during a gateway restart must not turn into a transferred refusal."""
    import asyncio
    import json
    from cron.bot_chat_legacy import drain_legacy_pending
    from gateway.session_bot import deliver
    from tools import bot_live_delivery as mailbox
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(mailbox, 'authority_delivery', lambda home, params: asyncio.run_coroutine_threadsafe(
        deliver(bot.connection, params), loop).result())
    monkeypatch.setattr(mailbox, 'find_canonical_live_owner', lambda home: dict(
        profile_home=str(bot.home.resolve()), session_id='', lease_id='fixture', live_session_id=''))
    root = bot.home / 'cron' / 'bot_chat_pending'
    root.mkdir(parents=True)
    path = root / f'{KEY}.json'
    path.write_text(json.dumps(dict(id=KEY, status='queued', job={'id': 'digest', 'name': 'Digest'},
                                    content='the report', home=str(bot.home), for_failure=False, sequence=1)))
    refusing = _refuse(monkeypatch, 'runtime_draining')
    await asyncio.to_thread(drain_legacy_pending)
    assert json.loads(path.read_text())['status'] == 'queued'
    refusing.clear()
    await asyncio.to_thread(drain_legacy_pending)
    record = json.loads(path.read_text())
    assert record['status'] == 'transferred' and record['receipt_status'] != 'failed', record
    rows = list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)
    assert [row['request_id'] for row in rows] == ['bot:' + KEY]


def _rpc_connection(bot):
    """The relay Desktop's own connection: ``bot_relay.deliver`` crosses the contract-checked dispatch."""
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(bot.authority, object(), {
        'user_id': 'uid:1000', 'provider': 'local', 'profile_id': bot.authority.profile_id,
        'instance_id': bot.authority.instance_id,
        'capabilities': ['session:read', 'session:submit', 'session:control', 'session:create']})


@pytest.mark.asyncio
async def test_a_refused_delivery_answers_through_the_deliver_contract(bot, monkeypatch):
    """The first refusal is a JSON-RPC refusal; the same-id resend is the declared refused result
    (terminal ``failed``, ``refused``, no admission), never a ``ContractViolation``."""
    connection = _rpc_connection(bot)
    # The fixture's profile_id is no home path: route ``profile`` to this authority, as for a real home.
    monkeypatch.setattr(connection, '_sibling_for', lambda profile: None)
    request = {'id': 1, 'method': 'bot_relay.deliver', 'params': dict(id=KEY, profile='default', message='ping')}
    refusing = _refuse(monkeypatch, 'permission_denied')
    first = await connection.dispatch(request)
    assert first['error']['data']['reason'] == 'permission_denied', first
    refusing.clear()
    again = await connection.dispatch(request)
    assert 'result' in again, again
    result = again['result']
    assert (result['status'], result['refused'], result['reason']) == ('failed', True, 'agent_blocked'), result
    assert result['delivery_id'] == KEY and 'admission_id' not in result
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []


AUTHOR = {'id': 'bot:cto', 'name': 'cto', 'is_bot': True}


@pytest.mark.parametrize('changed', [
    {'author': {'id': 'bot:cfo', 'name': 'cfo', 'is_bot': True}},
    {'notification_category': 'diagnostic'},
    {'message': 'pong'},
])
@pytest.mark.asyncio
async def test_a_refused_receipt_keeps_the_whole_delivery_identity(bot, monkeypatch, changed):
    """The refused receipt answers only the delivery it refused: the same id with another sender,
    category or message conflicts, exactly as an admission-backed receipt does."""
    from gateway.session_bot import deliver
    params = dict(id=KEY, profile='default', message='ping', author=AUTHOR)
    refusing = _refuse(monkeypatch, 'permission_denied')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await deliver(bot.connection, params)
    refusing.clear()
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await deliver(bot.connection, {**params, **changed})
    again = await deliver(bot.connection, params)
    assert (again['status'], again['reason']) == ('failed', 'agent_blocked'), again
    assert list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False) == []


@pytest.mark.asyncio
async def test_a_pin_without_refusal_proof_still_warns(bot, monkeypatch, caplog):
    """A refusal raised after the admission row committed is not a refusal: the pin stays the
    explicit unknown record, and the bulk scan keeps warning about it."""
    from gateway.session_bot import deliver
    from gateway.session_bot_mailbox import scan_receipts
    from tools.bot_live_delivery import read_delivery_result
    real = type(bot.authority).admit_automation
    async def committed_then_refused(self, *args):
        await real(self, *args)
        raise RuntimeStoreError('stale_epoch')
    monkeypatch.setattr(type(bot.authority), 'admit_automation', committed_then_refused)
    bot.authority._schedule = lambda ref: None
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    receipt = read_delivery_result(bot.home, KEY)
    assert receipt['status'] == 'ambiguous' and 'admission_id' not in receipt, receipt
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await scan_receipts(bot.home)
    assert _scan_warnings(caplog, KEY)
