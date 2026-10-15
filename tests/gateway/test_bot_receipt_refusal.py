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
