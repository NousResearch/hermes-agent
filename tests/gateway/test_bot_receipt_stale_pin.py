"""A pre-admission pin left on disk by an older build (or by a transient refusal on the legacy
migration path) whose ``bot:<key>`` admission never committed is withdrawn by the owner's scan
once past the grace window, so the scan stops warning and a same-id resend admits (#139138).
A pin inside the grace window, or one whose admission committed, is left as it is."""
import logging
import os
import time

import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_bot_receipt_refusal import _refuse, _scan_warnings

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'f' * 32
HOUR = 3600


def _pin(bot, key=KEY, **extra):
    """The pin ``_admit`` writes before it admits, as an older build left it after a refusal."""
    from tools.bot_live_delivery import _locked, _write
    with _locked(bot.home) as root:
        path = root / f'{key}.json'
        _write(path, dict(extra, delivery_id=key, profile_home=str(bot.home.resolve()), session_id=bot.chat,
                          principal_id=bot.connection.actor.subject, message='secret text', status='ambiguous'))
    return path


def _age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


def _rows(bot):
    return [row['request_id'] for row in
            list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)]


@pytest.mark.asyncio
async def test_an_old_pin_with_no_admission_is_withdrawn_and_the_resend_admits_once(bot, caplog):
    from gateway.session_bot import deliver, recover_bot_deliveries
    path = _pin(bot)
    _age(path, HOUR)
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert not path.exists()
    assert not _scan_warnings(caplog, KEY)
    withdrawn = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and KEY in r.getMessage()]
    assert len(withdrawn) == 1 and 'secret text' not in withdrawn[0], withdrawn
    receipt = await deliver(bot.connection, dict(id=KEY, profile='default', message='secret text'))
    assert receipt['admission_id'] and not receipt.get('refused'), receipt
    await deliver(bot.connection, dict(id=KEY, profile='default', message='secret text'))
    assert _rows(bot) == ['bot:' + KEY]


@pytest.mark.asyncio
async def test_a_pin_inside_the_grace_window_is_left_alone(bot, caplog):
    """An ``_admit`` between its pin and its commit must never lose its pin."""
    from gateway.session_bot import deliver, recover_bot_deliveries
    path = _pin(bot)
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert path.exists()
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='secret text'))
    assert _rows(bot) == []


@pytest.mark.asyncio
async def test_an_old_pin_whose_admission_committed_stays_and_still_warns(bot, monkeypatch, caplog):
    from gateway.session_bot import deliver, recover_bot_deliveries
    from tools.bot_live_delivery import read_delivery_result
    real = type(bot.authority).admit_automation
    async def committed_then_refused(self, *args):
        await real(self, *args)
        raise RuntimeStoreError('stale_epoch')
    monkeypatch.setattr(type(bot.authority), 'admit_automation', committed_then_refused)
    bot.authority._schedule = lambda ref: None
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    before = read_delivery_result(bot.home, KEY)
    _age(bot.home / 'runtime' / 'bot_live_delivery' / f'{KEY}.json', HOUR)
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert read_delivery_result(bot.home, KEY) == before
    assert before['status'] == 'ambiguous' and 'admission_id' not in before
    assert _scan_warnings(caplog, KEY)
    assert _rows(bot) == ['bot:' + KEY]


OTHER = 'e' * 32


def _legacy(bot, key, sequence, message):
    from tools.bot_live_delivery import _locked, _write
    ticket = dict(id=key, delivery_id=key, status='queued', created_at=sequence, sequence=sequence, message=message,
                  owner=dict(profile_home=str(bot.home.resolve()), session_id=bot.chat, lease_id='departed',
                             live_session_id='old-native'))
    with _locked(bot.home) as root:
        _write(root / f'{key}.json', ticket)
        return root / f'{key}.json'


@pytest.mark.asyncio
async def test_a_legacy_migration_pin_left_by_a_transient_refusal_is_requeued_and_admitted_once(
        bot, monkeypatch, caplog):
    """``_migrate`` keeps the pin on an owner-state refusal (``legacy=``), and that pin is the only
    copy of a never-run message: a later owner scan past the grace window restores the queued
    ticket (never deletes it), and ``_migrate`` admits it once, ahead of the ticket queued after it."""
    from gateway.session_bot import recover_bot_deliveries
    from tools.bot_live_delivery import _locked, _read
    path = _legacy(bot, KEY, 1, 'never-run text')
    _legacy(bot, OTHER, 2, 'later')
    refusing = _refuse(monkeypatch, 'runtime_draining')
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await recover_bot_deliveries(bot.authority)
    with _locked(bot.home):
        pin = _read(path) or {}
    assert pin['status'] == 'ambiguous' and 'admission_id' not in pin and 'owner' in pin, pin
    refusing.clear()
    _age(path, HOUR)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert path.exists()
    assert not _scan_warnings(caplog, KEY)
    assert not [r for r in caplog.records if 'never-run text' in r.getMessage()]
    assert _rows(bot) == ['bot:' + KEY, 'bot:' + OTHER]
    with _locked(bot.home):
        receipt = _read(path) or {}
    assert receipt['message'] == 'never-run text' and receipt['admission_id'], receipt
    assert receipt['sequence'] == 1 and receipt['owner']['lease_id'] == 'departed', receipt
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY, 'bot:' + OTHER]


@pytest.mark.asyncio
async def test_a_scan_outside_the_owner_withdraws_nothing(bot, caplog):
    """The bare mailbox scan has no authority to ask for the admission row: it only reads."""
    from gateway.session_bot_mailbox import scan_receipts
    path = _pin(bot)
    _age(path, HOUR)
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await scan_receipts(bot.home)
    assert path.exists()
    assert _scan_warnings(caplog, KEY)



@pytest.mark.asyncio
async def test_a_committed_pin_on_a_deleted_bot_chat_is_kept_and_the_resend_does_not_run_twice(bot, monkeypatch):
    """A delete moves the admission rows to retired receipts keyed by the route principal, which
    the pin never recorded: a deleted target must read as "may have committed", never withdrawn."""
    import asyncio
    from gateway.session_bot import deliver, recover_bot_deliveries
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    real = type(bot.authority).admit_automation
    async def committed_then_refused(self, *args):
        await real(self, *args)
        raise RuntimeStoreError('stale_epoch')
    monkeypatch.setattr(type(bot.authority), 'admit_automation', committed_then_refused)
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='run once'))
    monkeypatch.setattr(type(bot.authority), 'admit_automation', real)
    for _ in range(250):
        rows = list_session_admissions(bot.authority.db, session_id=bot.chat, pending_only=False)
        if rows and all(row['status'] == 'terminal' for row in rows):
            break
        await asyncio.sleep(0.02)
    assert bot.authority.db.delete_session(bot.chat)
    path = bot.home / 'runtime' / 'bot_live_delivery' / f'{KEY}.json'
    _age(path, HOUR)
    await recover_bot_deliveries(bot.authority)
    assert path.exists()
    chat = create_local_session(bot.authority, bot.connection.actor, {
        'request_id': 'bot2', 'source': 'gui', 'cwd': str(bot.home), 'model': 'fixture', 'toolsets': []})
    title_new_session(bot.authority, chat, 'Bot Chat')
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        await deliver(bot.connection, dict(id=KEY, profile='default', message='run once'))
    assert list_session_admissions(bot.authority.db, session_id=chat.session_id, pending_only=False) == []
