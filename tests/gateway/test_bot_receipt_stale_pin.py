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


@pytest.mark.asyncio
async def test_a_legacy_migration_pin_left_by_a_transient_refusal_is_cleaned_once_past_grace(
        bot, monkeypatch, caplog):
    """``_migrate`` keeps the pin on an owner-state refusal (``legacy=``): a later owner scan past
    the grace window withdraws it instead of warning about it forever."""
    from gateway.session_bot import recover_bot_deliveries
    from tools.bot_live_delivery import _locked, _read, _write
    home = str(bot.home.resolve())
    with _locked(bot.home) as root:
        path = root / f'{KEY}.json'
        _write(path, dict(id=KEY, delivery_id=KEY, status='queued', created_at=1, sequence=1, message='legacy',
                          owner=dict(profile_home=home, session_id=bot.chat, lease_id='departed',
                                     live_session_id='old-native')))
    refusing = _refuse(monkeypatch, 'runtime_draining')
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await recover_bot_deliveries(bot.authority)
    with _locked(bot.home) as root:
        pin = _read(path)
    assert pin['status'] == 'ambiguous' and 'admission_id' not in pin and 'owner' in pin, pin
    refusing.clear()
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert path.exists()  # still inside the grace window
    _age(path, HOUR)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='gateway.session_bot'):
        await recover_bot_deliveries(bot.authority)
    assert not path.exists()
    assert not _scan_warnings(caplog, KEY)
    assert _rows(bot) == []


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
