"""Legacy Bot mailbox tickets keep FIFO order across a pre-admission pin (#139138).

A legacy ticket whose ``_migrate`` admission was refused for owner state stays a pin until the
owner scan requeues it past the grace window. While that pin is unresolved no legacy ticket
ordered after it is admitted; tickets ordered before it, and pins whose admission committed,
never hold the queue."""
import pytest

from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_bot_receipt_refusal import _refuse
from tests.gateway.test_bot_receipt_stale_pin import HOUR, KEY, OTHER, _age, _legacy, _pin, _rows

pytest_plugins = ['tests.gateway.test_session_bot_retry']

THIRD = 'd' * 32


def _status(bot, key):
    from tools.bot_live_delivery import _locked, _read
    with _locked(bot.home) as root:
        return (_read(root / f'{key}.json') or {}).get('status')


@pytest.mark.asyncio
async def test_a_ticket_queued_after_an_unresolved_pin_waits_for_it(bot, monkeypatch):
    """Recovery inside the grace window admits nothing behind the pin; past it the order holds."""
    from gateway.session_bot import recover_bot_deliveries
    bot.authority._schedule = lambda ref: None
    path = _legacy(bot, KEY, 1, 'first')
    _legacy(bot, OTHER, 2, 'second')
    refusing = _refuse(monkeypatch, 'runtime_draining')
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await recover_bot_deliveries(bot.authority)
    refusing.clear()
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == []
    assert (_status(bot, KEY), _status(bot, OTHER)) == ('ambiguous', 'queued')
    _age(path, HOUR)
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY, 'bot:' + OTHER]


@pytest.mark.asyncio
async def test_a_ticket_queued_before_an_unresolved_pin_is_still_admitted(bot):
    from gateway.session_bot import recover_bot_deliveries
    bot.authority._schedule = lambda ref: None
    _legacy(bot, KEY, 1, 'first')
    _pin(bot, OTHER, id=OTHER, created_at=2, sequence=2, owner=dict(
        profile_home=str(bot.home.resolve()), session_id=bot.chat, lease_id='departed', live_session_id='old-native'))
    _legacy(bot, THIRD, 3, 'third')
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY]
    assert (_status(bot, OTHER), _status(bot, THIRD)) == ('ambiguous', 'queued')


@pytest.mark.asyncio
async def test_a_pin_whose_admission_committed_does_not_hold_the_queue(bot, monkeypatch):
    from gateway.session_bot import recover_bot_deliveries
    bot.authority._schedule = lambda ref: None
    real, failing = type(bot.authority).admit_automation, [True]
    async def committed_then_refused(self, *args):
        receipt = await real(self, *args)
        if failing:
            raise RuntimeStoreError('stale_epoch')
        return receipt
    monkeypatch.setattr(type(bot.authority), 'admit_automation', committed_then_refused)
    _legacy(bot, KEY, 1, 'first')
    _legacy(bot, OTHER, 2, 'second')
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        await recover_bot_deliveries(bot.authority)
    assert _status(bot, KEY) == 'ambiguous'
    failing.clear()
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY, 'bot:' + OTHER]


@pytest.mark.asyncio
async def test_an_owner_state_refusal_mid_pass_stops_the_pass(bot, monkeypatch):
    """The refused ticket becomes the barrier: nothing after it runs until it is requeued."""
    from gateway.session_bot import recover_bot_deliveries
    bot.authority._schedule = lambda ref: None
    real, refusing = type(bot.authority).admit_automation, [True]
    async def full_for_other(self, adapter, event, identity):
        if refusing and identity == 'bot:' + OTHER:
            raise RuntimeStoreError('queue_full')
        return await real(self, adapter, event, identity)
    monkeypatch.setattr(type(bot.authority), 'admit_automation', full_for_other)
    _legacy(bot, KEY, 1, 'first')
    path = _legacy(bot, OTHER, 2, 'second')
    _legacy(bot, THIRD, 3, 'third')
    with pytest.raises(RuntimeStoreError, match='queue_full'):
        await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY]
    assert (_status(bot, OTHER), _status(bot, THIRD)) == ('ambiguous', 'queued')
    refusing.clear()
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY]
    _age(path, HOUR)
    await recover_bot_deliveries(bot.authority)
    assert _rows(bot) == ['bot:' + KEY, 'bot:' + OTHER, 'bot:' + THIRD]


@pytest.mark.asyncio
async def test_a_delivery_waits_behind_an_unresolved_legacy_pin(bot, monkeypatch):
    """The delivery-triggered ``_migrate`` holds the same barrier as owner recovery."""
    from gateway.session_bot import deliver, recover_bot_deliveries
    bot.authority._schedule = lambda ref: None
    _legacy(bot, KEY, 1, 'first')
    _legacy(bot, OTHER, 2, 'second')
    refusing = _refuse(monkeypatch, 'runtime_draining')
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        await recover_bot_deliveries(bot.authority)
    refusing.clear()
    await deliver(bot.connection, dict(id=THIRD, profile='default', message='fresh'))
    assert 'bot:' + OTHER not in _rows(bot)
    assert (_status(bot, KEY), _status(bot, OTHER)) == ('ambiguous', 'queued')
