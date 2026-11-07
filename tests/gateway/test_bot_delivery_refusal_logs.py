"""A pre-admission Bot Chat refusal leaves a WARNING naming the ids it compared (#149563).

The refusal branches in ``gateway/session_bot.py`` raise before ``_admit`` writes a receipt, so
nothing is stored for a refused delivery: the log line is the only trace. A state-based refusal
(a pin left on a dead row, a runner entry that moved to a reset successor) never self-clears, so
without it the whole bot-chat lane of a profile can be dropped for hours with no log to grep.
"""
import logging

import pytest

from hermes_state_runtime import RuntimeStoreError

pytest_plugins = ['tests.gateway.test_session_bot_retry']

KEY = 'e' * 32
LOGGER = 'gateway.session_bot'


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]


def _line(caplog, needle: str) -> str:
    """The first refusal line naming *needle*; ``""`` when the branch logged nothing.

    A bare ``next(...)`` inside an ``async def`` test raises ``StopIteration``, which PEP 479 turns
    into ``RuntimeError``: the missing log line would report as an unrelated coroutine error.
    """
    for record in _warnings(caplog):
        if needle in record.getMessage():
            return record.getMessage()
    return ''


@pytest.mark.asyncio
async def test_a_pin_mismatch_names_the_pinned_and_live_session(bot, caplog):
    """The pair the RCA had to reconstruct by hand: the dead pin and the live runner entry."""
    from gateway.session_bot import deliver
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver(bot.connection, dict(
                id=KEY, profile='default', message='ping', session_id='dead-pinned-session'))
    line = _line(caplog, KEY)
    assert 'dead-pinned-session' in line and bot.chat in line, line


@pytest.mark.asyncio
async def test_a_missing_runner_entry_names_the_session(bot, caplog, monkeypatch):
    """Fewer runner entries than sessions: the door refuses a chat no lane can reach."""
    from gateway.session_bot import deliver
    store = bot.authority.runner.session_store
    monkeypatch.setattr(store, 'lookup_by_session_key', lambda route: None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    line = _line(caplog, bot.chat)
    assert 'runner_entry=None' in line, line


@pytest.mark.asyncio
async def test_a_missing_compression_tip_row_names_the_tip(bot, caplog, monkeypatch):
    """A title whose compression tip points at a row that no longer exists."""
    from gateway.session_bot import deliver
    monkeypatch.setattr(bot.authority.db, 'get_compression_tip', lambda sid: 'ghost-tip-row')
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(RuntimeStoreError, match='not_found'):
            await deliver(bot.connection, dict(id=KEY, profile='default', message='ping'))
    line = _line(caplog, 'ghost-tip-row')
    assert 'not_found' in line, line
