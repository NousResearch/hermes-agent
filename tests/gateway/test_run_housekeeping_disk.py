"""Gateway low-disk auto-prune question on a home channel that cannot take it."""

import asyncio
import logging
import shutil
import threading
import time
from collections import namedtuple
from types import SimpleNamespace

import pytest

from gateway import run_housekeeping_disk as rhd
from gateway.config import Platform
from hermes_cli import prune_advisor

_Usage = namedtuple("_Usage", "total used free")
MINUTE, HOUR, DAY = 60, 3600, 86400
GIB = prune_advisor.GIB
LOW_DISK = _Usage(100 * GIB, 99 * GIB, GIB)
ROOMY_DISK = _Usage(100 * GIB, 10 * GIB, 90 * GIB)


@pytest.fixture(autouse=True)
def _fresh_holds(monkeypatch):
    monkeypatch.setattr(rhd, "_GATEWAY_HOLD", {}, raising=False)


@pytest.fixture
def disk(monkeypatch):
    usage = [LOW_DISK]
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage[0])
    return usage


@pytest.fixture
def home(tmp_path, monkeypatch, disk):
    from hermes_state import SessionDB
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    SessionDB(db_path=h / "state.db").close()
    return h


@pytest.fixture
def clock(monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


@pytest.fixture
def gateway_loop():
    lp = asyncio.new_event_loop()
    thread = threading.Thread(target=lp.run_forever, daemon=True)
    thread.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    thread.join(timeout=5)
    lp.close()


def _runner(outcome):
    """*outcome* is what each send returns, or raises when it is an exception."""
    sends = []

    async def send(platform, home_channel, transport, message, failure_fmt):
        sends.append(home_channel.chat_id)
        if isinstance(runner.outcome, Exception):
            raise runner.outcome
        return runner.outcome

    runner = SimpleNamespace(
        outcome=outcome, home_channel=SimpleNamespace(chat_id="c1", thread_id=None),
        _send_home_channel_message=send, _served_profile_homes={},
        _served_home_channel_transports=lambda: iter(
            [(None, Platform.TELEGRAM, None, runner.home_channel, SimpleNamespace(adapter=None))]))
    return runner, sends


def _tick(runner, gateway_loop):
    from gateway.run import _housekeeping_chore
    _housekeeping_chore("Auto-prune low-disk prompt", rhd.maybe_post_auto_prune_prompt, runner, gateway_loop)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == rhd.__name__ and r.levelno == logging.WARNING]


def test_unreachable_home_channel_is_asked_again_after_an_hour_then_daily(home, clock, gateway_loop, caplog):
    runner, sends = _runner(False)
    with caplog.at_level(logging.INFO, logger=rhd.__name__):
        _tick(runner, gateway_loop)
        clock[0] += MINUTE
        _tick(runner, gateway_loop)
        assert len(sends) == 1
        clock[0] += HOUR
        _tick(runner, gateway_loop)
        assert len(sends) == 2
        clock[0] += HOUR
        _tick(runner, gateway_loop)
        assert len(sends) == 2  # a second failure in a row waits a day
        clock[0] += DAY
        _tick(runner, gateway_loop)
        assert len(sends) == 3
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "/sethome" in warnings[0] and "warning notifications" in warnings[0]
    assert not prune_advisor.snoozed(home)


def test_changing_the_home_channel_asks_on_the_next_tick(home, clock, gateway_loop):
    runner, sends = _runner(False)
    _tick(runner, gateway_loop)
    runner.home_channel = SimpleNamespace(chat_id="c2", thread_id=None)  # /sethome
    clock[0] += MINUTE
    _tick(runner, gateway_loop)
    assert sends == ["c1", "c2"]


@pytest.mark.parametrize("ends_by", ["disk_freed", "question_delivered"])
def test_a_later_episode_starts_again_with_an_hour_and_a_warning(home, clock, gateway_loop, disk, caplog, ends_by):
    runner, sends = _runner(False)
    with caplog.at_level(logging.WARNING, logger=rhd.__name__):
        _tick(runner, gateway_loop)
        clock[0] += HOUR
        _tick(runner, gateway_loop)
        assert len(sends) == 2  # two failures in a row: the next wait is a day
        clock[0] += DAY
        if ends_by == "disk_freed":
            disk[0] = ROOMY_DISK
            _tick(runner, gateway_loop)
            clock[0] += 30 * DAY
            disk[0] = LOW_DISK
        else:
            runner.outcome = True
            _tick(runner, gateway_loop)
            clock[0] += prune_advisor.SNOOZE_SECONDS + 1
            runner.outcome = False
        sent_before = len(sends)
        _tick(runner, gateway_loop)
        clock[0] += HOUR
        _tick(runner, gateway_loop)
    assert len(sends) == sent_before + 2
    assert len(_warnings(caplog)) == 2


def test_delivered_question_waits_a_week_even_when_the_snooze_cannot_be_saved(home, clock, gateway_loop, monkeypatch):
    runner, sends = _runner(True)
    _tick(runner, gateway_loop)
    assert len(sends) == 1 and prune_advisor.snoozed(home)
    clock[0] += prune_advisor.SNOOZE_SECONDS + 1

    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(prune_advisor, "snooze", disk_full)
    _tick(runner, gateway_loop)
    clock[0] += MINUTE
    _tick(runner, gateway_loop)
    assert len(sends) == 2


@pytest.mark.parametrize("case", ["warnings_off", "send_raises"])
def test_attempt_that_sent_nothing_waits_without_re_reading_the_store(home, clock, gateway_loop, monkeypatch, case):
    import gateway.warning_notifications as wn
    if case == "warnings_off":
        monkeypatch.setattr(wn, "warning_notifications_enabled", lambda *_a, **_k: False)
        runner, sends = _runner(True)
    else:
        runner, sends = _runner(RuntimeError("adapter crashed"))
    estimates = []
    real_advise = prune_advisor.advise
    monkeypatch.setattr(prune_advisor, "advise", lambda h: estimates.append(h) or real_advise(h))
    _tick(runner, gateway_loop)
    clock[0] += MINUTE
    _tick(runner, gateway_loop)
    assert len(estimates) == 1
    assert len(sends) == (0 if case == "warnings_off" else 1)
