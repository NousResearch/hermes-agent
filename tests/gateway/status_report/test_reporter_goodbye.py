"""The goodbye before a self-suspend, and what it does to the beats around it."""

from __future__ import annotations

import asyncio
import errno
import gc
import logging

import pytest

from gateway import status as gateway_status
from tests.gateway.status_report.support import (
    AFTER_INTERVAL_S,
    FAULT_PATH,
    QUICK_RETRY_S,
    RESPONSES,
    Receiver,
    Reply,
    Watcher,
    a_status_source,
    assert_matches_contract,
    make_reporter,
    settle,
    told,
)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio

GOODBYE_QUIET_S = RESPONSES["goodbye_quiet_s"]
# Past the longest interval a receiver can set, jitter included.
AFTER_THE_LONGEST_INTERVAL_S = RESPONSES["interval_max_s"] * 1.1 + 1


async def test_a_goodbye_is_one_bounded_report_with_no_retry(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)
    await reporter.poll()
    receiver.reply_with(Reply(500))

    reporter.goodbye()
    await settle()
    clock.advance(QUICK_RETRY_S)
    await reporter.poll()
    await settle()

    assert receiver.kinds == ["beat", "goodbye"]
    goodbye = receiver.requests[1]
    assert_matches_contract(goodbye.body, "goodbye")
    assert goodbye.body["seq"] == 2
    assert 2.0 <= goodbye.timeout <= 3.0

async def test_a_goodbye_that_cannot_be_built_is_dropped_without_an_unretrieved_error(clock, caplog):
    caplog.set_level(logging.DEBUG)
    unretrieved: list[str] = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, context: unretrieved.append(context["message"]))
    receiver = Receiver()

    def source():
        if receiver.requests:  # the first beat is out: this read is the goodbye's
            raise OSError("status record unreadable")
        return a_status_source()

    reporter = make_reporter(receiver, source=source)
    await reporter.poll()

    reporter.goodbye()
    await settle()
    gc.collect()  # a task dropped with its exception unread reports it to the loop when collected

    assert receiver.kinds == ["beat"]
    assert unretrieved == []
    assert any(record.levelno == logging.DEBUG and "goodbye" in record.getMessage()
               for record in caplog.records if record.name.startswith("gateway.status_report"))

async def test_no_beat_follows_a_goodbye_until_the_gateway_is_still_running_a_minute_later(clock):
    """A beat between the goodbye and the freeze would tell the receiver the gateway is awake again."""
    receiver = Receiver()
    reporter = make_reporter(receiver)
    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)  # a beat is due

    reporter.goodbye()
    await settle()
    clock.advance(GOODBYE_QUIET_S - 1)
    await reporter.poll()
    assert receiver.kinds == ["beat", "goodbye"]
    clock.advance(1)
    await reporter.poll()

    assert receiver.kinds == ["beat", "goodbye", "beat"]

async def test_the_first_beat_after_a_goodbye_waits_only_the_quiet_minute(clock):
    """A gateway whose machine woke soon after its goodbye, or never froze, says so once the quiet
    minute is over, whatever interval the receiver set: at a told 600 s, a three-minute suspend would
    otherwise leave it reported asleep for minutes after it woke."""
    receiver = Receiver()
    receiver.reply_with(told(RESPONSES["interval_max_s"]))
    reporter = make_reporter(receiver)
    await reporter.poll()
    clock.advance(30)

    reporter.goodbye()
    await settle()
    clock.advance(GOODBYE_QUIET_S - 1)
    await reporter.poll()
    assert receiver.kinds == ["beat", "goodbye"]
    clock.advance(1)
    await reporter.poll()

    assert receiver.kinds == ["beat", "goodbye", "beat"]

def _goodbye_while_the_second_beat_is_in_flight(receiver: Receiver, clock=None, freeze_s: float = 0):
    """A reporter whose suspend path says goodbye while its second beat waits for the reply; with
    *freeze_s*, the machine then freezes for that long before the reply comes back."""
    loop = asyncio.get_running_loop()
    reporter = None

    async def say_goodbye() -> None:
        reporter.goodbye()

    def answer_after_the_goodbye(request, timeout=None):
        try:
            return receiver(request, timeout)  # raises for a non-2xx reply, as urllib does
        finally:
            if receiver.kinds == ["beat", "beat"]:
                asyncio.run_coroutine_threadsafe(say_goodbye(), loop).result(timeout=5)
                assert receiver.goodbye_arrived.wait(timeout=5)
                if freeze_s:
                    clock.advance(freeze_s)

    reporter = make_reporter(answer_after_the_goodbye)
    return reporter

async def test_a_beat_answered_after_the_goodbye_leaves_the_next_beat_at_the_quiet_minute(clock):
    """The suspend path can say goodbye while a beat is still in flight. That beat's reply must not push
    the next beat back out to a told 600 s interval."""
    receiver = Receiver()
    receiver.reply_with(told(RESPONSES["interval_max_s"]), told(RESPONSES["interval_max_s"]))
    reporter = _goodbye_while_the_second_beat_is_in_flight(receiver)
    await reporter.poll()
    clock.advance(RESPONSES["interval_max_s"] * 1.1 + 1)
    await reporter.poll()
    await settle()
    assert receiver.kinds == ["beat", "beat", "goodbye"]

    clock.advance(GOODBYE_QUIET_S + 1)
    await reporter.poll()

    assert receiver.kinds == ["beat", "beat", "goodbye", "beat"]

async def test_a_beat_answered_after_the_machine_froze_and_woke_lets_the_next_go_out_at_once(clock):
    """The machine can freeze with a beat still in flight and wake after the quiet minute: the reply
    that comes back then must not push the first beat after the wake out to a told 600 s."""
    receiver = Receiver()
    receiver.reply_with(told(RESPONSES["interval_max_s"]), told(RESPONSES["interval_max_s"]))
    reporter = _goodbye_while_the_second_beat_is_in_flight(receiver, clock, freeze_s=3 * 60)
    await reporter.poll()
    clock.advance(RESPONSES["interval_max_s"] * 1.1 + 1)
    await reporter.poll()
    await settle()
    assert receiver.kinds == ["beat", "beat", "goodbye"]

    await reporter.poll()

    assert receiver.kinds == ["beat", "beat", "goodbye", "beat"]

async def test_a_beat_that_failed_across_the_freeze_is_not_retried_after_the_goodbye(clock):
    """A quick retry would re-send a report older than the goodbye, which the receiver drops: the
    first beat after the wake is a new report instead."""
    receiver = Receiver()
    receiver.reply_with(Reply(), Reply(500))
    reporter = _goodbye_while_the_second_beat_is_in_flight(receiver, clock, freeze_s=3 * 60)
    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    await settle()
    clock.advance(QUICK_RETRY_S + 1)

    await reporter.poll()

    assert receiver.kinds == ["beat", "beat", "goodbye", "beat"]
    goodbye_seq, wake_seq = (seen.body["seq"] for seen in receiver.requests[2:])
    assert wake_seq > goodbye_seq

@pytest.mark.parametrize("step", ["lock", "status"], ids=["confirming the lock", "reading the status"])
async def test_a_beat_still_being_prepared_when_the_goodbye_goes_out_is_not_sent(clock, monkeypatch, step):
    """The suspend path can say goodbye while the next beat is still confirming the sender lock or
    reading the gateway's status. That beat is superseded: sent, it would tell the receiver the machine
    is awake as it goes to sleep."""
    receiver = Receiver()
    loop = asyncio.get_running_loop()
    calls = {"lock": 0, "status": 0}
    reporter = None

    async def say_goodbye() -> None:
        reporter.goodbye()

    def on_the_second(name):
        calls[name] += 1
        if name == step and calls[name] == 2:
            asyncio.run_coroutine_threadsafe(say_goodbye(), loop).result(timeout=5)

    def read_status() -> dict:
        on_the_second("status")
        return a_status_source()

    def take_the_lock(scope, identity):
        on_the_second("lock")
        return gateway_status.acquire_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    reporter = make_reporter(receiver, source=read_status)
    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    await settle()
    assert receiver.kinds == ["beat", "goodbye"]

    clock.advance(GOODBYE_QUIET_S + 1)
    await reporter.poll()

    assert receiver.kinds == ["beat", "goodbye", "beat"]
    goodbye_seq, wake_seq = (seen.body["seq"] for seen in receiver.requests[1:])
    assert wake_seq > goodbye_seq

async def test_a_beat_whose_status_read_failed_after_the_goodbye_keeps_the_quiet_minute(clock):
    """A beat can fail after a goodbye overtook it. The goodbye set the next beat for the end of
    its quiet minute, and that failure must not push it back out to a told 600 s."""
    receiver = Receiver()
    receiver.reply_with(told(RESPONSES["interval_max_s"]))
    loop = asyncio.get_running_loop()
    reads, reporter = [], None

    async def say_goodbye() -> None:
        reporter.goodbye()

    def source() -> dict:
        reads.append("read")
        if len(reads) == 2:  # the second beat's read: the goodbye goes out, then the read fails
            asyncio.run_coroutine_threadsafe(say_goodbye(), loop).result(timeout=5)
            raise OSError(errno.EIO, "Input/output error", FAULT_PATH)
        return a_status_source()

    reporter = make_reporter(receiver, source=source)
    async with Watcher(reporter) as watcher:
        await async_wait_until(lambda: receiver.answered == 1, timeout=5, what="the first beat")
        clock.advance(AFTER_THE_LONGEST_INTERVAL_S)
        await async_wait_until(lambda: receiver.kinds == ["beat", "goodbye"], timeout=5, what="the goodbye")
        await watcher.polls_on()

        clock.advance(GOODBYE_QUIET_S + 1)

        await async_wait_until(lambda: receiver.kinds == ["beat", "goodbye", "beat"], timeout=5,
                               what="a beat once the quiet minute is over")


async def test_a_beat_that_could_not_confirm_the_lock_after_the_goodbye_keeps_the_quiet_minute(clock, monkeypatch):
    receiver = Receiver()
    receiver.reply_with(told(RESPONSES["interval_max_s"]))
    loop = asyncio.get_running_loop()
    tries, reporter = [], None

    async def say_goodbye() -> None:
        reporter.goodbye()

    def take_the_lock(scope, identity):
        tries.append(scope)
        if len(tries) == 2:  # the second beat's confirmation: the goodbye goes out, then it fails
            asyncio.run_coroutine_threadsafe(say_goodbye(), loop).result(timeout=5)
            return False, None
        return gateway_status.acquire_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    reporter = make_reporter(receiver)
    await reporter.poll()
    clock.advance(AFTER_THE_LONGEST_INTERVAL_S)
    await reporter.poll()
    await settle()
    assert receiver.kinds == ["beat", "goodbye"]

    clock.advance(GOODBYE_QUIET_S + 1)
    await reporter.poll()

    assert receiver.kinds == ["beat", "goodbye", "beat"]


async def test_a_wait_asked_for_by_a_beat_answered_after_the_goodbye_is_kept(clock):
    """The quiet minute only brings the next beat forward from the interval: a wait the receiver asks
    for in that late reply still holds."""
    receiver = Receiver()
    receiver.reply_with(Reply(), Reply(429, {"Retry-After": "300"}))
    reporter = _goodbye_while_the_second_beat_is_in_flight(receiver)
    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    await settle()
    assert receiver.kinds == ["beat", "beat", "goodbye"]

    clock.advance(GOODBYE_QUIET_S + 1)
    await reporter.poll()
    assert receiver.kinds == ["beat", "beat", "goodbye"]
    clock.advance(300 - GOODBYE_QUIET_S)
    await reporter.poll()

    assert receiver.kinds == ["beat", "beat", "goodbye", "beat"]

@pytest.mark.parametrize("held_off", [Reply(401), Reply(503, {"Retry-After": "300"})], ids=["rejected", "asked to wait"])
async def test_no_goodbye_while_the_receiver_holds_the_sender_off(clock, held_off):
    receiver = Receiver()
    receiver.reply_with(held_off)
    reporter = make_reporter(receiver)
    await reporter.poll()

    reporter.goodbye()
    await settle()

    assert receiver.kinds == ["beat"]

async def test_no_goodbye_from_a_gateway_that_never_became_the_sender(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)

    reporter.goodbye()
    await settle()

    assert receiver.requests == []
