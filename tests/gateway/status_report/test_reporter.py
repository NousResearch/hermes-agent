"""The sender's schedule: wall-clock timing, what each reply does to the next report, one sender."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import textwrap
import urllib.error
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope
from gateway import status as gateway_status
from gateway.status_report.reporter import StatusReporter, status_reporter_from_launch_env
from gateway.status_report.target import STATUS_REPORT_URL_ENV, report_target_from_env
from hermes_constants import get_hermes_home, get_process_hermes_home
from tests.gateway.status_report.support import (
    AFTER_INTERVAL_S,
    BEFORE_INTERVAL_S,
    REPORT_URL,
    RESPONSES,
    Clock,
    Receiver,
    Reply,
    a_status_source,
    assert_matches_contract,
    make_reporter,
    new_credential,
    settle,
    stamped_url,
)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio

QUICK_RETRY_S = RESPONSES["quick_retry_s"]
GOODBYE_QUIET_S = RESPONSES["goodbye_quiet_s"]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def clock(monkeypatch):
    return Clock(monkeypatch)


def _cases(then: str) -> list:
    return [pytest.param(case, id=case["name"]) for case in RESPONSES["cases"] if case["then"] == then]


def _reply(case: dict) -> Reply:
    return Reply(case["status"], case["headers"], case.get("body"))


async def _expect_next_report_at_the_interval(clock, reporter, receiver) -> None:
    sent = len(receiver.requests)
    clock.advance(BEFORE_INTERVAL_S)
    await reporter.poll()
    assert len(receiver.requests) == sent, "a report went out before the interval"
    clock.advance(AFTER_INTERVAL_S - BEFORE_INTERVAL_S)
    await reporter.poll()
    assert len(receiver.requests) == sent + 1, "no report at the interval"


async def test_the_first_report_goes_out_at_once_and_matches_the_contract(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)

    await reporter.poll()

    (seen,) = receiver.requests
    assert_matches_contract(seen.body, "beat")
    assert (seen.body["seq"], seen.body["boot_id"]) == (1, reporter.boot_id)
    assert seen.body["status"] == {
        "gateway_running": True, "gateway_state": "running", "gateway_exit_reason": None,
        "gateway_platforms": {"telegram": {"state": "connected"}}, "nous_session_valid": "valid"}


async def test_seq_grows_by_one_per_report_and_boot_id_is_per_sender(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)

    await reporter.poll()
    await _expect_next_report_at_the_interval(clock, reporter, receiver)
    await _expect_next_report_at_the_interval(clock, reporter, receiver)

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2, 3]
    assert {seen.body["boot_id"] for seen in receiver.requests} == {reporter.boot_id}
    assert make_reporter(Receiver()).boot_id != reporter.boot_id


@pytest.mark.parametrize("case", _cases("next_interval"))
async def test_these_replies_leave_the_normal_interval(clock, case):
    receiver = Receiver()
    receiver.reply_with(_reply(case))
    reporter = make_reporter(receiver)

    await reporter.poll()

    await _expect_next_report_at_the_interval(clock, reporter, receiver)
    assert receiver.requests[1].body["seq"] == 2


@pytest.mark.parametrize("case", _cases("wait"))
async def test_a_retry_after_is_honoured_up_to_the_cap(clock, case):
    receiver = Receiver()
    receiver.reply_with(_reply(case))
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(case["wait_s"] - 1)
    await reporter.poll()
    assert len(receiver.requests) == 1, "a report went out before the wait was over"
    clock.advance(1)
    await reporter.poll()

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2]


@pytest.mark.parametrize("case", _cases("backoff"))
async def test_a_rejected_credential_backs_off_in_steps_and_a_stored_report_resets_them(clock, case):
    steps = RESPONSES["rejected_backoff_s"]
    receiver = Receiver()
    receiver.reply_with(*[_reply(case) for _ in steps])
    reporter = make_reporter(receiver)

    await reporter.poll()
    for sent, step in enumerate(steps, start=1):
        clock.advance(step - 1)
        await reporter.poll()
        assert len(receiver.requests) == sent, f"a report went out before the {step}s step was over"
        clock.advance(1)
        await reporter.poll()
        assert len(receiver.requests) == sent + 1, f"no report after the {step}s step"

    # The last report was stored (the script ran out): the next rejection starts at the first step again.
    receiver.reply_with(_reply(case))
    await _expect_next_report_at_the_interval(clock, reporter, receiver)
    sent = len(receiver.requests)
    clock.advance(steps[0] - 1)
    await reporter.poll()
    assert len(receiver.requests) == sent
    clock.advance(1)
    await reporter.poll()
    assert len(receiver.requests) == sent + 1


_FAILURES = [
    *[pytest.param(lambda case=case.values[0]: _reply(case), id=case.id) for case in _cases("quick_retry")],
    pytest.param(lambda: Reply(error=urllib.error.URLError(ConnectionRefusedError(61, "refused"))), id="network error"),
    pytest.param(lambda: Reply(error=TimeoutError("timed out")), id="timeout"),
]


@pytest.mark.parametrize("failure", _FAILURES)
async def test_a_failed_report_gets_exactly_one_quick_retry(clock, failure):
    receiver = Receiver()
    receiver.reply_with(failure(), failure())
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(QUICK_RETRY_S - 1)
    await reporter.poll()
    assert len(receiver.requests) == 1, "the retry went out too early"
    clock.advance(1)
    await reporter.poll()
    assert len(receiver.requests) == 2, "no quick retry"
    assert receiver.requests[1].raw == receiver.requests[0].raw, "the retry is the same report"

    # The retry failed too: no second one, the next report waits for the interval.
    await _expect_next_report_at_the_interval(clock, reporter, receiver)
    assert receiver.requests[2].body["seq"] == 2


async def test_a_quick_retry_that_is_stored_returns_to_the_interval(clock):
    receiver = Receiver()
    receiver.reply_with(Reply(500))
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(QUICK_RETRY_S)
    await reporter.poll()
    assert len(receiver.requests) == 2

    await _expect_next_report_at_the_interval(clock, reporter, receiver)


async def test_after_a_freeze_the_next_report_goes_out_at_once(clock):
    """A machine suspend stops the monotonic clock while the wall clock keeps host time, so a loop
    that slept the interval on the event loop's clock would report late by the time it was frozen."""
    receiver = Receiver()
    reporter = make_reporter(receiver)
    running = [True]
    watcher = asyncio.create_task(reporter.run(lambda: running[0], tick=0.01))
    try:
        await async_wait_until(lambda: len(receiver.requests) == 1, timeout=5, what="the first report")
        clock.advance(6 * 3600)  # thawed hours later: only the wall clock moved
        await async_wait_until(lambda: len(receiver.requests) == 2, timeout=5, what="a report right after the thaw")
    finally:
        running[0] = False
        await asyncio.wait_for(watcher, timeout=5)

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2]


async def test_a_wall_clock_step_back_does_not_silence_reports(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(-86400)  # the host clock was corrected backwards by a day
    await reporter.poll()

    await _expect_next_report_at_the_interval(clock, reporter, receiver)


async def test_the_status_is_read_under_the_launch_profile_whatever_scope_is_ambient(clock, tmp_path):
    homes = []

    def source():
        homes.append(get_hermes_home())
        return a_status_source()

    reporter = make_reporter(Receiver(), source=source)
    other = tmp_path / "another-profile"
    other.mkdir()

    with bind_profile_scope(other, hydrate_secrets=False):
        await reporter.poll()

    assert homes == [get_process_hermes_home()]


async def test_neither_the_url_nor_the_credential_reaches_the_logs(clock, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    credential = new_credential()
    stamp = f"{REPORT_URL}?zone=eu&t={credential}"
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamp)
    assert status_reporter_from_launch_env() is not None
    receiver = Receiver()
    receiver.reply_with(
        Reply(), Reply(401), Reply(500), Reply(error=urllib.error.URLError(f"cannot reach {stamp}")),
        Reply(429, {"Retry-After": "30"}), Reply(404), Reply(error=ValueError(f"Invalid header value {credential}")))
    reporter = StatusReporter(report_target_from_env(), launch_home=get_process_hermes_home(),
                              read_source=a_status_source, opener=receiver)

    for _ in range(8):
        await reporter.poll()
        clock.advance(3600)
    reporter.goodbye()
    await settle()

    assert receiver.kinds == ["beat"] * 8 + ["goodbye"]
    ours = [record for record in caplog.records if record.name.startswith("gateway.status_report")]
    assert any(record.levelno == logging.WARNING and "401" in record.getMessage() for record in ours)
    for leaked in (credential, REPORT_URL, "status.example", "/v1/gateways/report", "zone=eu"):
        assert leaked not in caplog.text


_FIRST_SENDER = textwrap.dedent("""
    import asyncio, sys
    sys.path.insert(0, sys.argv[1])
    from gateway.status_report.reporter import StatusReporter
    from gateway.status_report.target import report_target_from_env
    from hermes_constants import get_process_hermes_home

    class Stored:
        status, headers = 204, {}
        def __enter__(self): return self
        def __exit__(self, *exc_info): return False

    reporter = StatusReporter(report_target_from_env(), launch_home=get_process_hermes_home(),
                              read_source=dict, opener=lambda request, timeout=None: Stored())
    asyncio.run(reporter.poll())
    print("reported", flush=True)
    sys.stdin.read()  # keep the lock until the test closes our stdin
""")


@pytest.fixture
def first_sender(tmp_path, monkeypatch):
    """Another gateway of this machine, in its own process, already reporting to the same URL.
    Closing its stdin ends it."""
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-c", _FIRST_SENDER, str(ROOT)], env={**os.environ, "HERMES_HOME": str(other_root)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "reported", process.stderr.read()
        yield process
    finally:
        _end(process)


def _end(process) -> None:
    if not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


async def test_a_second_sender_cannot_take_the_lock(clock, first_sender, monkeypatch):
    """Two gateways of one machine inherit the same stamp; only one of them may report."""
    receiver = Receiver()
    reporter = StatusReporter(report_target_from_env(), launch_home=get_process_hermes_home(),
                              read_source=a_status_source, opener=receiver)
    # The first sender is a bare interpreter here; a real one is a gateway process.
    monkeypatch.setattr(gateway_status, "_looks_like_gateway_process", lambda pid: True)

    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    assert receiver.requests == [], "a second gateway reported beside the first"

    await asyncio.to_thread(_end, first_sender)
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    assert receiver.kinds == ["beat"], "the remaining gateway never took over from a dead sender"


async def test_the_lock_is_released_when_the_watcher_stops(clock):
    receiver = Receiver()
    reporter = make_reporter(receiver)
    running = [True]
    watcher = asyncio.create_task(reporter.run(lambda: running[0], tick=0.01))
    try:
        await async_wait_until(lambda: len(receiver.requests) == 1, timeout=5, what="the first report")
        assert list(gateway_status._get_lock_dir().glob("*.lock")), "reporting without the sender lock"
    finally:
        running[0] = False
        await asyncio.wait_for(watcher, timeout=5)

    assert list(gateway_status._get_lock_dir().glob("*.lock")) == []


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
