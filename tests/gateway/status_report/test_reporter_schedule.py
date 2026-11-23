"""The sender's schedule: wall-clock timing, and what each reply does to the next report."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import stat
import urllib.error

import pytest

from agent.runtime_scope import bind_profile_scope
from gateway import status as gateway_status
from gateway.status_report.payload import read_status_source
from gateway.status_report.reporter import StatusReporter, status_reporter_from_launch_env
from gateway.status_report.target import STATUS_REPORT_URL_ENV, report_target_from_env
from hermes_constants import get_hermes_home, get_process_hermes_home
from tests.gateway.status_report.support import (
    AFTER_INTERVAL_S,
    FAULT_PATH,
    INTERVAL_HEADER,
    QUICK_RETRY_S,
    REPORT_URL,
    RESPONSES,
    TOLD_S,
    Receiver,
    Reply,
    Watcher,
    a_status_source,
    around,
    assert_matches_contract,
    cases,
    expect_next_report_at_the_interval,
    make_reporter,
    new_credential,
    reply_for,
    settle,
    told,
    warnings_said,
)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio


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
    await expect_next_report_at_the_interval(clock, reporter, receiver)
    await expect_next_report_at_the_interval(clock, reporter, receiver)

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2, 3]
    assert {seen.body["boot_id"] for seen in receiver.requests} == {reporter.boot_id}
    assert make_reporter(Receiver()).boot_id != reporter.boot_id

@pytest.mark.parametrize("case", cases("next_interval"))
async def test_these_replies_leave_the_normal_interval(clock, case):
    receiver = Receiver()
    receiver.reply_with(reply_for(case))
    reporter = make_reporter(receiver)

    await reporter.poll()

    await expect_next_report_at_the_interval(clock, reporter, receiver)
    assert receiver.requests[1].body["seq"] == 2

@pytest.mark.parametrize("case", cases("wait"))
async def test_a_retry_after_is_honoured_up_to_the_cap(clock, case):
    receiver = Receiver()
    receiver.reply_with(reply_for(case))
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(case["wait_s"] - 1)
    await reporter.poll()
    assert len(receiver.requests) == 1, "a report went out before the wait was over"
    clock.advance(1)
    await reporter.poll()

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2]

async def test_a_retry_after_too_long_to_read_as_an_integer_waits_the_cap(clock):
    receiver = Receiver()
    receiver.reply_with(Reply(429, {"Retry-After": "9" * 5000}))
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(RESPONSES["retry_after_max_s"] - 1)
    await reporter.poll()
    assert len(receiver.requests) == 1, "a report went out before the cap was over"
    clock.advance(1)
    await reporter.poll()

    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2]

@pytest.mark.parametrize("case", cases("backoff"))
async def test_a_rejected_credential_backs_off_in_steps_and_a_stored_report_resets_them(clock, case):
    steps = RESPONSES["rejected_backoff_s"]
    receiver = Receiver()
    receiver.reply_with(*[reply_for(case) for _ in steps])
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
    receiver.reply_with(reply_for(case))
    await expect_next_report_at_the_interval(clock, reporter, receiver)
    sent = len(receiver.requests)
    clock.advance(steps[0] - 1)
    await reporter.poll()
    assert len(receiver.requests) == sent
    clock.advance(1)
    await reporter.poll()
    assert len(receiver.requests) == sent + 1

_FAILURES = [
    *[pytest.param(lambda case=case.values[0]: reply_for(case), id=case.id) for case in cases("quick_retry")],
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
    await expect_next_report_at_the_interval(clock, reporter, receiver)
    assert receiver.requests[2].body["seq"] == 2

async def test_a_quick_retry_that_is_stored_returns_to_the_interval(clock):
    receiver = Receiver()
    receiver.reply_with(Reply(500))
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(QUICK_RETRY_S)
    await reporter.poll()
    assert len(receiver.requests) == 2

    await expect_next_report_at_the_interval(clock, reporter, receiver)

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

    await expect_next_report_at_the_interval(clock, reporter, receiver)

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

@pytest.fixture
def unreadable_profile_env():
    """The launch profile's ``.env`` as an elevated write leaves it: there, and not readable by this user."""
    env_file = get_process_hermes_home() / ".env"
    env_file.write_text("UNRELATED=1\n", encoding="utf-8")
    env_file.chmod(0)
    yield env_file
    env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    env_file.unlink()

@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")
async def test_an_unreadable_profile_env_file_neither_warns_nor_costs_the_status(clock, unreadable_profile_env, caplog):
    """A report reads the gateway's own record and the sign-in state, nothing a ``.env`` holds."""
    assert gateway_status.write_runtime_status(gateway_state="running", wait_timeout=10.0)
    receiver = Receiver()
    reporter = make_reporter(receiver, source=read_status_source)

    await reporter.poll()
    await expect_next_report_at_the_interval(clock, reporter, receiver)

    assert [seen.body["status"] for seen in receiver.requests] == [{
        "gateway_running": True, "gateway_state": "running", "gateway_exit_reason": None,
        "gateway_platforms": {}, "nous_session_valid": "unknown"}] * 2
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING] == []

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

async def test_a_report_that_cannot_be_built_costs_an_interval_and_never_ends_the_watcher(clock, caplog):
    caplog.set_level(logging.DEBUG)
    faults = [OSError(errno.EIO, "Input/output error", FAULT_PATH), OSError(errno.EIO, "Input/output error", FAULT_PATH),
              ValueError(f"not a status record: {FAULT_PATH}")]
    reads = []

    def source():
        reads.append(len(faults))
        if faults:
            raise faults.pop(0)
        return a_status_source()

    receiver = Receiver()

    async with Watcher(make_reporter(receiver, source=source)) as watcher:
        for failed in (1, 2, 3):
            await async_wait_until(lambda failed=failed: len(reads) == failed, timeout=5, what=f"status read {failed}")
            await watcher.polls_on()
            assert len(reads) == failed, "the status was read again within the interval"
            clock.advance(AFTER_INTERVAL_S)

        await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="a beat once the status can be read")
        await watcher.ends_when_cancelled_mid_beat(clock, receiver)

    # Each kind of error is said once, by its type alone.
    said = warnings_said(caplog)
    assert len(said) == 2 and "OSError" in said[0] and "ValueError" in said[1], said
    for leaked in (FAULT_PATH, "Input/output error", "not a status record"):
        assert leaked not in caplog.text

_TOLD = [
    *cases("told_interval"),
    pytest.param({"status": 204, "headers": {INTERVAL_HEADER: "9" * 5000}, "wait_s": RESPONSES["interval_max_s"]},
                 id="report interval too long to read as an integer"),
]

_STORED_WITHOUT_AN_INTERVAL = [case for case in cases("next_interval") if case.values[0]["outcome"] == "stored"]

_NOT_STORED = [case for case in cases("next_interval") if case.values[0]["outcome"] != "stored"]

@pytest.mark.parametrize("case", _TOLD)
async def test_a_stored_reply_sets_the_interval_of_the_next_and_later_reports(clock, case):
    receiver = Receiver()
    receiver.reply_with(reply_for(case), reply_for(case))
    reporter = make_reporter(receiver)

    await reporter.poll()

    await expect_next_report_at_the_interval(clock, reporter, receiver, case["wait_s"])
    await expect_next_report_at_the_interval(clock, reporter, receiver, case["wait_s"])
    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2, 3]

async def test_reports_at_a_told_interval_keep_the_jitter(clock):
    """Every gateway of a deployment is told the same interval; without the jitter they would report in step."""
    receiver = Receiver()
    receiver.reply_with(*[told(TOLD_S)] * 6)
    reporter = make_reporter(receiver)
    await reporter.poll()
    before, after = around(TOLD_S)

    waits = []
    for sent in range(1, 7):
        clock.advance(before)
        waited = before
        while waited < after and len(receiver.requests) == sent:
            clock.advance(1)
            waited += 1
            await reporter.poll()
        assert len(receiver.requests) == sent + 1, f"no report within the jitter band of {TOLD_S}s"
        waits.append(waited)

    assert len(set(waits)) > 1, f"every report waited exactly {waits[0]}s"

@pytest.mark.parametrize("case", _STORED_WITHOUT_AN_INTERVAL)
async def test_a_stored_reply_without_a_valid_interval_goes_back_to_the_normal_one(clock, case):
    receiver = Receiver()
    receiver.reply_with(told(TOLD_S), reply_for(case))
    reporter = make_reporter(receiver)
    await reporter.poll()
    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)

    await expect_next_report_at_the_interval(clock, reporter, receiver)

@pytest.mark.parametrize("case", _NOT_STORED)
async def test_a_told_interval_outlasts_a_reply_that_is_not_stored(clock, case):
    receiver = Receiver()
    receiver.reply_with(told(TOLD_S), reply_for(case))
    reporter = make_reporter(receiver)
    await reporter.poll()
    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)

    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)

async def test_a_report_interval_on_a_reply_that_is_not_stored_changes_nothing(clock):
    (case,) = [case for case in RESPONSES["cases"] if case["then"] == "wait" and INTERVAL_HEADER in case["headers"]]
    receiver = Receiver()
    receiver.reply_with(reply_for(case), Reply(404))
    reporter = make_reporter(receiver)
    await reporter.poll()
    clock.advance(case["wait_s"])
    await reporter.poll()
    assert len(receiver.requests) == 2

    # The second reply set no wait of its own, so the next report shows the interval in force.
    await expect_next_report_at_the_interval(clock, reporter, receiver)

async def test_a_told_interval_outlasts_a_failed_report_and_its_quick_retry(clock):
    receiver = Receiver()
    receiver.reply_with(told(TOLD_S), Reply(500), Reply(error=TimeoutError("timed out")))
    reporter = make_reporter(receiver)
    await reporter.poll()
    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)
    clock.advance(QUICK_RETRY_S)
    await reporter.poll()
    assert [seen.body["seq"] for seen in receiver.requests] == [1, 2, 2]

    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)

async def test_a_told_interval_outlasts_a_poll_that_raised(clock):
    faults, reads = [], []

    def source():
        reads.append("read")
        if faults:
            raise faults.pop()
        return a_status_source()

    receiver = Receiver()
    receiver.reply_with(told(TOLD_S))
    before, after = around(TOLD_S)

    async with Watcher(make_reporter(receiver, source=source)) as watcher:
        await async_wait_until(lambda: receiver.answered == 1, timeout=5, what="the first report")
        await watcher.polls_on()
        faults.append(OSError(errno.EIO, "Input/output error", FAULT_PATH))
        clock.advance(after)
        await async_wait_until(lambda: len(reads) == 2, timeout=5, what="the status read that fails")
        await watcher.polls_on()

        clock.advance(before)
        await watcher.polls_on()
        assert len(reads) == 2, "the status was read again before the told interval"
        clock.advance(after - before)
        await async_wait_until(lambda: receiver.kinds == ["beat", "beat"], timeout=5, what="a beat at the told interval")

async def test_every_reporter_starts_at_the_normal_interval_and_reports_at_once(clock):
    """The interval a receiver set is the reporter's own: a gateway that starts again was told nothing yet."""
    earlier = Receiver()
    earlier.reply_with(told(TOLD_S))
    await make_reporter(earlier).poll()
    receiver = Receiver()
    receiver.reply_with(Reply(404))  # sets neither an interval nor a wait: the next report shows where it started
    reporter = make_reporter(receiver)

    await reporter.poll()

    assert len(receiver.requests) == 1, "a new reporter did not report at once"
    await expect_next_report_at_the_interval(clock, reporter, receiver)

def _kept_outside_the_reporter() -> tuple[dict, dict]:
    home = get_process_hermes_home()
    files = {str(path.relative_to(home)): path.read_bytes() for path in home.rglob("*") if path.is_file()}
    return files, dict(os.environ)

async def test_a_told_interval_is_never_written_to_a_file_or_the_environment(clock):
    receiver = Receiver()
    receiver.reply_with(told(TOLD_S), told(RESPONSES["interval_min_s"]))
    reporter = make_reporter(receiver)
    kept = [_kept_outside_the_reporter()]

    await reporter.poll()
    kept.append(_kept_outside_the_reporter())
    await expect_next_report_at_the_interval(clock, reporter, receiver, TOLD_S)
    kept.append(_kept_outside_the_reporter())
    await expect_next_report_at_the_interval(clock, reporter, receiver, RESPONSES["interval_min_s"])

    # Told two different intervals in turn: a copy kept anywhere would differ between the last two.
    assert kept[1] == kept[0] and kept[2] == kept[0]
