"""One sender per machine and report URL: the sender lock, a takeover from a dead or stopped
sender, and a lock directory that cannot be used."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import stat
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from gateway import status as gateway_status
from gateway.status_scoped_locks import release_scoped_lock
from gateway.status_report.reporter import StatusReporter
from gateway.status_report.target import STATUS_REPORT_URL_ENV, report_target_from_env
from hermes_constants import get_process_hermes_home
from tests.gateway.status_report.support import (
    AFTER_INTERVAL_S,
    FAULT_PATH,
    TOLD_S,
    Receiver,
    Watcher,
    a_status_source,
    around,
    make_reporter,
    new_credential,
    settle,
    stamped_url,
    told,
    warnings_said,
)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio

ROOT = Path(__file__).resolve().parents[3]


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

def _start_other_sender(tmp_path):
    """Another gateway of this machine, in its own process, reporting to the URL in the environment.
    Closing its stdin ends it."""
    other_root = tmp_path / "other-root"
    other_root.mkdir(exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", _FIRST_SENDER, str(ROOT)], env={**os.environ, "HERMES_HOME": str(other_root)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "reported", process.stderr.read()
    return process

@pytest.fixture
def first_sender(tmp_path, monkeypatch):
    """Another gateway of this machine already reporting to the same URL."""
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    process = _start_other_sender(tmp_path)
    try:
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

async def test_a_sender_whose_lock_was_taken_over_stands_by(clock, tmp_path, monkeypatch):
    """A sender stopped long enough for another gateway to take its lock (a stopped owner counts as
    gone) must not report beside that gateway once it resumes: ownership is checked before each
    report, and the lock is taken back once the other gateway is gone."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    # The other sender is a bare interpreter here; a real one is a gateway process.
    monkeypatch.setattr(gateway_status, "_looks_like_gateway_process", lambda pid: True)
    receiver = Receiver()
    reporter = StatusReporter(report_target_from_env(), launch_home=get_process_hermes_home(),
                              read_source=a_status_source, opener=receiver)
    await reporter.poll()
    assert receiver.kinds == ["beat"]

    taken_over = [path for path in lock_dir.rglob("*") if path.is_file()]
    assert taken_over, "the sender holds no lock file"
    for path in taken_over:
        path.unlink()
    other = await asyncio.to_thread(_start_other_sender, tmp_path)
    try:
        clock.advance(AFTER_INTERVAL_S)
        await reporter.poll()
        assert receiver.kinds == ["beat"], "a sender that lost its lock kept reporting beside its replacement"
        reporter.goodbye()
        await settle()
        assert receiver.kinds == ["beat"], "a sender that lost its lock said goodbye for its replacement"
    finally:
        await asyncio.to_thread(_end, other)

    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()
    assert receiver.kinds == ["beat", "beat"], "the sender never took its lock back"

async def test_a_sender_whose_lock_was_taken_over_says_no_goodbye_before_its_next_beat(clock, tmp_path, monkeypatch):
    """The suspend path can ask a stopped sender, resumed after another gateway took its lock, to say
    goodbye before its next beat has checked the lock: the goodbye checks it first, and stays unsaid."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    monkeypatch.setattr(gateway_status, "_looks_like_gateway_process", lambda pid: True)
    receiver = Receiver()
    reporter = StatusReporter(report_target_from_env(), launch_home=get_process_hermes_home(),
                              read_source=a_status_source, opener=receiver)
    await reporter.poll()
    assert receiver.kinds == ["beat"]
    for path in [path for path in lock_dir.rglob("*") if path.is_file()]:
        path.unlink()
    other = await asyncio.to_thread(_start_other_sender, tmp_path)
    try:
        reporter.goodbye()
        await settle()

        assert receiver.kinds == ["beat"], "a sender that lost its lock said goodbye for its replacement"
    finally:
        await asyncio.to_thread(_end, other)


@pytest.fixture
def unusable_lock_dir(tmp_path, monkeypatch):
    """A lock directory this user may not create a lock in, as an elevated install leaves it."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    lock_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    yield lock_dir
    lock_dir.chmod(stat.S_IRWXU)

@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")
async def test_an_unusable_lock_directory_stands_the_sender_by_until_it_is_repaired(clock, unusable_lock_dir, caplog):
    receiver = Receiver()
    reporter = make_reporter(receiver)

    await reporter.poll()
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()

    assert receiver.requests == []
    said = [record for record in caplog.records
            if record.name.startswith("gateway.status_report") and record.levelno >= logging.WARNING]
    assert len(said) == 1, [record.getMessage() for record in said]
    assert "writable" in said[0].getMessage() and "free space" in said[0].getMessage(), "the warning names no remedy"

    unusable_lock_dir.chmod(stat.S_IRWXU)
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()

    assert receiver.kinds == ["beat"], "the sender never picked the repaired directory up"

_LOCK_FAULTS = [
    pytest.param(OSError(errno.EROFS, "Read-only file system", FAULT_PATH), id="read-only disk"),
    pytest.param(OSError(errno.ENOSPC, "No space left on device", FAULT_PATH), id="full disk"),
    pytest.param(NotADirectoryError(errno.ENOTDIR, "Not a directory", FAULT_PATH), id="a file where the lock directory goes"),
]

@pytest.mark.parametrize("fault", _LOCK_FAULTS)
async def test_a_sender_whose_lock_cannot_be_confirmed_keeps_reporting(clock, monkeypatch, fault):
    """No other gateway can take the lock from a directory this one cannot read either, so a sender
    that cannot confirm its lock keeps reporting rather than going quiet."""
    faults = []

    def take_the_lock(scope, identity):
        if faults:
            raise faults[0]
        return gateway_status.acquire_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    receiver = Receiver()
    reporter = make_reporter(receiver)
    await reporter.poll()
    faults.append(fault)
    clock.advance(AFTER_INTERVAL_S)
    await reporter.poll()

    assert receiver.kinds == ["beat", "beat"]

@pytest.mark.parametrize("fault", _LOCK_FAULTS)
async def test_a_lock_that_cannot_be_taken_stands_the_sender_by_and_never_ends_the_watcher(
        clock, caplog, monkeypatch, fault):
    """A watcher that dies on it is restarted a few times and then given up on: a healthy gateway
    that never reports again."""
    caplog.set_level(logging.DEBUG)
    faults, tries = [fault], []

    def take_the_lock(scope, identity):
        tries.append(scope)
        if faults:
            raise faults[0]
        return gateway_status.acquire_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    receiver = Receiver()

    async with Watcher(make_reporter(receiver)) as watcher:
        for failed in (1, 2):
            await async_wait_until(lambda failed=failed: len(tries) == failed, timeout=5, what=f"try {failed} for the lock")
            await watcher.polls_on()
            assert len(tries) == failed, "the lock was tried again within the interval"
            clock.advance(AFTER_INTERVAL_S)
        assert receiver.requests == []

        faults.clear()  # the disk is usable again
        clock.advance(AFTER_INTERVAL_S)
        await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="a beat once the lock can be taken")
        await watcher.ends_when_cancelled_mid_beat(clock, receiver)

    assert len(warnings_said(caplog)) == 1, warnings_said(caplog)
    # The reason is fixed: nothing of the error reaches the log, not even its type.
    for leaked in (FAULT_PATH, fault.strerror, type(fault).__name__):
        assert leaked not in caplog.text

async def test_a_sender_that_lost_its_lock_stands_by_at_the_told_interval(clock, monkeypatch):
    tries, held_elsewhere = [], []

    def take_the_lock(scope, identity):
        tries.append(scope)
        if held_elsewhere:
            return False, None
        return gateway_status.acquire_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    receiver = Receiver()
    receiver.reply_with(told(TOLD_S))
    reporter = make_reporter(receiver)
    await reporter.poll()
    held_elsewhere.append(True)
    before, after = around(TOLD_S)
    clock.advance(after)
    await reporter.poll()
    assert (len(tries), len(receiver.requests)) == (2, 1)

    clock.advance(before)
    await reporter.poll()
    assert len(tries) == 2, "the lock was tried again before the told interval"
    clock.advance(after - before)
    await reporter.poll()
    assert len(tries) == 3, "the lock was not tried at the told interval"

async def test_a_watcher_cancelled_while_it_takes_the_lock_leaves_no_lock_held(clock, monkeypatch):
    """A gateway shutdown can cancel the watcher while its thread is still taking the lock. The
    thread takes it after the cancel, and a lock held by a process that lives on keeps every
    other gateway of the machine standing by until that process exits."""
    entered, release, taken = threading.Event(), threading.Event(), []

    def take_the_lock_slowly(scope, identity):
        entered.set()
        release.wait(timeout=5)
        result = gateway_status.acquire_scoped_lock(scope, identity)
        taken.append(result[0])
        return result

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock_slowly)
    receiver = Receiver()
    watcher = asyncio.create_task(make_reporter(receiver).run(lambda: True, tick=0.01))
    await async_wait_until(entered.is_set, timeout=5, what="the lock being taken")

    watcher.cancel()
    await asyncio.wait({watcher}, timeout=5)
    release.set()
    await async_wait_until(lambda: taken, timeout=5, what="the lock taken after the cancel")

    assert watcher.cancelled() and taken == [True]
    await async_wait_until(lambda: not list(gateway_status._get_lock_dir().glob("*.lock")), timeout=5,
                           what="a cancelled watcher to give back the lock it took")
    assert receiver.requests == []


async def test_a_sender_cancelled_while_it_confirms_its_lock_leaves_no_lock_held(clock, monkeypatch):
    """The same for a sender confirming the lock it holds: run()'s cleanup releases it, and the
    thread still confirming then takes it again, after the cleanup."""
    confirming, release, taken = threading.Event(), threading.Event(), []

    def take_the_lock(scope, identity):
        if taken:  # the second beat's confirmation: this gateway is the sender by now
            confirming.set()
            release.wait(timeout=5)
        result = gateway_status.acquire_scoped_lock(scope, identity)
        taken.append(result[0])
        return result

    released_on = []

    def give_back(scope, identity):
        released_on.append(threading.current_thread().name)
        release_scoped_lock(scope, identity)

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    monkeypatch.setattr("gateway.status_report.reporter.release_scoped_lock", give_back)
    receiver = Receiver()
    watcher = asyncio.create_task(make_reporter(receiver).run(lambda: True, tick=0.01))
    await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="the first beat")
    clock.advance(AFTER_INTERVAL_S)
    await async_wait_until(confirming.is_set, timeout=5, what="the second beat confirming the lock")

    watcher.cancel()
    await asyncio.wait({watcher}, timeout=5)
    release.set()
    await async_wait_until(lambda: len(taken) == 2, timeout=5, what="the confirmation after the cancel")

    assert watcher.cancelled() and taken == [True, True]
    await async_wait_until(lambda: not list(gateway_status._get_lock_dir().glob("*.lock")), timeout=5,
                           what="a cancelled sender to give back the lock its confirmation took again")
    # The cleanup and the give-back each release it. A release reads the owner and then unlinks, so
    # two at once on different threads could unlink a lock another gateway took in between: both
    # run on the event loop, one after the other.
    assert released_on == [threading.current_thread().name] * 2


async def test_a_goodbye_still_taking_the_lock_when_the_watcher_stops_neither_takes_it_nor_is_sent(clock, monkeypatch):
    """The watcher's cleanup gives the lock up. A goodbye still confirming it then must not take it
    back and report for a gateway that has stopped, beside whichever gateway took over."""
    confirming, release, taken = threading.Event(), threading.Event(), []

    def take_the_lock(scope, identity):
        if taken:  # the goodbye's confirmation: the first one was the beat's
            confirming.set()
            release.wait(timeout=5)
        result = gateway_status.acquire_scoped_lock(scope, identity)
        taken.append(result[0])
        return result

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    receiver = Receiver()
    reporter = make_reporter(receiver)
    watcher = asyncio.create_task(reporter.run(lambda: True, tick=0.01))
    await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="the first beat")
    reporter.goodbye()
    await async_wait_until(confirming.is_set, timeout=5, what="the goodbye confirming the lock")

    watcher.cancel()
    await asyncio.wait({watcher}, timeout=5)
    release.set()
    await async_wait_until(lambda: len(taken) == 2, timeout=5, what="the goodbye's confirmation")
    await settle()

    assert receiver.kinds == ["beat"], "a goodbye went out after the watcher stopped"
    await async_wait_until(lambda: not list(gateway_status._get_lock_dir().glob("*.lock")), timeout=5,
                           what="the lock to be free after the watcher stopped")


async def test_a_goodbye_cancelled_while_it_confirms_the_lock_leaves_the_sender_holding_it(clock, monkeypatch):
    """The goodbye confirms the lock too. Cancelled there (the loop closing under it), it must not
    give back a lock the watcher still reports under."""
    confirming, release, confirmed = threading.Event(), threading.Event(), []

    def take_the_lock(scope, identity):
        if confirmed:  # the goodbye's confirmation: the first one was the beat's
            confirming.set()
            release.wait(timeout=5)
        result = gateway_status.acquire_scoped_lock(scope, identity)
        confirmed.append(result[0])
        return result

    monkeypatch.setattr("gateway.status_report.reporter.acquire_scoped_lock", take_the_lock)
    receiver = Receiver()
    reporter = make_reporter(receiver)
    await reporter.poll()
    before = {task for task in asyncio.all_tasks()}
    reporter.goodbye()
    (goodbye,) = asyncio.all_tasks() - before
    await async_wait_until(confirming.is_set, timeout=5, what="the goodbye confirming the lock")

    goodbye.cancel()
    await asyncio.wait({goodbye}, timeout=5)
    release.set()
    # Every piece of the reporter's work is done once its threads end, and what it queued on the
    # loop on completion runs before this resumes.
    await asyncio.to_thread(reporter._pool.shutdown, True)

    assert len(confirmed) == 2
    assert list(gateway_status._get_lock_dir().glob("*.lock")), "a cancelled goodbye gave the sender's lock back"
    assert receiver.kinds == ["beat"]


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
