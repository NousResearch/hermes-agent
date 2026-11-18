"""The sender: when a report goes out, what each reply does to the next one, and which gateway sends."""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent.runtime_scope import bind_state, scope_for_profile
from gateway.status import acquire_scoped_lock
from gateway.status_report.payload import ReportKind, build_report, read_status_source
from gateway.status_report.target import ReportTarget, report_target_from_env
from gateway.status_report.transport import Outcome, post_report
from gateway.status_scoped_locks import release_scoped_lock
from hermes_constants import get_process_hermes_home, profile_name_for_home

logger = logging.getLogger(__name__)

INTERVAL_S = 120.0
# The receiver may set another interval in a stored reply, within these bounds.
INTERVAL_MIN_S = 60.0
INTERVAL_MAX_S = 600.0
_JITTER = 0.1
_QUICK_RETRY_S = 5.0
_RETRY_AFTER_MAX_S = 600.0
_REJECTED_BACKOFF_S = tuple(minutes * 60.0 for minutes in (2, 4, 8, 16, 32, 60))
_BEAT_TIMEOUT_S = 10.0
# A machine freezes 2 to 4 s after its suspend is accepted; a goodbye still unanswered by then is lost anyway.
_GOODBYE_TIMEOUT_S = 3.0
_GOODBYE_QUIET_S = 60.0
_TICK_S = 1.0
_LOCK_SCOPE = "status-report"


def _jittered(interval_s: float) -> float:
    return random.uniform(interval_s * (1 - _JITTER), interval_s * (1 + _JITTER))


class StatusReporter:
    """Reports this gateway's status to the deployment's URL: a beat about every ``INTERVAL_S``, or
    the interval the receiver's last stored reply set, and a goodbye before the gateway suspends
    its own machine.

    Timed on the WALL clock. A machine suspend stops the monotonic clock the event loop sleeps on
    while the wall clock keeps host time, so a slept interval would resume with its remainder after
    the wake. Compared on ``time.time()`` the deadline is already past by then, and the first report
    after a wake goes out within a tick.

    State is in memory only and changes only on the event loop; file and network work runs on
    worker threads."""

    def __init__(self, target: ReportTarget, *, launch_home: Path,
                 read_source: Callable[[], Mapping[str, Any]] = read_status_source, opener: Any = None) -> None:
        self.boot_id = secrets.token_hex(16)
        self._target = target
        self._launch_home = launch_home
        self._read_source = read_source
        self._opener = opener
        self._seqs = itertools.count(1)
        self._interval_s = INTERVAL_S
        self._due_at = 0.0
        self._last_poll_at = 0.0
        self._no_beat_before = 0.0
        self._retry_body: Optional[bytes] = None
        self._rejections = 0
        self._receiver_holds_off = False
        self._is_sender = False
        self._standing_by_for: Optional[str] = None
        self._poll_failures: set[str] = set()
        self._last_kind: Optional[str] = None
        self._goodbyes: set[asyncio.Task] = set()

    async def run(self, is_running: Callable[[], bool], *, tick: float = _TICK_S) -> None:
        """Poll until the gateway stops. The sleep between polls only bounds how soon a due report
        is noticed; whether one is due is decided on the wall clock."""
        try:
            while is_running():
                try:
                    await self.poll()
                except Exception as exc:  # health: allow BLE001 -- must outlive any failed poll; the type is logged
                    self._poll_failed(exc)
                await asyncio.sleep(tick)
        finally:
            if self._is_sender:
                self._is_sender = False
                release_scoped_lock(_LOCK_SCOPE, self._target.url)

    def _poll_failed(self, exc: Exception) -> None:
        """A poll that raised costs one interval, never the watcher: a supervised task that keeps
        dying is given up on, and the gateway would then never report again. Said once per type of
        error, by the type alone: an error's text can quote a path or the report URL."""
        self._due_at = time.time() + _jittered(self._interval_s)
        kind = type(exc).__name__
        if kind not in self._poll_failures:
            self._poll_failures.add(kind)
            logger.warning("status report: no report this interval (%s)", kind)

    async def poll(self) -> None:
        """Send a beat if one is due."""
        now = time.time()
        if now < self._last_poll_at:
            self._due_at -= self._last_poll_at - now
            self._no_beat_before -= self._last_poll_at - now
        self._last_poll_at = now
        if now < self._due_at:
            return
        if not await self._become_sender():
            self._due_at = now + _jittered(self._interval_s)
            return
        retry, self._retry_body = self._retry_body, None
        body = retry or await self._next_report("beat")
        outcome = await asyncio.to_thread(
            post_report, self._target, body, timeout=_BEAT_TIMEOUT_S, opener=self._opener)
        self._settle(outcome, body, retried=retry is not None)

    def goodbye(self) -> None:
        """Start the last report before this machine suspends and return at once: the caller never
        waits for the receiver. One attempt with a short timeout, on its own task. Skipped by a
        gateway that is not the sender and while the receiver holds the sender off."""
        if not self._is_sender or self._receiver_holds_off:
            return
        self._retry_body = None
        self._no_beat_before = time.time() + _GOODBYE_QUIET_S
        self._due_at = max(self._due_at, self._no_beat_before)
        task = asyncio.get_running_loop().create_task(self._say_goodbye())
        self._goodbyes.add(task)  # the loop keeps only a weak reference to a running task
        task.add_done_callback(self._goodbye_done)

    def _goodbye_done(self, task: asyncio.Task) -> None:
        self._goodbyes.discard(task)
        # Read the task's exception here: left unread it is reported as never retrieved once the
        # task is collected. Only its type is logged, as for a report that got no reply.
        if not task.cancelled() and task.exception() is not None:
            logger.debug("status report: goodbye not sent (%s)", type(task.exception()).__name__)

    async def _say_goodbye(self) -> None:
        body = await self._next_report("goodbye")
        outcome = await asyncio.to_thread(
            post_report, self._target, body, timeout=_GOODBYE_TIMEOUT_S, opener=self._opener)
        logger.info("status report: goodbye %s (%s)", outcome.kind, outcome.status or "no reply")

    async def _become_sender(self) -> bool:
        """Take, or confirm, the machine-local sender lock before each report. Every gateway of one
        machine inherits the same report URL, so the lock decides which of them reports; one that
        died is replaced on a later poll, and so is a lock directory that was repaired. Confirming
        it each time matters because a stopped owner counts as gone: a sender that resumes after
        another gateway took its lock stands by instead of reporting beside it."""
        try:
            acquired, _holder = await asyncio.to_thread(acquire_scoped_lock, _LOCK_SCOPE, self._target.url)
        except OSError:
            # Permissions (gateway.status has logged the directory and its repair), a read-only or
            # full disk, a file where the directory goes. One fixed reason: the error's text is a path.
            # A sender keeps sending: no other gateway can take the lock from this directory either.
            if self._is_sender:
                return True
            self._stand_by(logging.WARNING, "the gateway lock directory is not usable")
            return False
        self._is_sender = acquired
        if acquired:
            self._standing_by_for = None
        else:
            self._stand_by(logging.INFO, "another gateway on this machine is the sender")
        return acquired

    def _stand_by(self, level: int, why: str) -> None:
        """Say why this gateway is not the sender once per reason, not on every poll."""
        if why != self._standing_by_for:
            self._standing_by_for = why
            logger.log(level, "status report: %s; standing by", why)

    async def _next_report(self, kind: ReportKind) -> bytes:
        seq = next(self._seqs)
        source = await asyncio.to_thread(self._read_scoped_source)
        return build_report(kind, seq=seq, boot_id=self.boot_id, source=source)

    def _read_scoped_source(self) -> Mapping[str, Any]:
        # The launch profile's home alone: the status is two files under it. Its secrets and its
        # terminal policy are left unbuilt, so a .env or config.yaml that cannot be read warns on no beat.
        with bind_state(scope_for_profile(self._launch_home), secrets={}, terminal=False):
            return self._read_source()

    def _settle(self, outcome: Outcome, body: bytes, *, retried: bool) -> None:
        """Set the next beat from how this one went."""
        now = time.time()
        if outcome.kind == "stored":
            self._rejections = 0
            # A stored reply without a usable Report-Interval sets the interval back: a receiver
            # that stops sending one gets the normal interval again without a gateway restart.
            told = outcome.report_interval_s
            self._interval_s = INTERVAL_S if told is None else min(max(told, INTERVAL_MIN_S), INTERVAL_MAX_S)
        wait, held_off = _jittered(self._interval_s), False
        if outcome.kind == "throttled" and outcome.retry_after_s is not None:
            wait, held_off = min(outcome.retry_after_s, _RETRY_AFTER_MAX_S), True
        elif outcome.kind == "rejected":
            wait, held_off = _REJECTED_BACKOFF_S[min(self._rejections, len(_REJECTED_BACKOFF_S) - 1)], True
            self._rejections += 1
        elif outcome.kind == "failed" and not retried and now >= self._no_beat_before:
            # One quick retry of the same report, unless a goodbye has gone out since: its seq is
            # higher, so the receiver would drop the retry.
            wait, self._retry_body = _QUICK_RETRY_S, body
        self._receiver_holds_off = held_off
        self._due_at = max(now + wait, self._no_beat_before)
        changed, self._last_kind = outcome.kind != self._last_kind, outcome.kind
        level = logging.DEBUG if not changed else logging.INFO if outcome.kind == "stored" else logging.WARNING
        logger.log(level, "status report: %s (%s), next in %.0fs",
                   outcome.kind, outcome.status or "no reply", self._due_at - now)


def status_reporter_from_launch_env(environ: Optional[Mapping[str, str]] = None) -> Optional[StatusReporter]:
    """The reporter of this gateway, or None: the launch environment has no report URL, or this is
    a named profile's gateway. A machine that runs one gateway per profile starts them all from the
    same environment, so each would report; only the launch (default) profile's gateway does."""
    target = report_target_from_env(environ)
    if target is None:
        return None
    launch_home = get_process_hermes_home()  # launch profile, pre-identity
    if profile_name_for_home(launch_home) != "default":
        logger.info("status report: off for this gateway (only the default profile's gateway reports)")
        return None
    return StatusReporter(target, launch_home=launch_home)
