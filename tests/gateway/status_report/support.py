"""Stand-ins for the status report tests: a scripted receiver in place of urllib's opener, a wall
clock the test moves, and the contract fixtures in ``tests/fixtures/status_report/v1``."""

from __future__ import annotations

import asyncio
import email.message
import io
import json
import logging
import re
import secrets
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from gateway.status_report.reporter import StatusReporter
from gateway.status_report.target import ReportTarget
from hermes_constants import get_process_hermes_home
from tests.waiting import async_wait_until

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "status_report" / "v1"
REPORT_URL = "https://status.example/v1/gateways/report"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


RESPONSES = fixture("responses.json")
INTERVAL_S = RESPONSES["interval_s"]
INTERVAL_HEADER = RESPONSES["interval_header"]


def around(interval_s: float) -> tuple[float, float]:
    """One second outside the jitter band of *interval_s* on either side: no report before, one after."""
    return interval_s * (1 - RESPONSES["jitter"]) - 1, interval_s * (1 + RESPONSES["jitter"]) + 1


AFTER_INTERVAL_S = around(INTERVAL_S)[1]


def new_credential() -> str:
    """A credential made at test time, so no credential-shaped literal is committed."""
    return secrets.token_urlsafe(24)


def stamped_url(credential: str, url: str = REPORT_URL) -> str:
    return f"{url}?t={credential}"


@dataclass
class Reply:
    status: int = 204
    headers: dict = field(default_factory=dict)
    body: Any = None
    error: Optional[BaseException] = None  # raised instead of answering: a network error
    hang: Optional[threading.Event] = None  # never answers until the test sets it, then times out


@dataclass
class Seen:
    url: str
    method: str
    headers: dict
    body: Any
    raw: bytes
    timeout: Optional[float]


class _UnreadBody(io.BytesIO):
    def read(self, *args) -> bytes:
        raise AssertionError("the sender read the reply body")


class _Response:
    def __init__(self, reply: Reply, headers: email.message.Message) -> None:
        self.status, self.headers = reply.status, headers

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def read(self, *args) -> bytes:
        raise AssertionError("the sender read the reply body")


class Receiver:
    """Called where urllib's opener would be: records each request, answers from a script
    (``204`` once the script runs out), and fails the way urllib does for a non-2xx status."""

    def __init__(self) -> None:
        self.requests: list[Seen] = []
        self.replies: list[Reply] = []
        self.goodbye_arrived = threading.Event()
        self.answered = 0  # requests that got their reply, error or timeout

    def reply_with(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append(Seen(request.full_url, request.get_method(), dict(request.header_items()),
                                  body, request.data, timeout))
        if body.get("kind") == "goodbye":
            self.goodbye_arrived.set()
        reply = self.replies.pop(0) if self.replies else Reply()
        try:
            return self._answer(request, reply)
        finally:
            self.answered += 1

    def _answer(self, request, reply: Reply):
        if reply.hang is not None:
            reply.hang.wait()
            raise TimeoutError("timed out")
        if reply.error is not None:
            raise reply.error
        headers = email.message.Message()
        for name, value in reply.headers.items():
            headers[name] = value
        if not 200 <= reply.status < 300:
            raise urllib.error.HTTPError(request.full_url, reply.status, "status", headers, _UnreadBody())
        return _Response(reply, headers)

    @property
    def kinds(self) -> list[str]:
        return [seen.body["kind"] for seen in self.requests]


class Clock:
    """The wall clock, moved by the test. asyncio runs on the monotonic clock, so the loop is unaffected."""

    def __init__(self, monkeypatch, start: float = 1_900_000_000.0) -> None:
        self.now = start
        monkeypatch.setattr(time, "time", lambda: self.now)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def a_status_source() -> dict:
    """What the gateway knows about itself, with the extras a status body also carries."""
    return {
        "gateway_running": True, "gateway_state": "running", "gateway_exit_reason": None,
        "gateway_platforms": {"telegram": {"state": "connected"}},
        "nous_session_valid": "valid",
        "install_id": "install-0001", "profiles": ["default", "work"], "active_sessions": 3,
        "version": "9.9.9", "hermes_home": "/srv/hermes",
    }


def make_reporter(receiver: Receiver, *, source=a_status_source, credential: Optional[str] = None,
                  url: str = REPORT_URL) -> StatusReporter:
    return StatusReporter(ReportTarget(url, credential or new_credential()),
                          launch_home=get_process_hermes_home(), read_source=source, opener=receiver)


async def settle(timeout: float = 10.0) -> None:
    """Let every task this test started finish, so none is still running when the loop closes.
    Fails when one is still running at *timeout*: a hung goodbye or watcher must not pass."""
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if pending:
        _done, still_running = await asyncio.wait(pending, timeout=timeout)
        assert not still_running, f"{len(still_running)} task(s) still running after {timeout}s: {still_running!r}"


def assert_matches_contract(body: dict, kind: str) -> None:
    """*body* has the shape of the v1 fixtures: ``beat-minimal.json`` names what is required,
    ``beat.json`` / ``goodbye.json`` everything that is allowed."""
    full, minimal = fixture("goodbye.json" if kind == "goodbye" else "beat.json"), fixture("beat-minimal.json")
    assert set(body) == set(full) == set(minimal)
    assert body["v"] == full["v"] == 1
    assert body["kind"] == full["kind"] == kind
    assert type(body["seq"]) is int and body["seq"] >= 1
    for boot_id in (body["boot_id"], full["boot_id"]):
        assert re.fullmatch(r"[0-9a-f]{32}", boot_id)
    status, allowed = body["status"], fixture("beat.json")["status"]
    assert set(minimal["status"]) <= set(status) <= set(allowed)
    assert status["gateway_running"] is True
    for key in ("gateway_state", "gateway_exit_reason"):
        assert status[key] is None or isinstance(status[key], str)
    entry_keys = {key for entry in allowed["gateway_platforms"].values() for key in entry}
    for key, entry in status["gateway_platforms"].items():
        assert isinstance(key, str) and set(entry) <= entry_keys
        assert all(isinstance(value, str) for value in entry.values())
    if "nous_session_valid" in status:
        assert status["nous_session_valid"] in {"valid", "terminal", "unknown"}


QUICK_RETRY_S = RESPONSES["quick_retry_s"]
# An interval a receiver sets, far enough from the normal one that their jitter bands do not touch.
TOLD_S = 300
# What the OS puts into an error's text: the path it failed on.
FAULT_PATH = "/mnt/state/gateway-locks/status-report.lock"


def cases(then: str) -> list:
    return [pytest.param(case, id=case["name"]) for case in RESPONSES["cases"] if case["then"] == then]


def reply_for(case: dict) -> Reply:
    return Reply(case["status"], case["headers"], case.get("body"))


def told(seconds: int) -> Reply:
    """A stored reply that sets the report interval."""
    return Reply(204, {INTERVAL_HEADER: str(seconds)})


async def expect_next_report_at_the_interval(clock, reporter, receiver, interval_s: float = INTERVAL_S) -> None:
    before, after = around(interval_s)
    sent = len(receiver.requests)
    clock.advance(before)
    await reporter.poll()
    assert len(receiver.requests) == sent, f"a report went out before the {interval_s}s interval"
    clock.advance(after - before)
    await reporter.poll()
    assert len(receiver.requests) == sent + 1, f"no report at the {interval_s}s interval"


def warnings_said(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name.startswith("gateway.status_report") and record.levelno >= logging.WARNING]


class Watcher:
    """``run()`` on its own task, as the gateway's supervisor starts it. Cancelled when the test leaves."""

    def __init__(self, reporter: StatusReporter) -> None:
        self.polls = 0
        self.task = asyncio.create_task(reporter.run(self._is_running, tick=0.01))

    def _is_running(self) -> bool:
        self.polls += 1
        return True

    async def __aenter__(self) -> Watcher:
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.task.cancel()
        await settle()

    async def polls_on(self) -> None:
        """Let a few more polls pass; fail if the watcher has ended instead."""
        wanted = self.polls + 3
        await async_wait_until(lambda: self.task.done() or self.polls >= wanted, timeout=5, what="the next polls")
        assert not self.task.done(), f"the watcher ended: {self.task.exception()!r}"

    async def ends_when_cancelled_mid_beat(self, clock: Clock, receiver: Receiver) -> None:
        """A gateway shutdown cancels the watcher, and a beat may be on the wire when it does."""
        hang, sent = threading.Event(), len(receiver.requests)
        receiver.reply_with(Reply(hang=hang))
        clock.advance(AFTER_INTERVAL_S)
        try:
            await async_wait_until(lambda: len(receiver.requests) == sent + 1, timeout=5, what="a beat on the wire")
            self.task.cancel()
            await asyncio.wait({self.task}, timeout=5)
            assert self.task.cancelled(), "cancellation did not end the watcher"
        finally:
            hang.set()
