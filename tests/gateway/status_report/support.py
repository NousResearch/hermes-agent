"""Stand-ins for the status report tests: a scripted receiver in place of urllib's opener, a wall
clock the test moves, and the contract fixtures in ``tests/fixtures/status_report/v1``."""

from __future__ import annotations

import asyncio
import email.message
import io
import json
import re
import secrets
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gateway.status_report.reporter import StatusReporter
from gateway.status_report.target import ReportTarget
from hermes_constants import get_process_hermes_home

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "status_report" / "v1"
REPORT_URL = "https://status.example/v1/gateways/report"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


RESPONSES = fixture("responses.json")
INTERVAL_S = RESPONSES["interval_s"]
# One second outside the jitter band on either side: no report before, one after.
BEFORE_INTERVAL_S = INTERVAL_S * (1 - RESPONSES["jitter"]) - 1
AFTER_INTERVAL_S = INTERVAL_S * (1 + RESPONSES["jitter"]) + 1


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

    def reply_with(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append(Seen(request.full_url, request.get_method(), dict(request.header_items()),
                                  body, request.data, timeout))
        if body.get("kind") == "goodbye":
            self.goodbye_arrived.set()
        reply = self.replies.pop(0) if self.replies else Reply()
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
    """Let every task this test started finish, so none is still running when the loop closes."""
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


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
