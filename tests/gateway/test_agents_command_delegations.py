"""Gateway /agents surfaces background delegations with live activity (#51690).

Drives the REAL GatewayRunner._handle_agents_command against a REAL
async-delegation registry dispatch (no mocked list function), so the test
covers the whole projection: registry record → list_async_delegations()
live sampling → /agents rendering.
"""

import threading
import time

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tools import async_delegation as ad
from tools.process_registry import process_registry

_SESSION_A = "agent:main:test:dm:1"
_SESSION_B = "agent:main:test:dm:2"


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_source(session_key: str = _SESSION_A) -> SessionSource:
    # session_key is derived by runner override; source fields are only
    # needed for platform checks (Matrix redaction) inside /agents.
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-a" if session_key == _SESSION_A else "user-b",
        chat_id="chat-a" if session_key == _SESSION_A else "chat-b",
        user_name="Alice" if session_key == _SESSION_A else "Bob",
        chat_type="dm",
    )


def _make_event(text: str = "/agents", *, session_key: str = _SESSION_A) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(session_key),
        message_id="m1",
    )


def _make_runner(*, session_key: str = _SESSION_A, is_admin: bool = False):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda _source: session_key
    runner._resume_caller_is_admin = lambda _source: is_admin
    return runner


@pytest.mark.asyncio
async def test_agents_command_marks_stalling_delegation(monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    # Long grace so the record stays in 'stalling' while we render.
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 30.0)
    gate = threading.Event()

    res = ad.dispatch_async_delegation(
        goal="wedged child", context=None, toolsets=None, role="leaf",
        model="m", session_key=_SESSION_A, max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            items = ad.list_async_delegations()
            if any(
                d["delegation_id"] == res["delegation_id"]
                and d.get("status") == "stalling"
                for d in items
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("delegation never reached stalling state")

        runner = _make_runner()
        out = await runner._handle_agents_command(_make_event())
    finally:
        gate.set()

    assert res["delegation_id"] in out
    assert "stalling" in out
    assert "no progress" in out


@pytest.mark.asyncio
async def test_agents_hides_other_sessions_delegations(monkeypatch):
    """Non-admin /agents must not list another session's async delegations."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 30.0)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 60.0)
    gate_a = threading.Event()
    gate_b = threading.Event()

    res_a = ad.dispatch_async_delegation(
        goal="mine only", context=None, toolsets=None, role="leaf",
        model="m", session_key=_SESSION_A, max_async_children=2,
        runner=lambda: {} if gate_a.wait(timeout=10) else {},
        progress_fn=lambda: ((1, "terminal"), False),
    )
    res_b = ad.dispatch_async_delegation(
        goal="secret other chat", context=None, toolsets=None, role="leaf",
        model="m", session_key=_SESSION_B, max_async_children=2,
        runner=lambda: {} if gate_b.wait(timeout=10) else {},
        progress_fn=lambda: ((1, "terminal"), False),
    )
    assert res_a["status"] == "dispatched"
    assert res_b["status"] == "dispatched"

    try:
        runner = _make_runner(session_key=_SESSION_A, is_admin=False)
        out = await runner._handle_agents_command(_make_event("/agents"))
    finally:
        gate_a.set()
        gate_b.set()

    assert res_a["delegation_id"] in out
    assert "mine only" in out
    assert res_b["delegation_id"] not in out
    assert "secret other chat" not in out


@pytest.mark.asyncio
async def test_agents_admin_all_shows_other_sessions_delegations(monkeypatch):
    """Configured slash admin with --all may see cross-session delegations."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 30.0)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 60.0)
    gate_a = threading.Event()
    gate_b = threading.Event()

    res_a = ad.dispatch_async_delegation(
        goal="admin own", context=None, toolsets=None, role="leaf",
        model="m", session_key=_SESSION_A, max_async_children=2,
        runner=lambda: {} if gate_a.wait(timeout=10) else {},
        progress_fn=lambda: ((1, "terminal"), False),
    )
    res_b = ad.dispatch_async_delegation(
        goal="other chat visible", context=None, toolsets=None, role="leaf",
        model="m", session_key=_SESSION_B, max_async_children=2,
        runner=lambda: {} if gate_b.wait(timeout=10) else {},
        progress_fn=lambda: ((1, "terminal"), False),
    )
    assert res_a["status"] == "dispatched"
    assert res_b["status"] == "dispatched"

    try:
        runner = _make_runner(session_key=_SESSION_A, is_admin=True)
        out = await runner._handle_agents_command(
            _make_event("/agents --all", session_key=_SESSION_A)
        )
    finally:
        gate_a.set()
        gate_b.set()

    assert res_a["delegation_id"] in out
    assert res_b["delegation_id"] in out
    assert "other chat visible" in out
