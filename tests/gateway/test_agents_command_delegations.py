"""Gateway /agents surfaces background delegations with live activity (#51690).

Drives the REAL GatewayRunner._handle_agents_command against a REAL
async-delegation registry dispatch (no mocked list function), so the test
covers the whole projection: registry record → list_async_delegations()
live sampling → /agents rendering.
"""

import threading
import time
from contextlib import contextmanager

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key
from tools import async_delegation as ad
from tools.process_registry import process_registry
from tests.waiting import wait_until


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    wait_until(lambda: ad.active_count() == 0, what="delegations finished")
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: "agent:main:test:dm:1"
    return runner


class _Event:
    source = None


@pytest.mark.asyncio
@pytest.mark.parametrize("discord", [False, True])
async def test_agents_command_marks_stalling_delegation(monkeypatch, discord):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    # Force-finalization is a separate contract. Disable it so the stalling
    # projection stays observable until this test releases the worker.
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", float("inf"))
    gate = threading.Event()
    stalling = threading.Event()
    source = SessionSource(platform=Platform.DISCORD, chat_id="111111111111111111", profile="research")
    key = build_session_key(source, profile=source.profile)

    def blocked_runner():
        assert gate.wait(30), "test did not release delegation"
        return {}

    res = ad.dispatch_async_delegation(
        goal="wedged child", context=None, toolsets=None, role="leaf",
        model="m", session_key=key, max_async_children=1,
        runner=blocked_runner,
        interrupt_fn=stalling.set,
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    try:
        assert stalling.wait(30.0), "delegation never reached stalling state"

        runner = _make_runner()
        runner._cache_session_source(key, source)
        event = MessageEvent(text="/agents", source=source) if discord else _Event()
        out = await runner._handle_agents_command(event)
    finally:
        gate.set()

    assert res["delegation_id"] in out
    assert "stalling" in out
    assert "no progress" in out
    assert ("<#111111111111111111>" in out) is discord


@contextmanager
def _running_delegation(session_key):
    gate = threading.Event()

    def blocked_runner():
        assert gate.wait(30), "test did not release delegation"
        return {}

    result = ad.dispatch_async_delegation(
        goal="inspect background work", context=None, toolsets=None, role="leaf",
        model="m", session_key=session_key, max_async_children=1,
        runner=blocked_runner,
        progress_fn=lambda: (((7, "terminal", time.time()),), True),
    )
    assert result["status"] == "dispatched"
    try:
        yield result["delegation_id"]
    finally:
        gate.set()
        wait_until(lambda: ad.active_count() == 0, what="delegation finished")


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "research"])
@pytest.mark.parametrize("thread_id,prospective_thread_id,expected", [
    ("222222222222222222", "333333333333333333", "222222222222222222"),
    (None, "333333333333333333", "333333333333333333"),
    (None, None, "111111111111111111"),
])
async def test_discord_delegation_links_inactive_parent_conversation(
    profile, thread_id, prospective_thread_id, expected,
):
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="111111111111111111",
        thread_id=thread_id, prospective_thread_id=prospective_thread_id,
        parent_chat_id="444444444444444444", profile=profile,
    )
    runner = _make_runner()
    key = build_session_key(source, profile=source.profile)
    runner._cache_session_source(key, source)
    request = MessageEvent(text="/agents", source=SessionSource(platform=Platform.DISCORD, chat_id="555555555555555555"))

    with _running_delegation(key) as delegation_id:
        out = await runner._handle_agents_command(request)

    assert f"- `{delegation_id}` · running" in out
    assert f"<#{expected}>" in out
    assert key not in out
    assert out.count("<#") == 1
    assert "<#444444444444444444>" not in out
    assert "inspect background work" in out
    assert "child 1: 7 api calls · `terminal` · active 0s ago" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "missing-cache", "missing-profile", "missing-destination", "other-source", "empty-key",
    "telegram-request", "slack-request", "local-request", "no-source",
])
async def test_delegation_conversation_fallbacks_and_non_discord_output(case):
    source = SessionSource(
        platform=Platform.TELEGRAM if case == "other-source" else Platform.DISCORD,
        chat_id="111111111111111111", thread_id="222222222222222222", profile="research",
    )
    key = "" if case == "empty-key" else build_session_key(source, profile=source.profile)
    runner = _make_runner()
    if case == "missing-destination":
        runner._cache_session_source(key, SessionSource(platform=Platform.DISCORD, chat_id=""))
    elif case == "missing-profile":
        runner._cache_session_source(build_session_key(source), source)
    elif case != "missing-cache":
        runner._cache_session_source(key, source)
    platform = {
        "telegram-request": Platform.TELEGRAM, "slack-request": Platform.SLACK,
        "local-request": Platform.LOCAL,
    }.get(case, Platform.DISCORD)
    request = _Event() if case == "no-source" else MessageEvent(
        text="/agents", source=SessionSource(platform=platform, chat_id="555555555555555555"),
    )

    with _running_delegation(key) as delegation_id:
        out = await runner._handle_agents_command(request)

    expected_row = f"- `{delegation_id}` · running · inspect background work"
    if platform == Platform.DISCORD and key and case != "no-source":
        expected_row += f" · `{key}`"
    assert expected_row in out.splitlines()
    assert "<#" not in out
    assert "child 1: 7 api calls · `terminal` · active 0s ago" in out
