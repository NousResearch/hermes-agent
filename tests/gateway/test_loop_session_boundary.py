"""Session-boundary regression and in-process E2E coverage for gateway /loop.

The end-to-end test uses the real SessionStore, persistent LoopManager state,
and GatewayRunner wakeup watcher against a temporary HERMES_HOME.  The adapter
is deliberately a local spy: the invariant under test is that a stale loop
never reaches an outbound platform adapter after the route rotates.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_cli import goals, loops


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="loop-channel",
        chat_type="channel",
        thread_id="loop-thread",
        user_id="loop-user",
        user_name="Loop owner",
    )


def _route() -> dict[str, str]:
    return {
        "platform": "discord",
        "chat_id": "loop-channel",
        "chat_type": "channel",
        "thread_id": "loop-thread",
        "user_id": "loop-user",
        "user_name": "Loop owner",
    }


@pytest.fixture
def loop_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_store(tmp_path) -> SessionStore:
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="test-token")}
    )
    return SessionStore(sessions_dir=tmp_path / "sessions", config=config)


def _make_runner(store: SessionStore, adapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running = True
    runner._running_agents = {}
    return runner


def _set_due_loop(session_id: str) -> None:
    manager = loops.LoopManager(session_id=session_id)
    state = manager.set("check the deployment", interval_seconds=60, route=_route())
    state.next_due_at = time.time() - 1
    loops.save_loop(session_id, state)


async def _run_one_watcher_scan(monkeypatch, runner: GatewayRunner) -> None:
    """Run the watcher's post-startup scan exactly once without wall-clock wait."""
    sleep_calls = 0

    async def _sleep_once(_delay: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            runner._running = False

    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep_once)
    await runner._loop_wakeup_watcher(interval=0)


@pytest.mark.asyncio
async def test_end_to_end_stale_loop_is_not_injected_after_reset(loop_home, monkeypatch):
    """A due S1 loop must not become a turn in the replacement S2 session."""
    store = _make_store(loop_home)
    first_entry = store.get_or_create_session(_source())
    _set_due_loop(first_entry.session_id)

    replacement = store.reset_session(first_entry.session_key)
    assert replacement is not None
    assert replacement.session_id != first_entry.session_id

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _make_runner(store, adapter)
    await _run_one_watcher_scan(monkeypatch, runner)

    adapter.handle_message.assert_not_awaited()
    stale = loops.load_loop(first_entry.session_id)
    assert stale is not None
    assert stale.status == "cleared"
    assert stale.awaiting_response is False


@pytest.mark.asyncio
async def test_watcher_retains_due_loop_when_session_lookup_fails(loop_home, monkeypatch):
    """A transient SessionStore failure must not destroy a healthy loop."""
    store = _make_store(loop_home)
    entry = store.get_or_create_session(_source())
    _set_due_loop(entry.session_id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _make_runner(store, adapter)
    runner._async_session_store.lookup_by_session_key = AsyncMock(
        side_effect=RuntimeError("temporary database failure")
    )
    await _run_one_watcher_scan(monkeypatch, runner)

    adapter.handle_message.assert_not_awaited()
    retained = loops.load_loop(entry.session_id)
    assert retained is not None
    assert retained.status == "active"
    assert retained.awaiting_response is False


@pytest.mark.asyncio
async def test_watcher_retains_due_loop_when_route_resolution_raises(loop_home, monkeypatch):
    """An unverified route failure stays due for a later retry."""
    store = _make_store(loop_home)
    entry = store.get_or_create_session(_source())
    _set_due_loop(entry.session_id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _make_runner(store, adapter)

    def _raise_route_error(_source):
        raise ValueError("temporary route failure")

    monkeypatch.setattr(runner, "_session_key_for_source", _raise_route_error)
    await _run_one_watcher_scan(monkeypatch, runner)

    adapter.handle_message.assert_not_awaited()
    retained = loops.load_loop(entry.session_id)
    assert retained is not None
    assert retained.status == "active"
    assert retained.awaiting_response is False


@pytest.mark.asyncio
async def test_watcher_stamps_each_wakeup_with_its_exact_session(loop_home, monkeypatch):
    """The adapter receives an event that the downstream strict gate can bind."""
    store = _make_store(loop_home)
    entry = store.get_or_create_session(_source())
    _set_due_loop(entry.session_id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _make_runner(store, adapter)
    await _run_one_watcher_scan(monkeypatch, runner)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.metadata == {
        "gateway_session_key": entry.session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
        "hermes_loop_wakeup": True,
    }


def test_rejected_strict_wakeup_clears_claimed_loop(loop_home):
    """A reset racing after watcher verification cannot leave an inert loop."""
    session_id = "loop-race-session"
    _set_due_loop(session_id)
    manager = loops.LoopManager(session_id=session_id)
    assert manager.fire_tick() is not None
    assert manager.state is not None and manager.state.awaiting_response is True

    GatewayRunner._clear_rejected_loop_wakeup(
        {"hermes_loop_wakeup": True},
        session_id,
    )

    cleared = loops.load_loop(session_id)
    assert cleared is not None
    assert cleared.status == "cleared"
    assert cleared.awaiting_response is False
