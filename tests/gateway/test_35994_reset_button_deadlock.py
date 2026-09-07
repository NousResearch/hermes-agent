"""Regression test for #35994: Telegram /new confirm-button deadlock.

The /new confirmation button callback runs the slash-confirm handler on the
asyncio event loop (see GatewayRunner._request_slash_confirm). That handler
calls _handle_reset_command, which used to invoke the SYNCHRONOUS, potentially
long-blocking _cleanup_agent_resources (agent.close() tears down terminal
sandboxes / browser daemons / background processes; shutdown_memory_provider()
may make a network call) inline on the loop. A slow teardown wedged the entire
event loop, so the bot went silent until a manual restart.

The fix offloads _cleanup_agent_resources to a worker thread with a bounded
timeout, so the loop is never blocked and a stuck teardown degrades gracefully.
"""
import asyncio
import logging
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner_with_cached_agent(close_fn):
    """Build a bare GatewayRunner with a cached agent whose close() runs
    ``close_fn`` (used to simulate slow / blocking teardown)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key, session_id="sess-old",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    new_entry = SessionEntry(
        session_key=session_key, session_id="sess-new",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.reset_session.return_value = new_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    # Enable the cache-lock path (this is what the button callback exercises)
    runner._agent_cache_lock = threading.RLock()
    agent = MagicMock()
    agent.close = close_fn
    agent.shutdown_memory_provider = MagicMock()
    runner._agent_cache = {session_key: agent}
    return runner


@pytest.mark.asyncio
async def test_reset_does_not_block_event_loop_during_cleanup():
    """#35994: a slow agent.close() must NOT block the event loop. A
    concurrent loop task must keep ticking WHILE close() is still blocking
    (proving cleanup was offloaded to a worker thread, not run inline on
    the loop). With the pre-fix inline call, the loop is frozen for the
    whole duration of close() and no ticks accumulate until it returns."""
    close_started = threading.Event()
    release = threading.Event()

    def slow_close():
        close_started.set()
        # Block the WORKER thread (not the loop) until released.
        release.wait(timeout=5)

    runner = _make_runner_with_cached_agent(slow_close)

    ticks = {"n": 0}
    stop = threading.Event()

    async def _heartbeat():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    hb = asyncio.create_task(_heartbeat())
    reset_task = asyncio.create_task(
        runner._handle_reset_command(_make_event("/new"))
    )

    # Wait until close() has actually started blocking in its worker thread.
    for _ in range(200):
        if close_started.is_set():
            break
        await asyncio.sleep(0.005)
    assert close_started.is_set(), "close() never ran"

    # Now sample ticks while close() is STILL blocking. If the loop were
    # frozen (pre-fix inline call), this stays ~0.
    ticks_at_block = ticks["n"]
    await asyncio.sleep(0.1)
    ticks_during_block = ticks["n"] - ticks_at_block

    release.set()
    await reset_task
    stop.set()
    await hb

    assert ticks_during_block >= 5, (
        f"event loop was blocked during agent cleanup (#35994): only "
        f"{ticks_during_block} ticks while close() was running"
    )
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_reset_completes_when_cleanup_raises(caplog):
    """#35994: if the offloaded cleanup itself raises, the handler swallows it
    (logs a warning) and still rotates the session — it must not abort /new.

    Note: _cleanup_agent_resources swallows its own internal errors, so to
    exercise the handler's `except Exception` branch we make the cleanup call
    itself raise (patched on the instance), then assert the warning fired —
    proving the branch executed rather than the success path.
    """
    runner = _make_runner_with_cached_agent(lambda: None)

    def boom_cleanup(_agent):
        raise RuntimeError("cleanup blew up")

    runner._cleanup_agent_resources = boom_cleanup

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        result = await asyncio.wait_for(
            runner._handle_reset_command(_make_event("/new")), timeout=3
        )

    assert any(
        "failed during /new reset" in r.message and "#35994" in r.message
        for r in caplog.records
    ), "expected the cleanup-failure warning to be logged (except branch not hit)"
    runner.session_store.reset_session.assert_called_once()
    assert result is not None


@pytest.mark.asyncio
async def test_reset_completes_when_cleanup_times_out(caplog):
    """#35994: if cleanup exceeds the bounded timeout, the reset still completes
    (graceful degradation) and the timeout warning fires."""
    import gateway.slash_commands as _sc

    # Force the wait_for to time out immediately, closing the offloaded awaitable
    # so no worker thread dangles past the test.
    async def _instant_timeout(aw, timeout=None):
        if asyncio.iscoroutine(aw):
            aw.close()
        raise asyncio.TimeoutError

    runner = _make_runner_with_cached_agent(lambda: None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        with patch.object(_sc.asyncio, "wait_for", _instant_timeout):
            result = await runner._handle_reset_command(_make_event("/new"))

    assert any(
        "exceeded" in r.message and "#35994" in r.message for r in caplog.records
    ), "expected the timeout warning to be logged"
    runner.session_store.reset_session.assert_called_once()
    assert result is not None
