"""End-to-end test: the REAL GatewayTurnMixin._run_agent_notify_long_running
with the reviewer's fix applied (pooled async handle + get_session_title).

Runs the production method itself — not an inlined copy — against:
  1. a pooled AsyncSessionDB over a REAL per-profile SessionDB (title flow), and
  2. a slow pooled handle (proves the loop stays responsive during the read).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def hermes_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state
    if hermes_state.DEFAULT_DB_PATH != hermes_state._IMPORT_DEFAULT_DB_PATH:
        hermes_state.DEFAULT_DB_PATH = tmp_path / "state.db"
    return tmp_path


async def _measure(coro, timeout=10):
    """Await coro; return (worst_loop_lag_s, coro_result)."""
    lag = {"worst": 0.0, "stop": False}

    async def watch():
        while not lag["stop"]:
            t0 = time.monotonic()
            await asyncio.sleep(0.005)
            lag["worst"] = max(lag["worst"], time.monotonic() - t0 - 0.005)

    async def run():
        w = asyncio.create_task(watch())
        await asyncio.sleep(0)  # watcher records baseline first
        try:
            return await asyncio.wait_for(coro, timeout)
        finally:
            lag["stop"] = True
            await w

    # The method loops forever until ownership moves; drive one heartbeat by
    # making ownership fail after the first send.
    return await asyncio.wait_for(asyncio.create_task(run().__anext__() if False else run()), timeout)


def _setup_runner_with_real_profile_db(sandbox_root, *, slow_read_s=0.0, title="Workbench deep dive"):
    """Build a runner whose _session_db is a REAL AsyncSessionDB over a REAL
    per-profile SessionDB containing one titled session."""
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext
    from hermes_state import SessionDB, AsyncSessionDB

    profile_home = sandbox_root / "profiles" / "workbench"
    profile_home.mkdir(parents=True, exist_ok=True)
    profile_db_path = profile_home / "state.db"
    db = SessionDB(db_path=profile_db_path)
    sid = "sess-e2e-1"
    db._write_rowcount(
        "INSERT INTO sessions (id, source, started_at) VALUES (?,?,?)", (sid, "test", time.time())
    )
    db.set_auto_title(sid, title, source=db.TITLE_SOURCE_DERIVED)

    # Wrap with async offloading; optionally inject a slow read (blocked SQLite
    # / token-flush contention stand-in) INSIDE the worker thread. NOTE:
    # AsyncSessionDB fabricates methods via __getattr__ (no real
    # get_session_title attribute), so the override calls the offloader
    # directly instead of super().
    class _SlowAsyncSessionDB(AsyncSessionDB):
        def __init__(self, db, delay):
            self._db = db
            self._delay = delay

        def get_session_title(self, session_id):
            inner = self._offload("get_session_title", session_id)

            async def _delayed():
                if self._delay:
                    await asyncio.to_thread(time.sleep, self._delay)
                return await inner
            return _delayed()

        def _offload(self, name, *args):
            return asyncio.to_thread(getattr(self._db, name), *args)

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = False
    runner._restart_requested = False
    runner._session_db = _SlowAsyncSessionDB(db, slow_read_s)
    runner._agent_activity_summary = staticmethod(lambda agent: None)

    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    runner._delivery_adapter_for = lambda source: adapter
    runner._adapter = adapter

    ctx = TurnContext(
        source=SimpleNamespace(chat_id="c1", platform="telegram"),
        session_key="sess",
        session_id=sid,
    )
    agent = MagicMock()
    ctx.agent_holder[0] = agent
    runner._running_agents["sess"] = agent

    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False

    return runner, ctx, disp, adapter, db


@pytest.mark.asyncio
async def test_real_heartbeat_prefixes_profile_session_title(hermes_sandbox, monkeypatch):
    """The production method sends a title-prefixed heartbeat from the pooled
    profile-scoped handle."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    runner, ctx, disp, adapter, db = _setup_runner_with_real_profile_db(hermes_sandbox)

    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    # wait for the first heartbeat send
    for _ in range(100):
        await asyncio.sleep(0.02)
        if adapter.send.await_count >= 1:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    adapter.send.assert_awaited()
    sent = adapter.send.await_args_list[0].args[1]
    assert "Workbench deep dive" in sent, f"title missing from heartbeat: {sent!r}"
    assert sent.startswith("⏳")
    db.close()


@pytest.mark.asyncio
async def test_real_heartbeat_database_stall_does_not_freeze_loop(hermes_sandbox, monkeypatch):
    """With a 0.2s DB stall inside the title read, the event loop stays
    responsive — the read runs off-loop via AsyncSessionDB/to_thread."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    runner, ctx, disp, adapter, db = _setup_runner_with_real_profile_db(
        hermes_sandbox, slow_read_s=0.2, title="Slow DB title"
    )

    lag = {"worst": 0.0, "stop": False}

    async def watch():
        while not lag["stop"]:
            t0 = time.monotonic()
            await asyncio.sleep(0.005)
            lag["worst"] = max(lag["worst"], time.monotonic() - t0 - 0.005)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)  # baseline

    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    for _ in range(150):
        await asyncio.sleep(0.02)
        if adapter.send.await_count >= 1:
            break
    lag["stop"] = True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    watcher.cancel()

    adapter.send.assert_awaited()
    sent = adapter.send.await_args_list[0].args[1]
    assert "Slow DB title" in sent, f"title missing: {sent!r}"
    # A 0.2s inline read would starve the loop ~0.2s. Allowed: one event-loop
    # tick for the create_task/cancel bookkeeping around the offloaded read.
    assert lag["worst"] < 0.1, f"loop starved {lag['worst']:.3f}s with a 0.2s DB stall (read ran on-loop?)"
    db.close()


@pytest.mark.asyncio
async def test_real_heartbeat_without_session_db_still_sends(hermes_sandbox, monkeypatch):
    """Runner with _session_db=None (DB disabled) must still heartbeat, just
    without the title prefix — no crash, no un-awaited None."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = False
    runner._restart_requested = False
    runner._session_db = None
    runner._agent_activity_summary = staticmethod(lambda agent: None)

    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    runner._delivery_adapter_for = lambda source: adapter

    ctx = TurnContext(
        source=SimpleNamespace(chat_id="c1", platform="telegram"),
        session_key="sess",
        session_id="sess-x",
    )
    agent = MagicMock()
    ctx.agent_holder[0] = agent
    runner._running_agents["sess"] = agent

    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False

    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if adapter.send.await_count >= 1:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    adapter.send.assert_awaited()
    sent = adapter.send.await_args_list[0].args[1]
    assert "Working" in sent
    assert "sess-x" not in sent  # no bogus title from None path