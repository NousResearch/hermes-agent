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
import json
import logging
import shlex
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
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
        raise TimeoutError

    runner = _make_runner_with_cached_agent(lambda: None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        with patch.object(_sc.asyncio, "wait_for", _instant_timeout):
            result = await runner._handle_reset_command(_make_event("/new"))

    assert any(
        "exceeded" in r.message and "#35994" in r.message for r in caplog.records
    ), "expected the timeout warning to be logged"
    runner.session_store.reset_session.assert_called_once()
    assert result is not None


def test_reset_broad_process_cleanup_uses_stable_session_key(monkeypatch):
    """Reset cleans non-persisted processes for the stable session_key without
    touching persisted jobs, foreign sessions, or replacement turns."""
    import tools.process_registry as process_registry_module
    from gateway.slash_commands_session import (
        _kill_processes_for_session_reset,
        _snapshot_processes_for_session_reset,
    )
    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    session_key = build_session_key(_make_source())
    target = ProcessSession(
        id="proc_old_compression_owner",
        command="sleep 60",
        task_id="default",
        owner_task_id="session-before-compression",
        session_key=session_key,
    )
    persisted = ProcessSession(
        id="proc_persisted",
        command="sleep 60",
        task_id="default",
        owner_task_id="session-before-compression",
        session_key=session_key,
        persist_on_release=True,
    )
    foreign = ProcessSession(
        id="proc_foreign",
        command="sleep 60",
        task_id="default",
        owner_task_id="other-session",
        session_key="agent:main:telegram:dm:other",
    )
    registry._running = {
        target.id: target,
        persisted.id: persisted,
        foreign.id: foreign,
    }
    kill_calls = []

    def fake_kill(session_id, **kwargs):
        kill_calls.append((session_id, kwargs))
        registry._running[session_id].exited = True
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)
    monkeypatch.setattr(process_registry_module, "process_registry", registry)

    reset_snapshot = _snapshot_processes_for_session_reset(session_key)
    assert reset_snapshot == (target.id,)

    replacement = ProcessSession(
        id="proc_replacement_turn",
        command="sleep 60",
        task_id="default",
        owner_task_id="replacement-session",
        session_key=session_key,
    )
    registry._running[replacement.id] = replacement

    assert _kill_processes_for_session_reset(reset_snapshot) == 1

    assert target.exited is True
    assert persisted.exited is False
    assert foreign.exited is False
    assert replacement.exited is False
    assert kill_calls == [
        (target.id, {"source": "kill_all", "consume_output": True}),
    ]


@pytest.mark.asyncio
@pytest.mark.platforms("linux")
async def test_busy_reset_kills_background_process_registered_after_snapshot(
    monkeypatch, tmp_path,
):
    """A busy /new and a blocked local spawn cannot straddle the reset boundary."""
    import tools.process_registry as process_registry_module
    import tools.terminal_tool as terminal_tool_module
    from tools import terminal_tool_sudo
    from tools.environments import local as local_env
    from tools.interrupt import set_interrupt
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.setattr(terminal_tool_sudo, "_rewrite_compound_background", lambda c: c)

    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(process_registry_module, "process_registry", registry)

    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    interrupt_requested = threading.Event()
    reset_snapshots = []
    created = {}
    real_spawn_local = registry.spawn_local
    real_snapshot = registry.snapshot_running_ids_for_session

    def blocked_spawn_local(*args, **kwargs):
        spawn_entered.set()
        if not release_spawn.wait(timeout=10):
            raise RuntimeError("reset never released the blocked spawn")
        session = real_spawn_local(*args, **kwargs)
        created["session"] = session
        return session

    def snapshot_then_release(session_key):
        snapshot = real_snapshot(session_key)
        reset_snapshots.append(snapshot)
        release_spawn.set()
        return snapshot

    monkeypatch.setattr(registry, "spawn_local", blocked_spawn_local)
    monkeypatch.setattr(registry, "snapshot_running_ids_for_session", snapshot_then_release)

    config = {
        "env_type": "local",
        "cwd": str(tmp_path),
        "timeout": 60,
        "lifetime_seconds": 3600,
    }
    environment = SimpleNamespace(env={}, cwd=str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "_active_environments", {"default": environment})
    monkeypatch.setattr(terminal_tool_module, "_last_activity", {})
    monkeypatch.setattr(terminal_tool_module, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool_module, "_container_aliases", {})
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_a, **_k: {"approved": True})
    session_key = build_session_key(_make_source())
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": session_key)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    tool_result = {}
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(60)')}"

    def run_background_tool():
        set_interrupt(False)
        try:
            tool_result["value"] = terminal_tool_module.terminal_tool(
                command=command,
                background=True,
                notify_on_complete=True,
                task_id="sess-old",
            )
        finally:
            set_interrupt(False)

    tool_thread = threading.Thread(target=run_background_tool, daemon=True)
    tool_thread.start()
    assert await asyncio.to_thread(spawn_entered.wait, 5), (
        "background tool never reached its spawn boundary"
    )

    class BusyAgent:
        _gateway_turn_process_task_id = "sess-old"
        _gateway_turn_process_baseline = frozenset()

        def __init__(self):
            self.interrupt_reasons = []

        def hard_interrupt(self, reason=None):
            self.interrupt_reasons.append(reason)
            set_interrupt(True, thread_id=tool_thread.ident)
            interrupt_requested.set()

        def release_clients(self):
            return None

    agent = BusyAgent()
    runner = _make_runner_with_cached_agent(lambda: None)
    runner._agent_cache = {session_key: agent}
    runner._session_state(session_key).turn.agent = agent
    runner._persist_active_agents = lambda: None

    try:
        reset_result = await asyncio.wait_for(
            runner._busy_new_command(_make_event("/new"), session_key, _make_source()),
            timeout=10,
        )
    finally:
        release_spawn.set()
        await asyncio.to_thread(tool_thread.join, 15)
        set_interrupt(False, thread_id=tool_thread.ident)
        registry.kill_all(source="test_cleanup")

    assert not tool_thread.is_alive(), "background tool did not drain after /new"
    assert interrupt_requested.is_set()
    assert agent.interrupt_reasons
    assert reset_snapshots == [()]
    assert reset_result is not None

    result = json.loads(tool_result["value"])
    session = created["session"]
    assert result["exit_code"] == 130, result
    assert result["output"] == "[Command interrupted]"
    assert "session_id" not in result
    assert session.exited is True
    assert session.completion_reason == "killed"
    assert session.termination_source == "terminal_interrupt"
    assert registry.is_completion_consumed(session.id) is True
    assert registry.pending_watchers == []
    assert registry.completion_queue.empty()
    assert session.process is not None
    assert session.process.wait(timeout=5) is not None
