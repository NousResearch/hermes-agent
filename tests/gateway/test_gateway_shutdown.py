import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import MessageEvent
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_cancel_background_tasks_cancels_inflight_message_processing():
    _runner, adapter = make_restart_runner()
    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    session_key = build_session_key(event.source)
    assert session_key in adapter._active_sessions
    assert adapter._background_tasks

    await adapter.cancel_background_tasks()

    assert adapter._background_tasks == set()
    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


def test_cleanup_agent_resources_reaps_stale_aux_clients():
    runner, _adapter = make_restart_runner()
    agent = MagicMock()

    with patch("agent.auxiliary_client.cleanup_stale_async_clients") as cleanup_mock:
        runner._cleanup_agent_resources(agent)

    agent.shutdown_memory_provider.assert_called_once()
    agent.close.assert_called_once()
    cleanup_mock.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_running_agents_and_cancels_adapter_tasks():
    runner, adapter = make_restart_runner()
    runner._pending_messages = {"session": "pending text"}
    runner._pending_approvals = {"session": {"command": "rm -rf /tmp/x"}}
    runner._restart_drain_timeout = 0.0

    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")
    await adapter.handle_message(event)
    await asyncio.sleep(0)

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
        patch("agent.auxiliary_client.shutdown_cached_clients") as shutdown_cached_clients,
    ):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    shutdown_cached_clients.assert_called_once()
    assert runner.adapters == {}
    assert runner._running_agents == {}
    assert runner._pending_messages == {}
    assert runner._pending_approvals == {}
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_drains_running_agents_before_disconnect():
    runner, adapter = make_restart_runner()
    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    running_agent = MagicMock()
    runner._running_agents = {"session": running_agent}

    async def finish_agent():
        await asyncio.sleep(0.05)
        runner._running_agents.clear()

    asyncio.create_task(finish_agent())

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_not_called()
    disconnect_mock.assert_awaited_once()
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_after_drain_timeout():
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.05

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    running_agent = MagicMock()
    runner._running_agents = {"session": running_agent}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_service_restart_sets_named_exit_code():
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop(restart=True, service_restart=True)

    assert runner._exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.asyncio
async def test_drain_active_agents_throttles_status_updates():
    runner, _adapter = make_restart_runner()
    runner._update_runtime_status = MagicMock()

    runner._running_agents = {"a": MagicMock(), "b": MagicMock()}

    async def finish_agents():
        await asyncio.sleep(0.12)
        runner._running_agents.pop("a")
        await asyncio.sleep(0.12)
        runner._running_agents.clear()

    task = asyncio.create_task(finish_agents())
    await runner._drain_active_agents(1.0)
    await task

    # Start, one count-change update, and final update. Allow one extra update
    # if the loop observes the zero-agent state before exiting.
    assert 3 <= runner._update_runtime_status.call_count <= 4


@pytest.mark.asyncio
async def test_gateway_stop_kills_tool_subprocesses_before_adapter_disconnect_on_timeout(monkeypatch):
    """On drain timeout, tool subprocesses must be killed BEFORE adapter
    disconnect so systemd's TimeoutStopSec doesn't SIGKILL the cgroup with
    bash/sleep children still attached (#8202)."""
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.01  # force timeout path

    call_order: list[str] = []

    def _fake_kill_all(task_id=None):
        call_order.append("kill_all")
        return 2

    def _fake_cleanup_envs():
        call_order.append("cleanup_environments")

    def _fake_cleanup_browsers():
        call_order.append("cleanup_browsers")

    async def _disconnect():
        call_order.append("disconnect")

    # Patch the module-level names the stop() helper imports lazily.
    import tools.process_registry as _pr
    import tools.terminal_tool as _tt
    import tools.browser_tool as _bt
    monkeypatch.setattr(_pr.process_registry, "kill_all", _fake_kill_all)
    monkeypatch.setattr(_tt, "cleanup_all_environments", _fake_cleanup_envs)
    monkeypatch.setattr(_bt, "cleanup_all_browsers", _fake_cleanup_browsers)

    adapter.disconnect = _disconnect

    runner._running_agents = {"session": MagicMock()}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    # First kill_all must precede the first disconnect.  (Both the eager
    # post-interrupt cleanup and the final catch-all call _kill_tool_
    # subprocesses, so we expect kill_all to appear twice total.)
    assert "kill_all" in call_order
    assert "disconnect" in call_order
    first_kill = call_order.index("kill_all")
    first_disconnect = call_order.index("disconnect")
    assert first_kill < first_disconnect, (
        f"Tool subprocesses must be killed before adapter disconnect on "
        f"drain timeout, got order: {call_order}"
    )
    # Defense-in-depth final cleanup still runs.
    assert call_order.count("kill_all") >= 2


@pytest.mark.asyncio
async def test_gateway_stop_kills_tool_subprocesses_on_graceful_path(monkeypatch):
    """Graceful shutdown (no drain timeout) must still kill tool subprocesses
    exactly once via the final catch-all — regression guard against
    accidentally removing that call when refactoring."""
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    kill_count = 0

    def _fake_kill_all(task_id=None):
        nonlocal kill_count
        kill_count += 1
        return 0

    import tools.process_registry as _pr
    import tools.terminal_tool as _tt
    import tools.browser_tool as _bt
    monkeypatch.setattr(_pr.process_registry, "kill_all", _fake_kill_all)
    monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
    monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

    # No running agents → drain returns immediately, no timeout, no eager cleanup.
    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    # Only the final catch-all fires on the graceful path.
    assert kill_count == 1
