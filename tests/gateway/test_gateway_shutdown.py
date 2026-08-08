import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import MessageEvent
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def test_systemd_planned_restart_helper_bounds_stale_main_pid(monkeypatch):
    """A wedged post-drain gateway must not leave the helper waiting forever."""
    runner, _adapter = make_restart_runner()
    runner._restart_drain_timeout = 12.9
    monkeypatch.setenv("INVOCATION_ID", "systemd-test")
    monkeypatch.setattr(gateway_run.sys, "platform", "linux")
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 4321)

    def fake_which(name):
        return f"/usr/bin/{name}"

    def fake_run(argv, **_kwargs):
        # The gateway belongs to the user unit, not the system unit.
        stdout = "4321\n" if "--user" in argv else "0\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    popen = MagicMock()
    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.Popen", popen)

    runner._launch_systemd_restart_shortcut()

    argv = popen.call_args.args[0]
    assert argv[:3] == ["/usr/bin/systemd-run", "--user", "--collect"]
    shell_cmd = argv[-1]
    assert "deadline=$(( $(date +%s) + 17 ))" in shell_cmd
    assert "kill -0 4321" in shell_cmd
    assert "[ $(date +%s) -lt $deadline ]" in shell_cmd
    assert (
        'main_pid=$(/usr/bin/systemctl --user show hermes-gateway '
        '--property=MainPID --value 2>/dev/null)' in shell_cmd
    )
    assert 'if [ "$main_pid" = "4321" ]' in shell_cmd
    assert (
        "/usr/bin/systemctl --user kill --kill-whom=main --signal=SIGKILL "
        "hermes-gateway" in shell_cmd
    )
    assert shell_cmd.index("--signal=SIGKILL") < shell_cmd.index("reset-failed")
    assert shell_cmd.index("reset-failed") < shell_cmd.index("restart hermes-gateway")


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
    # Simulate the agent exiting once interrupted so stop()'s 5s
    # interrupt-deadline poll loop returns immediately.
    running_agent.interrupt.side_effect = lambda *a, **k: runner._running_agents.clear()

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
async def test_in_chat_restart_skips_home_shutdown_even_with_active_session():
    runner, adapter = make_restart_runner()
    source = make_restart_source(thread_id="42")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)
    restart_source = make_restart_source(thread_id="42")
    restart_source.message_id = "restart-command"
    runner._restart_requested = True
    runner._restart_command_source = restart_source
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent_calls) == 1
    chat_id, message, metadata = adapter.sent_calls[0]
    assert chat_id == source.chat_id
    assert "Gateway restarting" in message
    assert metadata["telegram_reply_to_message_id"] == "restart-command"


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
    runner._running_agents["session"].interrupt.side_effect = (
        lambda *a, **k: runner._running_agents.clear()
    )

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


# ---------------------------------------------------------------------------
# gateway_state persistence on shutdown (issue #42675)
#
# On Docker/s6, container_boot.py only auto-starts gateways whose last
# persisted gateway_state was "running". An unexpected external signal
# (the SIGTERM s6/Docker sends on `docker compose up --force-recreate`,
# OOM, bare kill) must NOT persist "stopped" — otherwise the gateway
# stays down after every container restart. An operator-initiated stop
# writes a planned-stop marker first, so it is NOT signal-initiated and
# DOES persist "stopped", respecting the explicit intent.
# ---------------------------------------------------------------------------


def _persisted_states(runner) -> list:
    """All gateway_state values passed to _update_runtime_status, in order."""
    states = []
    for call in runner._update_runtime_status.call_args_list:
        args, kwargs = call
        state = kwargs.get("gateway_state", args[0] if args else None)
        states.append(state)
    return states


def _stopped_state_persisted(runner) -> bool:
    """True iff _update_runtime_status was called with gateway_state='stopped'."""
    return "stopped" in _persisted_states(runner)


@pytest.mark.asyncio
async def test_signal_initiated_shutdown_persists_running_not_stopped(tmp_path, monkeypatch):
    """Unexpected SIGTERM (container restart / OOM / kill) must persist
    gateway_state=running — NOT stopped, and NOT leave the mid-shutdown
    'draining' marker — so container_boot auto-starts on next boot (#42675)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._signal_initiated_shutdown = True  # set by handler on unmarked signal

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert not _stopped_state_persisted(runner), (
        "signal-initiated shutdown must NOT persist gateway_state=stopped"
    )
    # The FINAL terminal write must be 'running' so container_boot's
    # _AUTOSTART_STATES check passes (it only auto-starts 'running').
    assert _persisted_states(runner)[-1] == "running", (
        f"final state must be 'running', got: {_persisted_states(runner)}"
    )


# ── #42126: zombie PID must be treated as dead in _pid_exists ────────────────
# Under systemd Restart=always, the old gateway becomes a zombie (still in the
# process table, not yet reaped) when the replacement starts. _pid_exists must
# report it dead so --replace proceeds instead of waiting on it and aborting
# with exit 1 (a silent crash loop).


def test_pid_exists_zombie_via_psutil_returns_false(monkeypatch):
    """The live path is psutil. psutil.pid_exists() returns True for a zombie,
    so _pid_exists must additionally check Process.status() == STATUS_ZOMBIE."""
    import sys
    import types

    from gateway import status

    fake_psutil = types.SimpleNamespace()
    fake_psutil.STATUS_ZOMBIE = "zombie"

    class NoSuchProcess(Exception):
        pass

    class PsutilError(Exception):
        pass

    fake_psutil.NoSuchProcess = NoSuchProcess
    fake_psutil.Error = PsutilError

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def status(self):
            return "zombie"

    fake_psutil.Process = _Proc
    # Without the zombie guard, this True would make the caller treat the
    # zombie as a live gateway.
    fake_psutil.pid_exists = lambda pid: True

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert status._pid_exists(4242) is False

