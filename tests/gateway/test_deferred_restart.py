"""Acceptance handoff: admission, bounded uncertainty, stop ordering, and existing drain."""
import asyncio
import json
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import control_socket, deferred_restart as deferred, restart
from gateway.control_socket import GatewayControlServer
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture
def runner():
    return SimpleNamespace(_restart_drain_timeout=7.5, _stop_task=None, _draining=False,
                           _restart_requested=False, request_restart=Mock(return_value=True))


@pytest.fixture
def supervised(monkeypatch):
    monkeypatch.setattr(control_socket, "_detect_supervisor", lambda: "external")
    monkeypatch.setattr(restart, "is_gateway_supervisor_process", lambda: True)


class QueuedLoop:
    def __init__(self, immediate=False):
        self.callbacks = []
        self.immediate = immediate

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)
        if self.immediate:
            callback()


def response(runner, loop):
    server = GatewayControlServer(verb_handlers={"defer-restart": deferred.deferred_restart_verb(runner, loop)})
    return json.loads(server.handle_request_line(b'{"verb":"defer-restart"}'))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("supervisor,capable", [("manual", False), ("desktop", True), ("s6", True),
    (None, True), ("unknown", True), ("external", False), ("systemd", False), ("launchd", False)])
def test_server_gate_is_independent_of_cli_preflight(runner, monkeypatch, supervisor, capable):
    monkeypatch.setattr(control_socket, "_detect_supervisor", lambda: supervisor)
    monkeypatch.setattr(restart, "is_gateway_supervisor_process", lambda: capable)
    loop = QueuedLoop()
    result = response(runner, loop)
    assert result["result"]["reason"] == "unsupported_supervisor"
    assert result["result"]["accepted"] is False
    assert loop.callbacks == []
    runner.request_restart.assert_not_called()


@pytest.mark.platforms("windows")
def test_native_windows_external_marker_is_refused(runner, monkeypatch):
    monkeypatch.setenv(restart.EXTERNAL_GATEWAY_SUPERVISOR_ENV, "1")
    monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "run", "--external-supervisor"])
    loop = QueuedLoop()
    result = response(runner, loop)
    assert result["result"]["reason"] == "unsupported_supervisor"
    assert loop.callbacks == []
    runner.request_restart.assert_not_called()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("accepted", [True, False])
def test_callback_true_false_and_loaded_budget(runner, supervised, tmp_path, accepted):
    (tmp_path / "config.yaml").write_text("agent: {restart_drain_timeout: 999}\n")
    runner.request_restart.return_value = accepted
    result = response(runner, QueuedLoop(immediate=True))
    assert result["ok"] is True
    assert result["result"] == {"accepted": accepted, "pid": os.getpid(), "supervisor": "external",
                                "drain_timeout": 7.5, **({} if accepted else {"reason": "already_stopping"})}
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("budget", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_loaded_budget_rejected_before_mutation(runner, supervised, budget):
    runner._restart_drain_timeout = budget
    loop = QueuedLoop()
    result = response(runner, loop)
    assert result["result"]["reason"] == "unsupported_drain_timeout"
    assert not loop.callbacks
    runner.request_restart.assert_not_called()


@pytest.mark.platforms("posix")
def test_callback_exception_is_protocol_error(runner, supervised):
    runner.request_restart.side_effect = RuntimeError("callback failed")
    assert response(runner, QueuedLoop(immediate=True))["ok"] is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("stop_state", ["_stop_task", "_stop_requested_by_signal", "_signal_initiated_shutdown", "_draining"])
def test_ordinary_stop_before_callback_preserves_intent(runner, supervised, stop_state):
    setattr(runner, stop_state, True)
    result = response(runner, QueuedLoop(immediate=True))
    assert result["result"]["reason"] == "already_stopping"
    assert runner._restart_requested is False
    runner.request_restart.assert_not_called()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("stop_wins", [False, True])
def test_timeout_is_unknown_without_retry_and_late_callback_runs_once(runner, supervised, monkeypatch, tmp_path, capsys, stop_wins):
    from hermes_cli import gateway_deferred_restart as cli
    monkeypatch.setattr(deferred, "HANDLER_TIMEOUT", 0)
    monkeypatch.setattr(cli, "resolve_target", lambda: (tmp_path, {"supervisor": "external"}, False))
    loop = QueuedLoop()
    query = Mock(side_effect=lambda *a, **kw: response(runner, loop).get("result"))
    monkeypatch.setattr(control_socket, "query_gateway_control", query)
    assert cli.run_deferred_restart() == 1
    assert "UNKNOWN" in capsys.readouterr().out
    query.assert_called_once()
    assert len(loop.callbacks) == 1
    runner._stop_task = object() if stop_wins else None
    loop.callbacks[0]()
    assert runner.request_restart.call_count == (0 if stop_wins else 1)
    assert runner._restart_requested is False


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_active_work_ack_precedes_stop_and_duplicate_creates_one_flow(supervised, monkeypatch):
    runner, _ = make_restart_runner()
    runner._restart_after_turn_timeout = 30
    runner._restart_drain_timeout = 6
    work = True
    observed_wait = asyncio.Event()

    def count():
        observed_wait.set()
        return int(work)

    # Real request_restart and real after-turn wait consume controlled eligible work.
    runner._active_work_count = count
    runner._awaitable_work_count = count
    runner._describe_active_work = list
    runner._scale_to_zero_status = Mock()
    runner.stop = AsyncMock()
    handler = deferred.deferred_restart_verb(runner, asyncio.get_running_loop())
    first = await asyncio.to_thread(handler)
    assert first["accepted"] is True
    await asyncio.wait_for(observed_wait.wait(), 5)
    runner.stop.assert_not_awaited()
    task = runner._restart_task
    second = await asyncio.to_thread(handler)
    assert second["reason"] == "already_stopping"
    assert runner._restart_task is task
    runner.stop.assert_not_awaited()
    work = False
    await asyncio.wait_for(task, 5)
    runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_real_ordinary_stop_suspended_at_async_boundary(supervised, monkeypatch):
    from gateway.run import GatewayRunner
    runner, _ = make_restart_runner()
    entered, release = asyncio.Event(), asyncio.Event()

    async def stopping(self):
        entered.set()
        await release.wait()
        self._exit_code = 0

    monkeypatch.setattr(GatewayRunner, "_stop_impl", stopping)
    stop = asyncio.create_task(runner.stop())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        result = await asyncio.to_thread(deferred.deferred_restart_verb(runner, asyncio.get_running_loop()))
        assert result["reason"] == "already_stopping"
        assert not runner._restart_task_started and not runner._restart_requested
        assert not runner._restart_via_service and runner._restart_task is None
    finally:
        release.set()
        await asyncio.wait_for(stop, 5)
    assert runner._exit_code == 0


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_planned_takeover_rejects_deferred_restart_before_stop_starts(supervised, monkeypatch, tmp_path):
    from gateway import run as run_mod, shutdown_forensics, status
    from gateway.run_bootstrap import _start_gateway_make_shutdown_signal_handler
    from gateway.run_shutdown import _resolve_gateway_exit_verdict

    runner, _ = make_restart_runner()
    runner._restart_drain_timeout = 180.0
    runner._launchd_exit_timeout_s = 60.0
    runner._stop_requested_by_signal = False
    runner._restart_task = None
    runner._exit_with_failure = False
    runner.request_restart = Mock(wraps=runner.request_restart)
    drain = AsyncMock(wraps=run_mod.GatewayRunner._stop_drain_active_work)
    monkeypatch.setattr(run_mod.GatewayRunner, "_stop_drain_active_work", drain)
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(shutdown_forensics, "snapshot_shutdown_context", lambda *_: None)
    status.write_takeover_marker(os.getpid())
    signal_shutdown = [False]
    shutdown = _start_gateway_make_shutdown_signal_handler(runner, signal_shutdown)
    loop = asyncio.get_running_loop()
    stop_tasks_at_request = []

    def before_stop(callback):
        stop_tasks_at_request.append(runner._stop_task)
        callback()

    def enqueue_request(callback):
        # Both callbacks enter the real loop's ready queue before the signal handler
        # can append its stop task: signal -> deferred callback -> stop's first step.
        loop.call_soon(shutdown, signal.SIGTERM)
        loop.call_soon(before_stop, callback)

    callback_loop = SimpleNamespace(
        call_soon_threadsafe=lambda callback: loop.call_soon_threadsafe(enqueue_request, callback)
    )
    try:
        result = await asyncio.to_thread(deferred.deferred_restart_verb(runner, callback_loop))
        await asyncio.wait_for(runner._shutdown_event.wait(), 10)
        await asyncio.wait_for(runner._stop_task, 10)
        assert stop_tasks_at_request == [None]
        assert result == {"accepted": False, "reason": "already_stopping", "pid": os.getpid(),
                          "supervisor": "external", "drain_timeout": 180.0}
        runner.request_restart.assert_not_called()
        assert runner._restart_task is None and runner._restart_task_started is False
        assert (runner._restart_requested, runner._restart_via_service, runner._restart_detached) == (False, False, False)
        assert signal_shutdown == [False]
        assert runner._signal_initiated_shutdown is False and runner._stop_requested_by_signal is False
        assert runner._restart_drain_timeout == restart.effective_stop_drain_timeout(runner) == 180.0
        drain.assert_awaited_once()
        assert drain.await_args.args[1] == 180.0
        assert runner.exit_code is None
        assert _resolve_gateway_exit_verdict(runner, signal_shutdown[0]) is True
        runner._update_runtime_status.assert_called_with("stopped", None)
    finally:
        if runner._restart_task is not None:
            runner._restart_task.cancel()
            await asyncio.gather(runner._restart_task, return_exceptions=True)
