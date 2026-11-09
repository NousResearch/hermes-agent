"""Real production bootstrap, local socket, owner loop and parser-to-command integration."""
import asyncio
import os
from unittest.mock import Mock

import pytest
from gateway.run import GatewayRunner


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_bootstrap_socket_cli_and_updater_compatibility(tmp_path, monkeypatch, capsys):
    from gateway import control_socket, restart
    from gateway.run_bootstrap import _start_gateway_start_control_socket
    from hermes_cli.gateway_deferred_restart import resolve_target, run_deferred_restart

    monkeypatch.setattr(control_socket, "_detect_supervisor", lambda: "external")
    monkeypatch.setattr(restart, "is_gateway_supervisor_process", lambda: True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent: {restart_drain_timeout: 999}\n")
    runner = object.__new__(GatewayRunner)
    runner.__dict__.update(session_runtime_descriptor={}, session_ticket_store=None,
                             _stop_task=None, _draining=False, _restart_requested=False,
                             _restart_drain_timeout=12.5, request_restart=Mock(return_value=True))
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        await asyncio.to_thread(resolve_target)
        assert await asyncio.to_thread(run_deferred_restart) == 0
        assert f"PID {os.getpid()}" in capsys.readouterr().out
        runner.request_restart.assert_called_once_with(detached=False, via_service=True)
        # A manual replacement after supported preflight must reject on the receiving instance.
        monkeypatch.setattr(control_socket, "_detect_supervisor", lambda: "manual")
        runner.request_restart.reset_mock()
        result = await asyncio.to_thread(control_socket.query_gateway_control, tmp_path, "defer-restart", timeout=8)
        assert result["reason"] == "unsupported_supervisor"
        runner.request_restart.assert_not_called()
        # The updater still owns relaunching a manual gateway, so its old verb remains usable.
        result = await asyncio.to_thread(control_socket.pause_gateway_for_update, tmp_path, timeout=8)
        assert result["pausing"] is True
        runner.request_restart.assert_called_once_with(detached=False, via_service=True)
    finally:
        await server.stop()


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_older_gateway_unknown_verb_is_unknown_without_fallback(tmp_path, monkeypatch, capsys):
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_deferred_restart as cli

    pauses = Mock()
    server = GatewayControlServer(home=tmp_path, verb_handlers={"pause-for-update": pauses})
    monkeypatch.setattr(cli, "resolve_target", lambda: (tmp_path, {"supervisor": "external"}, False))
    assert await server.start()
    try:
        assert await asyncio.to_thread(cli.run_deferred_restart) == 1
        output = capsys.readouterr().out
        assert "UNKNOWN" in output and "may already be processing" in output and "status" in output
        pauses.assert_not_called()
    finally:
        await server.stop()
