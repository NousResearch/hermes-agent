"""Tests for /restart service-manager detection (launchd vs interactive).

The /restart handler routes through ``request_restart(via_service=True)``
when a service manager supervises the gateway, so the process exits with
the service-restart code and the manager relaunches it.  Under macOS
launchd the plist uses ``KeepAlive.SuccessfulExit=false`` — a clean exit 0
is treated as a deliberate stop and the gateway stays dead (#43475) — so
launchd must be detected here in the handler, not only at the exit-code
site (which never runs unless ``via_service=True`` is already set).

launchd sets ``XPC_SERVICE_NAME`` to the job label for processes it
spawns.  Interactive macOS shells inherit ``XPC_SERVICE_NAME=0`` (a
truthy string), so the probe must treat ``"0"`` as not-under-launchd:
routing an unsupervised interactive gateway to the service path would
make it exit non-zero with nothing to revive it.
"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent, MessageType
from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _make_restart_event(update_id: int | None = 100) -> MessageEvent:
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
        platform_update_id=update_id,
    )


def _make_runner_with_mock_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    return runner


@pytest.mark.asyncio
async def test_restart_under_launchd_uses_service_path(tmp_path, monkeypatch):
    """launchd job label in XPC_SERVICE_NAME routes /restart via the service path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_in_interactive_macos_shell_uses_detached_path(tmp_path, monkeypatch):
    """XPC_SERVICE_NAME=0 (inherited by interactive macOS shells) is NOT a service."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("XPC_SERVICE_NAME", "0")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
async def test_restart_without_service_env_uses_detached_path(tmp_path, monkeypatch):
    """No service-manager env at all falls back to the detached restart."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
async def test_restart_under_systemd_uses_service_path(tmp_path, monkeypatch):
    """INVOCATION_ID (systemd) still routes via the service path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("INVOCATION_ID", "abc123")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_with_external_supervisor_marker_uses_service_path(
    tmp_path, monkeypatch
):
    """Wrapped supervisors can retain restart ownership without native markers."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, "1")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "0", "false", "off"])
async def test_false_external_supervisor_marker_keeps_detached_path(
    value, tmp_path, monkeypatch
):
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, value)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=True, via_service=False)
