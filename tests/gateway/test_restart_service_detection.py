"""Tests for /restart service-manager and container detection.

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

Container detection uses ``hermes_constants.is_container()`` which covers
Docker, Podman, Kubernetes (containerd/CRI-O), and LXC — not just the
``.dockerenv`` / ``.containerenv`` sentinel files.

s6-overlay exports ``HERMES_S6_SUPERVISED_CHILD`` for container longruns.
"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
import hermes_constants
from gateway.platforms.base import MessageEvent, MessageType
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
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
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
async def test_restart_under_s6_uses_service_path(tmp_path, monkeypatch):
    """HERMES_S6_SUPERVISED_CHILD (s6-overlay container) routes via the service path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_inside_kubernetes_pod_uses_service_path(tmp_path, monkeypatch):
    """Kubernetes pod (containerd/CRI-O) routes via the service path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: True)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_inside_docker_uses_service_path(tmp_path, monkeypatch):
    """Docker container (is_container() True) routes via the service path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: True)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_called_once_with(detached=False, via_service=True)
