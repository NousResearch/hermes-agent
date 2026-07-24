from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock, patch

from tools.computer_use.transports.http_mcp import HttpMcpTransport
from tools.computer_use.transports.stdio import StdioMcpTransport
from tools.environments.capability_adapter import resolve_tools
from tools.environments.capability_manifest import CapabilityManifest, desktop_manifest
from tools.environments.compute_provider import ComputeLease, EnvironmentCapabilities
from tools.environments.cua_fleet import CuaFleetConfig
from tools.environments.desktop_lease import DesktopSandboxManager
from tools.environments.modal_desktop import ModalDesktopConfig


def test_compute_lease_fields() -> None:
    capabilities = EnvironmentCapabilities(computer_use=True)
    lease = ComputeLease("task-1", "lease-1", "modal", "desktop:latest", capabilities)

    assert lease.task_id == "task-1"
    assert lease.lease_id == "lease-1"
    assert lease.provider == "modal"
    assert lease.capabilities.computer_use


def test_environment_capabilities_to_capabilities() -> None:
    capabilities = EnvironmentCapabilities(computer_use=True, extras=frozenset({"browser"}))

    assert capabilities.to_capabilities() == {"terminal", "files", "process", "computer_use", "browser"}


def test_manifest_parses_dict_and_json() -> None:
    manifest = CapabilityManifest.from_dict({
        "image": "desktop:latest",
        "capabilities": {"terminal": True, "computer_use": {"service": "cua-driver"}},
    })
    json_manifest = CapabilityManifest.from_json(json.dumps({"capabilities": ["files"]}))

    assert manifest.image == "desktop:latest"
    assert manifest.capabilities["computer_use"].service == "cua-driver"
    assert json_manifest.enabled_capabilities() == {"files"}


def test_desktop_manifest_defaults() -> None:
    manifest = desktop_manifest("desktop:latest")

    assert manifest.image == "desktop:latest"
    assert {"terminal", "files", "process", "computer_use"} <= manifest.enabled_capabilities()


def test_capability_adapter_resolves_authorized_intersection() -> None:
    tools = resolve_tools(
        {"terminal", "files", "computer_use"},
        {"terminal", "computer_use"},
    )

    assert tools == {"terminal", "computer_use"}


def test_desktop_sandbox_manager_acquire_release() -> None:
    provider = Mock()
    lease = ComputeLease("task-1", "lease-1", "fake", "desktop", EnvironmentCapabilities(computer_use=True))
    environment = Mock()
    provider.acquire.return_value = lease
    provider.create_environment.return_value = environment
    manager = DesktopSandboxManager(provider)

    first = manager.acquire("task-1")
    second = manager.acquire("task-1")
    manager.release("task-1")
    manager.release("task-1")

    assert first is second
    assert first.references == 0
    provider.acquire.assert_called_once()
    environment.cleanup.assert_called_once()
    provider.release.assert_called_once_with(lease)


def test_stdio_transport_start_stop() -> None:
    process = Mock()
    process.poll.return_value = None
    with patch("tools.computer_use.transports.stdio.subprocess.Popen", return_value=process) as popen:
        transport = StdioMcpTransport(("cua-driver", "mcp"))
        transport.start()
        assert transport.is_alive()
        transport.stop()

    popen.assert_called_once()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
    assert not transport.is_alive()


def test_http_transport_start_stop_and_alive() -> None:
    response = Mock()
    response.read.return_value = b'{"result": {}}'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    transport = HttpMcpTransport("https://cua.example/mcp")

    assert not transport.is_alive()
    transport.start()
    with patch("tools.computer_use.transports.http_mcp.urlopen", return_value=response):
        assert transport.is_alive()
    transport.stop()
    assert not transport.is_alive()


def test_modal_desktop_config_defaults() -> None:
    config = ModalDesktopConfig()

    assert config.image == "trycua/cua:latest"
    assert config.cua_driver_command == ("cua-driver", "mcp")
    assert config.persistent_filesystem


def test_cua_fleet_config_defaults() -> None:
    config = CuaFleetConfig()

    assert config.pool == "default"
    assert config.endpoint == ""
