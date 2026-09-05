"""Unit tests for the host_bridge_cli launcher guards.

Pure unit: no network, no X11, no real cua-driver — the blocking seams
(``_serve_app``, allowlist validation, child-session build) are no-op'd so
each test isolates one fail-closed guard.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

from tools.computer_use import host_bridge_cli

_TOKEN = "a" * 64  # 64 hex-style chars: ASCII, >= 32 bytes


@pytest.fixture(autouse=True)
def _clean_launcher_env(monkeypatch):
    """Guards read os.environ directly; start each test from a quiet slate."""
    for key in (
        host_bridge_cli._CUA_REMOTE_TOKEN_ENV,
        host_bridge_cli._CUA_PERMISSION_MODE_ENV,
        host_bridge_cli._CUA_BYPASS_APPROVALS_ENV,
        host_bridge_cli._BRIDGE_ALLOW_PLAINTEXT_ENV,
    ):
        monkeypatch.delenv(key, raising=False)


def _no_op_run_host_bridge_deps(monkeypatch, serve_calls=None):
    """Neutralize everything around run_host_bridge's guards + serve seam."""
    monkeypatch.setattr(host_bridge_cli, "_ensure_interactive_session", lambda: None)
    monkeypatch.setattr(
        host_bridge_cli, "validate_security_allowlists",
        lambda hosts, origins: (list(hosts), list(origins)),
    )
    monkeypatch.setattr(
        host_bridge_cli, "_build_child_session_context", lambda: object()
    )
    monkeypatch.setattr(host_bridge_cli, "_create_host_bridge_app", lambda **kw: object())
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)

    def fake_serve(app, host, port):
        if serve_calls is not None:
            serve_calls.append((host, port))

    monkeypatch.setattr(host_bridge_cli, "_serve_app", fake_serve)


def test_permission_mode_unrestricted_rejected(monkeypatch):
    monkeypatch.setenv("CUA_DRIVER_PERMISSION_MODE", "unrestricted")
    with pytest.raises(RuntimeError, match="standard permission mode"):
        host_bridge_cli._validate_standard_permission_environment()


def test_bypass_approvals_rejected(monkeypatch):
    monkeypatch.setenv("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", "1")
    with pytest.raises(RuntimeError, match="bypassing approvals"):
        host_bridge_cli._validate_standard_permission_environment()


def test_permission_environment_passes_when_unset():
    # Must not raise: unset mode/bypass is the default safe configuration.
    host_bridge_cli._validate_standard_permission_environment()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="DISPLAY policy is Linux-specific"
)
def test_interactive_session_requires_display_on_linux(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="DISPLAY is not set"):
        host_bridge_cli._ensure_interactive_session()


@pytest.mark.parametrize("bad_port", [0, 65536])
def test_run_host_bridge_rejects_out_of_range_ports(monkeypatch, bad_port):
    # Isolate the port check: every earlier guard becomes a no-op. The token
    # env is popped by run_host_bridge, so it must be set via monkeypatch.
    _no_op_run_host_bridge_deps(monkeypatch)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", _TOKEN)
    with pytest.raises(RuntimeError, match="between 1 and 65535"):
        host_bridge_cli.run_host_bridge(
            allowed_hosts=["localhost"],
            allowed_origins=["http://localhost:8765"],
            port=bad_port,
        )


def test_run_host_bridge_refuses_non_loopback_plaintext(monkeypatch):
    _no_op_run_host_bridge_deps(monkeypatch)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", _TOKEN)
    with pytest.raises(RuntimeError, match="plaintext"):
        host_bridge_cli.run_host_bridge(
            allowed_hosts=["localhost"],
            allowed_origins=["http://localhost:8765"],
            port=8765,
            bind="0.0.0.0",
        )


def test_run_host_bridge_allows_non_loopback_with_plaintext_ack(monkeypatch):
    # Explicit HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT=1 acknowledgement + every
    # heavy dependency no-op'd: run_host_bridge completes and reaches the
    # serve seam without uvicorn ever being imported or started.
    serve_calls: list[tuple[str, int]] = []
    _no_op_run_host_bridge_deps(monkeypatch, serve_calls)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", _TOKEN)
    monkeypatch.setenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", "1")
    host_bridge_cli.run_host_bridge(
        allowed_hosts=["localhost"],
        allowed_origins=["http://localhost:8765"],
        port=8765,
        bind="0.0.0.0",
    )
    assert serve_calls == [("0.0.0.0", 8765)]


def test_bind_security_passes_loopback_without_ack(monkeypatch):
    monkeypatch.delenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", raising=False)
    for bind in ("127.0.0.1", "localhost", "::1", "[::1]"):
        host_bridge_cli._ensure_bind_security(bind)  # must not raise


def test_build_child_session_context_pins_telemetry_off(monkeypatch):
    # Security parity with the standalone launcher: the driver child env must
    # carry the telemetry kill-switch even when the parent opts into telemetry.
    captured: dict[str, dict] = {}

    def fake_context(*, command, args, env):
        captured["env"] = env
        return MagicMock()

    monkeypatch.setattr(host_bridge_cli, "_cua_driver_session_context", fake_context)
    # Patch the defining modules — the facade doesn't re-export these names.
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd",
        lambda: "/usr/bin/cua-driver",
    )
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_driver._resolve_mcp_invocation",
        lambda cmd: (cmd, ["mcp"]),
    )
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_driver.cua_driver_install_hint",
        lambda: "install cua-driver",
    )
    monkeypatch.setattr(
        "tools.computer_use.cua_backend.cua_driver_child_env",
        lambda base_env=None: {
            **(base_env or os.environ),
            "CUA_DRIVER_RS_TELEMETRY_ENABLED": "1",
        },
    )
    monkeypatch.setattr(
        "tools.environments.local._sanitize_subprocess_env", lambda env, extra=None: dict(env)
    )
    monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN", raising=False)
    monkeypatch.delenv("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", raising=False)

    host_bridge_cli._build_child_session_context()
    assert captured["env"]["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"