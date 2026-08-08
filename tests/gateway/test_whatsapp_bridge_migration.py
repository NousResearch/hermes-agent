"""Upgrade and unsupported-platform boundaries for the WhatsApp bridge."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import (
    WhatsAppAdapter,
    _bridge_client_session,
    _bridge_ipc_endpoint,
    _file_content_hash,
    _kill_stale_bridge_by_pidfile,
    _listener_pids_on_port,
    _migrate_legacy_tcp_bridge,
    _standalone_send,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_listener(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"listener on port {port} did not start")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX process migration integration test")
async def test_no_pid_legacy_tcp_bridge_is_verified_and_retired(tmp_path: Path) -> None:
    """A command-line and health-verified pre-IPC bridge is safely migrated."""
    node_binary = shutil.which("node")
    if node_binary is None:
        pytest.skip("Node.js is required for the legacy bridge migration test")

    port = _free_port()
    session_path = tmp_path / "session"
    session_path.mkdir()
    bridge_script = tmp_path / "legacy_bridge.mjs"
    capture_path = tmp_path / "legacy-request.json"
    bridge_script.write_text(
        textwrap.dedent(
            """
            import http from 'node:http';
            import { writeFileSync } from 'node:fs';
            const args = process.argv.slice(2);
            const value = (name) => args[args.indexOf(name) + 1];
            const port = Number(value('--port'));
            const server = http.createServer((req, res) => {
              if (req.url !== '/health') { res.writeHead(404).end(); return; }
              writeFileSync(value('--capture'), JSON.stringify({
                authorization: req.headers.authorization || null,
                contentLength: req.headers['content-length'] || null,
                method: req.method,
                url: req.url,
              }));
              res.setHeader('content-type', 'application/json');
              res.end(JSON.stringify({
                status: 'connected', queueLength: 0, uptime: process.uptime(),
                scriptHash: '0123456789abcdef', sendReadReceipts: false,
              }));
            });
            server.listen(port, '127.0.0.1');
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [
            node_binary,
            str(bridge_script),
            "--port",
            str(port),
            "--session",
            str(session_path),
            "--capture",
            str(capture_path),
        ]
    )
    try:
        _wait_for_listener(port)
        assert not (session_path / "bridge.pid").exists()

        with patch("aiohttp.ClientSession") as client_session:
            migrated = await _migrate_legacy_tcp_bridge(
                port, bridge_script, session_path
            )

        assert migrated is True
        client_session.assert_not_called()
        assert proc.wait(timeout=5) is not None
        assert capture_path.read_text(encoding="utf-8") == (
            '{"authorization":null,"contentLength":null,'
            '"method":"GET","url":"/health"}'
        )
    finally:
        if proc.poll() is None:
            _stop(proc)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX process migration integration test")
async def test_unproven_legacy_port_owner_blocks_without_termination(tmp_path: Path) -> None:
    """An arbitrary listener is never killed to make room for the IPC bridge."""
    port = _free_port()
    session_path = tmp_path / "session"
    session_path.mkdir()
    bridge_script = tmp_path / "bridge.js"
    bridge_script.write_text("// configured bridge\n", encoding="utf-8")
    server_script = tmp_path / "unrelated.py"
    server_script.write_text(
        "import http.server,sys; "
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), "
        "http.server.BaseHTTPRequestHandler).serve_forever()\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(server_script), str(port)])
    try:
        _wait_for_listener(port)
        with pytest.raises(RuntimeError, match="legacy TCP listener"):
            await _migrate_legacy_tcp_bridge(port, bridge_script, session_path)
        assert proc.poll() is None
    finally:
        _stop(proc)


def test_windows_client_gate_sends_no_protected_bytes(tmp_path: Path) -> None:
    """Unsupported native Windows fails before connector/session construction."""
    fake_aiohttp = MagicMock()
    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True):
        with pytest.raises(RuntimeError, match="native Windows"):
            _bridge_ipc_endpoint(tmp_path, 3000, "secret-token")
        with pytest.raises(RuntimeError, match="native Windows"):
            _bridge_client_session(fake_aiohttp, tmp_path, 3000, "secret-token")

    fake_aiohttp.NamedPipeConnector.assert_not_called()
    fake_aiohttp.ClientSession.assert_not_called()


def test_windows_listener_enumeration_spawns_no_child() -> None:
    """Native retirement uses psutil directly instead of netstat/task shells."""
    connections = [
        SimpleNamespace(
            status="LISTEN",
            laddr=SimpleNamespace(port=30131),
            pid=321,
        ),
        SimpleNamespace(
            status="ESTABLISHED",
            laddr=SimpleNamespace(port=30131),
            pid=654,
        ),
    ]
    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True), patch(
        "psutil.net_connections", return_value=connections
    ), patch("subprocess.run") as run:
        pids = _listener_pids_on_port(30131)

    assert pids == [321]
    run.assert_not_called()


def test_windows_standalone_send_fails_before_http(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / ".bridge-token").write_text("A" * 43, encoding="utf-8")
    config = SimpleNamespace(
        token="",
        extra={"bridge_port": 3000, "session_path": str(session_path)},
    )

    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True), patch(
        "aiohttp.ClientSession"
    ) as client_session:
        result = asyncio.run(_standalone_send(config, "15551234567", "secret"))

    assert "native Windows" in result["error"]
    client_session.assert_not_called()


@pytest.mark.asyncio
async def test_windows_gateway_connect_fails_before_process_start(tmp_path: Path) -> None:
    bridge_script = tmp_path / "bridge.js"
    session_path = tmp_path / "session"
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "bridge_script": str(bridge_script),
        "bridge_port": 30127,
        "session_path": str(session_path),
    }))

    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True), patch(
        "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements"
    ) as requirements, patch.object(
        adapter, "_acquire_platform_lock", return_value=True
    ) as acquire_lock, patch.object(
        adapter, "_release_platform_lock"
    ) as release_lock, patch(
        "plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"
    ) as pidfile_cleanup, patch(
        "plugins.platforms.whatsapp.adapter._migrate_legacy_tcp_bridge",
        new_callable=AsyncMock,
        return_value=True,
    ) as migrate_legacy, patch(
        "plugins.platforms.whatsapp.adapter._read_bridge_token"
    ) as read_token, patch(
        "plugins.platforms.whatsapp.adapter._rotate_bridge_token"
    ) as rotate_token, patch(
        "plugins.platforms.whatsapp.adapter._bridge_client_session"
    ) as client_session, patch("subprocess.run") as run, patch(
        "subprocess.Popen"
    ) as popen:
        connected = await adapter.connect()

    assert connected is False
    assert adapter.fatal_error_code == "whatsapp_windows_ipc_unsupported"
    acquire_lock.assert_called_once_with(
        "whatsapp-session", str(session_path), "WhatsApp session"
    )
    pidfile_cleanup.assert_called_once_with(session_path, bridge_script, 30127)
    migrate_legacy.assert_awaited_once_with(30127, bridge_script, session_path)
    release_lock.assert_called_once_with()
    requirements.assert_not_called()
    read_token.assert_not_called()
    rotate_token.assert_not_called()
    client_session.assert_not_called()
    run.assert_not_called()
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_windows_gateway_unproven_legacy_listener_blocks_unsupported(
    tmp_path: Path,
) -> None:
    """An ambiguous listener stays alive and produces migration guidance."""
    bridge_script = tmp_path / "bridge.js"
    session_path = tmp_path / "session"
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "bridge_script": str(bridge_script),
        "bridge_port": 30128,
        "session_path": str(session_path),
    }))

    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True), patch.object(
        adapter, "_acquire_platform_lock", return_value=True
    ), patch.object(adapter, "_release_platform_lock") as release_lock, patch(
        "plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"
    ), patch(
        "plugins.platforms.whatsapp.adapter._migrate_legacy_tcp_bridge",
        new_callable=AsyncMock,
        side_effect=RuntimeError("unproven legacy TCP listener; stop it manually"),
    ), patch(
        "plugins.platforms.whatsapp.adapter._read_bridge_token"
    ) as read_token, patch(
        "plugins.platforms.whatsapp.adapter._rotate_bridge_token"
    ) as rotate_token, patch(
        "plugins.platforms.whatsapp.adapter._bridge_client_session"
    ) as client_session, patch("subprocess.Popen") as popen:
        connected = await adapter.connect()

    assert connected is False
    assert adapter.fatal_error_code == "whatsapp_legacy_bridge_migration_required"
    assert "stop it manually" in adapter.fatal_error_message
    release_lock.assert_called_once_with()
    read_token.assert_not_called()
    rotate_token.assert_not_called()
    client_session.assert_not_called()
    popen.assert_not_called()


@pytest.mark.asyncio
async def test_windows_verified_migration_rebinds_before_termination(tmp_path: Path) -> None:
    """Windows retirement signals only a stable exact legacy bridge owner."""
    bridge_script = tmp_path / "bridge.js"
    session_path = tmp_path / "session"
    events = []

    def identity(*args):
        events.append(("identity", args))
        return 123456

    async def health(port):
        events.append(("health", port))
        return True

    def terminate(pid, *, force=False):
        events.append(("terminate", pid, force))

    with patch("plugins.platforms.whatsapp.adapter._IS_WINDOWS", True), patch(
        "plugins.platforms.whatsapp.adapter._tcp_listener_is_occupied",
        side_effect=[True, False],
    ), patch(
        "plugins.platforms.whatsapp.adapter._listener_pids_on_port",
        return_value=[4123],
    ), patch(
        "plugins.platforms.whatsapp.adapter._legacy_bridge_process_identity",
        side_effect=identity,
    ), patch(
        "plugins.platforms.whatsapp.adapter._legacy_bridge_health_is_valid",
        side_effect=health,
    ), patch("gateway.status.terminate_pid", side_effect=terminate):
        migrated = await _migrate_legacy_tcp_bridge(
            30129, bridge_script, session_path
        )

    assert migrated is True
    assert events == [
        ("identity", (4123, bridge_script, session_path, 30129)),
        ("health", 30129),
        ("identity", (4123, bridge_script, session_path, 30129)),
        ("terminate", 4123, False),
    ]


@pytest.mark.asyncio
async def test_unproven_legacy_listener_blocks_replacement_spawn(tmp_path: Path) -> None:
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    bridge_script = bridge_dir / "bridge.js"
    bridge_script.write_text("// current bridge\n", encoding="utf-8")
    package_json = bridge_dir / "package.json"
    package_json.write_text('{"name":"bridge"}\n', encoding="utf-8")
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / ".hermes-pkg-hash").write_text(
        _file_content_hash(package_json), encoding="utf-8"
    )
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "creds.json").write_text("{}", encoding="utf-8")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "bridge_script": str(bridge_script),
        "session_path": str(session_path),
        "bridge_port": 30126,
    }))

    with patch(
        "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
        return_value=True,
    ), patch.object(
        adapter, "_acquire_platform_lock", return_value=True
    ), patch(
        "plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile",
        return_value=False,
    ), patch(
        "plugins.platforms.whatsapp.adapter._migrate_legacy_tcp_bridge",
        side_effect=RuntimeError("unproven legacy TCP listener"),
    ), patch("subprocess.Popen") as popen:
        connected = await adapter.connect()

    assert connected is False
    assert adapter.fatal_error_code == "whatsapp_legacy_bridge_migration_required"
    popen.assert_not_called()


def test_windows_legacy_pidfile_requires_exact_configured_identity(
    tmp_path: Path,
) -> None:
    """A loose node/session match cannot authorize Windows upgrade cleanup."""
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "bridge.pid").write_text("4123", encoding="utf-8")
    bridge_script = tmp_path / "bridge.js"

    with patch(
        "plugins.platforms.whatsapp.adapter._legacy_bridge_process_identity",
        return_value=None,
    ) as exact_identity, patch(
        "plugins.platforms.whatsapp.adapter._bridge_pid_is_ours",
        return_value=True,
    ) as loose_identity, patch(
        "gateway.status._pid_exists", return_value=True
    ), patch("plugins.platforms.whatsapp.adapter.os.kill") as kill:
        terminated = _kill_stale_bridge_by_pidfile(
            session_path, bridge_script, 30130
        )

    assert terminated is False
    exact_identity.assert_called_once_with(4123, bridge_script, session_path, 30130)
    loose_identity.assert_not_called()
    kill.assert_not_called()
