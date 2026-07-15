"""Security and real-path tests for the remote_run plugin."""

from __future__ import annotations

import json
import shlex
import socket
import threading
from unittest.mock import patch

import paramiko
from paramiko.common import (
    AUTH_SUCCESSFUL,
    OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED,
    OPEN_SUCCEEDED,
)

from plugins.remote_run import (
    MAX_RESULT_SIZE_CHARS,
    _build_command,
    register,
    remote_run_handler,
)


class _SSHServer(paramiko.ServerInterface):
    def __init__(self) -> None:
        self.exec_requested = threading.Event()

    def check_auth_password(self, username: str, password: str) -> int:
        return AUTH_SUCCESSFUL

    def get_allowed_auths(self, username: str) -> str:
        return "password"

    def check_channel_request(self, kind: str, chanid: int) -> int:
        return OPEN_SUCCEEDED if kind == "session" else OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_exec_request(self, channel: paramiko.Channel, command: bytes) -> bool:
        self.exec_requested.set()
        return True


def _start_ssh_server(host_key: paramiko.PKey, payload: bytes, port: int = 0) -> tuple[int, threading.Thread]:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    actual_port = listener.getsockname()[1]

    def serve() -> None:
        transport = None
        try:
            client, _ = listener.accept()
            transport = paramiko.Transport(client)
            transport.add_server_key(host_key)
            server = _SSHServer()
            transport.start_server(server=server)
            channel = transport.accept(5)
            if channel is None or not server.exec_requested.wait(5):
                return
            channel.sendall(payload)
            channel.send_exit_status(0)
            channel.close()
        except (EOFError, OSError, paramiko.SSHException):
            pass
        finally:
            if transport is not None:
                transport.close()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return actual_port, thread


def test_registers_as_plugin_with_fixed_result_budget() -> None:
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    registry.deregister("remote_run")
    manager = PluginManager()
    register(PluginContext(PluginManifest(name="remote-run"), manager))
    try:
        entry = registry.get_entry("remote_run")
        assert entry is not None
        assert entry.toolset == "ssh"
        assert entry.max_result_size_chars == MAX_RESULT_SIZE_CHARS
        assert "remote_run" in manager._plugin_tool_names
    finally:
        registry.deregister("remote_run")


def test_command_builder_quotes_inputs_and_rejects_env_key_injection() -> None:
    inner = _build_command("printf ok", "/tmp/a b", {"VALUE": "it's safe"}, False)
    command = _build_command("printf ok", "/tmp/a b", {"VALUE": "it's safe"}, True)
    assert "cd '/tmp/a b'" in inner
    assert "it'\"'\"'s safe" in inner
    assert shlex.split(command)[-1] == inner

    try:
        _build_command("true", None, {"X; whoami": "bad"}, False)
    except ValueError as exc:
        assert "Invalid environment variable key" in str(exc)
    else:
        raise AssertionError("invalid environment key was accepted")


def test_dangerous_command_is_blocked_before_connect() -> None:
    with patch("paramiko.SSHClient") as ssh_client:
        result = json.loads(remote_run_handler({"host": "example.test", "command": "rm -rf /"}))
    assert result["status"] == "blocked"
    ssh_client.assert_not_called()


def test_real_ssh_path_persists_host_key_rejects_change_and_bounds_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first_key = paramiko.RSAKey.generate(1024)
    payload = b"x" * (MAX_RESULT_SIZE_CHARS + 20_000)
    port, first_server = _start_ssh_server(first_key, payload)

    first = json.loads(
        remote_run_handler(
            {
                "host": "127.0.0.1",
                "port": port,
                "user": "tester",
                "password": "secret",
                "command": "printf payload",
                "timeout": 10,
            }
        )
    )
    first_server.join(5)

    assert first["exit_code"] == 0
    assert len(first["stdout"].encode()) == MAX_RESULT_SIZE_CHARS
    assert "output truncated" in first["stderr"]
    known_hosts = tmp_path / "ssh" / "known_hosts"
    assert known_hosts.exists()
    assert f"[127.0.0.1]:{port}" in known_hosts.read_text()

    changed_key = paramiko.RSAKey.generate(1024)
    _, second_server = _start_ssh_server(changed_key, b"should not run", port=port)
    second = json.loads(
        remote_run_handler(
            {
                "host": "127.0.0.1",
                "port": port,
                "user": "tester",
                "password": "secret",
                "command": "printf payload",
                "timeout": 10,
            }
        )
    )
    second_server.join(5)

    assert second["exit_code"] == 1
    assert "Host key verification failed" in second["error"]
