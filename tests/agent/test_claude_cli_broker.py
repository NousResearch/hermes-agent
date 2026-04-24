import os
import socket

from agent.claude_cli_broker import _acquire_server_lock, _session_key, _socket_is_alive


def test_no_state_requests_do_not_share_first_turn_worker():
    config = {"command": "claude", "args": [], "cwd": "/tmp", "strip_runtime": True}

    assert _session_key({"request_id": "a", "config": config}) != _session_key(
        {"request_id": "b", "config": config}
    )


def test_stateful_requests_reuse_session_worker_key():
    config = {"command": "claude", "args": [], "cwd": "/tmp", "strip_runtime": True}
    state = {"claude_session_id": "sess-123"}

    assert _session_key({"request_id": "a", "config": config, "transport_state": state}) == _session_key(
        {"request_id": "b", "config": config, "transport_state": state}
    )


def test_server_lock_allows_only_one_owner(tmp_path):
    socket_path = tmp_path / "broker.sock"

    first_fd = _acquire_server_lock(socket_path)
    assert first_fd is not None
    try:
        assert _acquire_server_lock(socket_path) is None
    finally:
        os.close(first_fd)

    second_fd = _acquire_server_lock(socket_path)
    assert second_fd is not None
    os.close(second_fd)


def test_socket_is_alive_detects_live_and_stale_sockets(tmp_path):
    socket_path = tmp_path / "broker.sock"
    socket_path.touch()
    assert _socket_is_alive(socket_path) is False
    socket_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        assert _socket_is_alive(socket_path) is True
    finally:
        server.close()
