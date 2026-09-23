"""Required pre-update control-socket probes distinguish absence from observation failure."""

from pathlib import Path

import pytest

import gateway.control_socket as control_socket


def test_required_query_rejects_malformed_response(tmp_path, monkeypatch):
    monkeypatch.setattr(control_socket, "_IS_WINDOWS", False)
    monkeypatch.setattr(control_socket, "_query_unix_socket",
                        lambda *_args, **_kwargs: b"not json")
    assert control_socket.identify_gateway(tmp_path) is None
    with pytest.raises(ValueError):
        control_socket.identify_gateway(tmp_path, require_complete=True)


def test_required_unix_socket_connect_failure_is_not_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(control_socket, "resolve_client_socket_path",
                        lambda _home, **_kwargs: tmp_path / "gateway.sock")

    class FailingSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise OSError("socket connect failed")

    monkeypatch.setattr(control_socket.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(control_socket.socket, "socket", lambda *_args: FailingSocket())
    assert control_socket._query_unix_socket(tmp_path, b"{}", 0.1) is None
    with pytest.raises(OSError, match="socket connect failed"):
        control_socket._query_unix_socket(tmp_path, b"{}", 0.1, require_complete=True)


def test_required_windows_pipe_busy_failure_is_not_absence(tmp_path, monkeypatch):
    def busy_pipe(*_args, **_kwargs):
        raise OSError("pipe busy")

    monkeypatch.setattr("builtins.open", busy_pipe)
    assert control_socket._query_windows_pipe(tmp_path, b"{}", 0) is None
    with pytest.raises(OSError, match="pipe busy"):
        control_socket._query_windows_pipe(tmp_path, b"{}", 0, require_complete=True)


def test_required_missing_socket_and_pipe_are_verified_absence(tmp_path, monkeypatch):
    assert control_socket.resolve_client_socket_path(tmp_path, require_complete=True) is None

    def missing_pipe(*_args, **_kwargs):
        raise FileNotFoundError("no pipe")

    monkeypatch.setattr("builtins.open", missing_pipe)
    assert control_socket._query_windows_pipe(tmp_path, b"{}", 0, require_complete=True) is None


def test_required_socket_path_lookup_rejects_unreadable_pointer(tmp_path, monkeypatch):
    pointer = tmp_path / "gateway.sock.path"
    original_stat = Path.stat

    def unreadable_pointer(path, *args, **kwargs):
        if path == pointer:
            raise PermissionError("pointer unavailable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unreadable_pointer)
    assert control_socket.resolve_client_socket_path(tmp_path) is None
    with pytest.raises(PermissionError, match="pointer unavailable"):
        control_socket.resolve_client_socket_path(tmp_path, require_complete=True)
