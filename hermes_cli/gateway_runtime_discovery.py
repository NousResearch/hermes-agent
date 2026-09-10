"""Bounded local-control discovery without diagnostic fallback or mutation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
from typing import Literal
import time


class DiscoveryError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _private_node(path: Path, *, kind: str) -> os.stat_result:
    node = path.lstat()
    predicates = {"socket": stat.S_ISSOCK, "file": stat.S_ISREG, "directory": stat.S_ISDIR}
    if not predicates[kind](node.st_mode) or node.st_uid != os.getuid():  # windows-footgun: ok — POSIX socket path only
        raise DiscoveryError("unsafe_control_path")
    if stat.S_IMODE(node.st_mode) & 0o077:
        raise DiscoveryError("unsafe_control_permissions")
    return node


def _socket_path(home: Path) -> Path:
    _private_node(home, kind="directory")
    direct = home / "gateway.sock"
    if os.path.lexists(direct):
        _private_node(direct, kind="socket")
        return direct
    pointer = home / "gateway.sock.path"
    # Reject non-regular metadata before a FIFO can wait for a writer.
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK  # windows-footgun: ok — POSIX socket discovery only
    with os.fdopen(os.open(pointer, flags), "rb") as stream:  # windows-footgun: ok — binary POSIX descriptor
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()  # windows-footgun: ok — POSIX pointer only
                or stat.S_IMODE(metadata.st_mode) & 0o077):
            raise DiscoveryError("unsafe_control_pointer")
        data = stream.read(4097)
    if len(data) > 4096:
        raise DiscoveryError("invalid_control_pointer")
    target = Path(data.decode("utf-8").strip())
    if not target.is_absolute():
        raise DiscoveryError("invalid_control_pointer")
    from gateway.control_socket import _home_hash
    # The owner may have a different TMPDIR (notably launchd/native apps).
    # Authenticate its private pointer and profile-specific directory, not our
    # process-local temporary-root preference.
    if target.name != "control.sock" or target.parent.name != f"hermes-gw-{_home_hash(home)}":
        raise DiscoveryError("invalid_control_pointer")
    directory = _private_node(target.parent, kind="directory")
    if directory.st_mode & 0o077:
        raise DiscoveryError("unsafe_control_permissions")
    _private_node(target, kind="socket")
    return target


def query_identify(home: Path, *, timeout: float) -> dict:
    """Unlike diagnostic queries, retain timeout/access/protocol failures."""
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        return _identify_response(query_runtime_control(
            home, b'{"protocol":1,"verb":"identify","id":1}\n', timeout))
    deadline = time.monotonic() + timeout
    path = _socket_path(home)
    request = b'{"protocol":1,"verb":"identify","id":1}\n'
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError
        client.settimeout(budget)
        client.connect(str(path))
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError
        client.settimeout(budget)
        client.sendall(request)
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            client.settimeout(remaining)
            chunk = client.recv(min(65536, 524289 - len(data)))
            if not chunk:
                raise DiscoveryError("incomplete_control_response")
            data.extend(chunk)
            if len(data) > 524288:
                raise DiscoveryError("oversized_control_response")
    return _identify_response(bytes(data))


def _identify_response(data: bytes) -> dict:
    response = json.loads(data.split(b"\n", 1)[0])
    if (not isinstance(response, dict) or response.get("ok") is not True
            or response.get("protocol") != 1 or response.get("id") != 1
            or not isinstance(response.get("result"), dict)):
        raise DiscoveryError("invalid_control_response")
    return response["result"]


def missing_owner_state(home: Path) -> Literal["starting", "absent", "inaccessible"]:
    """A reservation before PID/control publication already excludes a new owner."""
    from gateway.status import _is_gateway_runtime_lock_active_strict
    lock = home / "gateway.lock"
    try:
        if os.name != "nt":
            _private_node(lock, kind="file")
        return "starting" if _is_gateway_runtime_lock_active_strict(lock) else "absent"
    except FileNotFoundError:
        return "absent"
    except (OSError, RuntimeError, DiscoveryError):
        return "inaccessible"
