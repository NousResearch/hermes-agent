"""Bounded local-control discovery without diagnostic fallback or mutation."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Literal
import time


class DiscoveryError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _may_have_access_acl(path: Path) -> bool:
    """A Linux POSIX access ACL is present, or its absence cannot be established. Python has no
    os.listxattr on macOS (whose chmod +a ACLs are separate); mode bits stay the policy there."""
    if not hasattr(os, "listxattr"):
        return False
    try:
        return "system.posix_acl_access" in os.listxattr(path, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            return False  # filesystem without xattrs cannot carry an ACL
        return True


def home_mode_unsafe(node: os.stat_result, path: Path) -> bool:
    """The profile home keeps the operator's mode (main's home policy: symlinked homes,
    HERMES_HOME_MODE 0701/0750). Only write by another user lets them swap the socket; read/search
    bits grant nothing against our 0600 socket, and group-write is harmless when the group is the
    owner's private group (the umask-002 user-private-group default). What we create stays 0700."""
    mode = stat.S_IMODE(node.st_mode)
    if mode & 0o002:
        return True
    if not mode & 0o020:
        return False
    # With an access ACL the group bits are the mask: a named user (setfacl u:other:rwx) may
    # hold write even though the owning group is private.
    if _may_have_access_acl(path):
        return True
    # ``gr_mem`` never lists accounts whose PRIMARY gid is the group, so privacy also needs the
    # passwd scan; membership we cannot establish is treated as shared.
    try:
        import grp
        import pwd
        user, group = pwd.getpwuid(node.st_uid), grp.getgrgid(node.st_gid)
        primary_members = {p.pw_uid for p in pwd.getpwall() if p.pw_gid == group.gr_gid}
    except (ImportError, KeyError, OSError):
        return True
    return not (group.gr_gid == user.pw_gid and group.gr_name == user.pw_name
                and set(group.gr_mem) <= {user.pw_name} and primary_members <= {node.st_uid})


def _private_node(path: Path, *, kind: str, home: bool = False) -> os.stat_result:
    node = path.lstat()
    predicates = {"socket": stat.S_ISSOCK, "file": stat.S_ISREG, "directory": stat.S_ISDIR}
    if not predicates[kind](node.st_mode) or node.st_uid != os.getuid():  # windows-footgun: ok — POSIX socket path only
        raise DiscoveryError("unsafe_control_path")
    if home_mode_unsafe(node, path) if home else stat.S_IMODE(node.st_mode) & 0o077:
        raise DiscoveryError("unsafe_control_permissions")
    return node


def socket_peer_uid(sock: socket.socket) -> int | None:
    """Kernel-reported uid of the process on the other end of a connected AF_UNIX socket
    (SO_PEERCRED on Linux, getpeereid on macOS); None where neither exists."""
    if hasattr(socket, "SO_PEERCRED"):
        import struct
        return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
    if sys.platform != "darwin":
        return None
    import ctypes
    uid, gid = ctypes.c_uint(), ctypes.c_uint()
    getpeereid = ctypes.CDLL(None, use_errno=True).getpeereid
    getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    getpeereid.restype = ctypes.c_int
    if getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
        raise OSError(ctypes.get_errno(), "getpeereid failed")
    return uid.value


def connect_private(home: Path, timeout: float) -> socket.socket:
    """Connected control socket for *home* whose listener runs as this user (caller closes it).
    The path is validated by lstat first, but connect() follows a symlink swapped in afterwards,
    so the listening process itself is authenticated. Platforms without a peer-credential API
    (neither SO_PEERCRED nor getpeereid) fall back to the path checks alone."""
    path = _socket_path(home)
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        peer.settimeout(timeout)
        peer.connect(str(path))
        uid = socket_peer_uid(peer)
        if uid is not None and uid != os.getuid():  # windows-footgun: ok — POSIX socket peer only
            raise DiscoveryError("unsafe_control_peer")
        return peer
    except BaseException:
        peer.close()
        raise


def _socket_path(home: Path) -> Path:
    _private_node(home, kind="directory", home=True)
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
    request = b'{"protocol":1,"verb":"identify","id":1}\n'
    with connect_private(home, timeout) as client:
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
