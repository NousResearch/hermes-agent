"""Read-only snapshots of a realm's wlr foreign-toplevel protocol.

No capture, input, accessibility or process-count fallback. This small Wayland
wire client binds only the registry and foreign-toplevel manager (v1); it never
binds a seat or sends a window-management request. Two display syncs delimit
registry discovery and the initial toplevel snapshot.
"""

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

from collections import OrderedDict
import threading

from .lifecycle import RealmError, identity, validate_environment, validate_live


class WindowCounter:
    """Per-service bounded cache; short-lived unknowns are cached too."""

    def __init__(self, *, ttl=1.0, max_entries=128):
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def count(self, record):
        key = (
            record["home"],
            record["id"],
            record["generation"],
            record["runtime_dir"],
        )
        with self._lock:
            now = time.monotonic()
            cached = self._cache.get(key)
            if cached is not None and now < cached[0]:
                self._cache.move_to_end(key)
                return cached[1]
            value = count_windows(record)
            self._cache[key] = (time.monotonic() + self.ttl, value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return value


class WindowCountUnavailable(RealmError):
    pass


def _message(object_id, opcode, payload=b""):
    return struct.pack("=II", object_id, ((len(payload) + 8) << 16) | opcode) + payload


def _string(value):
    encoded = value.encode() + b"\0"
    return struct.pack("=I", len(encoded)) + encoded + b"\0" * (-len(encoded) % 4)


def _snapshot(connection, timeout):
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    received = 0
    windows = set()
    manager_name = None

    def roundtrip(callback):
        nonlocal received, manager_name
        connection.sendall(_message(1, 0, struct.pack("=I", callback)))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WindowCountUnavailable("Compositor snapshot timed out")
            connection.settimeout(remaining)
            while (
                len(buffer) < 8
                or len(buffer) < struct.unpack_from("=I", buffer, 4)[0] >> 16
            ):
                chunk = connection.recv(65536)
                if not chunk:
                    raise WindowCountUnavailable("Compositor disconnected")
                received += len(chunk)
                if received > 4 * 1024 * 1024:
                    raise WindowCountUnavailable("Compositor snapshot exceeds limit")
                buffer.extend(chunk)
            obj, header = struct.unpack_from("=II", buffer)
            size, opcode = header >> 16, header & 0xFFFF
            if size < 8 or size % 4:
                raise WindowCountUnavailable("Invalid Wayland message")
            payload = bytes(buffer[8:size])
            del buffer[:size]
            if obj == callback and opcode == 0:
                return
            if obj == 1 and opcode == 0:
                raise WindowCountUnavailable("Compositor rejected snapshot request")
            if obj == 2 and opcode == 0:
                name, length = struct.unpack_from("=II", payload)
                if payload[8 : 8 + length] == b"zwlr_foreign_toplevel_manager_v1\0":
                    manager_name = name
            elif obj == 4 and opcode == 0:
                windows.add(struct.unpack("=I", payload)[0])
            elif obj == 4 and opcode == 1:
                raise WindowCountUnavailable("Toplevel manager finished")
            elif obj in windows and opcode == 6:
                windows.remove(obj)

    connection.sendall(_message(1, 1, struct.pack("=I", 2)))  # get_registry
    roundtrip(3)
    if manager_name is None:
        raise WindowCountUnavailable("Foreign-toplevel protocol unavailable")
    connection.sendall(
        _message(
            2,
            0,
            struct.pack("=I", manager_name)
            + _string("zwlr_foreign_toplevel_manager_v1")
            + struct.pack("=II", 1, 4),
        )
    )
    roundtrip(5)
    return len(windows)


def count_windows(record, *, timeout=1.0):
    """Return a compositor count or None; never consult ambient display state."""
    try:
        validate_live(record)
        runtime = Path(record["runtime_dir"])
        env = json.loads((runtime / "ready.json").read_text(encoding="utf-8"))["env"]
        validate_environment(record, env)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(runtime / env["WAYLAND_DISPLAY"]))
            pid, uid, _ = struct.unpack(
                "=3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid() or identity(pid) != record["processes"]["compositor"]:  # windows-footgun: ok — runtime package rejects non-Linux hosts
                raise WindowCountUnavailable("Compositor peer ownership mismatch")
            return _snapshot(connection, timeout)
    except (
        RealmError,
        OSError,
        ValueError,
        KeyError,
        struct.error,
        subprocess.SubprocessError,
    ):
        return None
