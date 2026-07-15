"""Strict AF_UNIX execution boundary for an unprivileged isolated worker.

The protocol in this module is deliberately mechanical.  It does not choose
commands, classify requests, or make task decisions.  A caller supplies an
already model-authored command and an owner-sealed lease; the worker only
validates the fixed transport/session boundary and executes inside that
lease's pre-created workspace.

Network and filesystem namespace isolation are service-manager obligations.
``WorkerPolicy`` requires an attested network-isolated profile and forwards no
ambient environment or credential-file registration into child processes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "muncho.isolated-worker.v1"
REQUEST_SCHEMA = "muncho.isolated-worker.request.v1"
RESPONSE_SCHEMA = "muncho.isolated-worker.response.v1"
MAX_FRAME_BYTES = 256 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_STDIN_BYTES = 128 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_POLL_CHUNK_BYTES = 64 * 1024
MAX_REQUEST_CACHE = 256
MAX_ACTIVE_CONNECTIONS = 128
MAX_ACTIVE_JOBS_PER_CONNECTION = 64
MAX_ACTIVE_JOBS_PER_LEASE_LIMIT = 64
MAX_LEASES_LIMIT = 1024
MAX_LEASE_TTL_SECONDS = 86_400
MAX_LEASE_QUOTA_BYTES = 16 * 1024 * 1024 * 1024
MAX_LEASE_QUOTA_ENTRIES = 1_000_000
MAX_GLOBAL_QUOTA_BYTES = 16 * 1024 * 1024 * 1024
MAX_GLOBAL_QUOTA_ENTRIES = 2_000_000
DEFAULT_BWRAP_PATH = Path("/usr/bin/bwrap")
VIRTUAL_WORKSPACE_ROOT = Path("/workspace")
HOST_READ_ONLY_ROOT = Path("/opt/hermes-shared")
VIRTUAL_READ_ONLY_ROOT = Path("/opt/hermes-shared")
VIRTUAL_SHELL_PATH = Path("/run/hermes-shell")
FIXED_RUNTIME_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
)
_DENIED_READ_ONLY_SOURCE_COMPONENTS = frozenset(
    {".hermes", "credentials", "memory", "memories", "plugins", "secrets", "skills"}
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_REQUEST_FIELDS = frozenset(
    {"schema", "protocol", "request_id", "lease_id", "operation", "parameters"}
)
_PARAMETER_FIELDS = {
    "exec.start": frozenset({"command", "cwd", "stdin_b64", "timeout_seconds"}),
    "exec.poll": frozenset({"session_id", "wait_milliseconds"}),
    "exec.cancel": frozenset({"session_id"}),
}


class ProtocolError(RuntimeError):
    """A stable fail-closed protocol violation."""


@dataclass(frozen=True)
class ReadOnlyBind:
    """One operator-sealed, server-owned read-only tree.

    Bind declarations are loaded from the privileged service configuration;
    the wire protocol deliberately has no mount fields.  Sources must be
    immutable to the worker identity and may only appear below the dedicated
    ``/opt/hermes-shared`` namespace inside the sandbox.
    """

    source: Path
    destination: Path
    source_uid: int = 0
    source_gid: int = 0

    def __post_init__(self) -> None:
        source = Path(self.source)
        destination = Path(self.destination)
        if (
            not source.is_absolute()
            or source != Path(os.path.normpath(source))
            or not destination.is_absolute()
            or destination != Path(os.path.normpath(destination))
        ):
            raise ValueError("read_only_bind_path_invalid")
        try:
            destination_relative = destination.relative_to(VIRTUAL_READ_ONLY_ROOT)
        except ValueError as exc:
            raise ValueError("read_only_bind_destination_invalid") from exc
        if len(destination_relative.parts) != 1:
            raise ValueError("read_only_bind_destination_invalid")
        try:
            source_relative = source.relative_to(HOST_READ_ONLY_ROOT)
        except ValueError as exc:
            raise ValueError("read_only_bind_source_namespace_invalid") from exc
        if len(source_relative.parts) != 1:
            raise ValueError("read_only_bind_source_namespace_invalid")
        if any(
            component.lower() in _DENIED_READ_ONLY_SOURCE_COMPONENTS
            for component in source.parts
        ):
            raise ValueError("read_only_bind_source_forbidden")
        if type(self.source_uid) is not int or self.source_uid < 0:
            raise ValueError("read_only_bind_uid_invalid")
        if type(self.source_gid) is not int or self.source_gid < 0:
            raise ValueError("read_only_bind_gid_invalid")
        _verify_read_only_tree(
            source,
            expected_uid=self.source_uid,
            expected_gid=self.source_gid,
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ProtocolError(f"{label}_fields_not_exact")
    return value


def _bounded_text(value: Any, *, maximum: int, label: str) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
        raise ProtocolError(f"{label}_invalid")
    if "\x00" in value:
        raise ProtocolError(f"{label}_invalid")
    return value


def _reject_constant(_value: str) -> None:
    raise ProtocolError("request_json_invalid")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("request_json_duplicate_key")
        result[key] = value
    return result


def parse_request(frame: bytes) -> Mapping[str, Any]:
    """Parse one canonical request and reject aliases or extra fields."""

    if not frame or len(frame) > MAX_FRAME_BYTES or b"\n" in frame:
        raise ProtocolError("request_frame_invalid")
    try:
        value = json.loads(
            frame.decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("request_json_invalid") from exc
    if canonical_bytes(value) != frame:
        raise ProtocolError("request_not_canonical")
    raw = _exact_mapping(value, _REQUEST_FIELDS, "request")
    operation = raw["operation"]
    if (
        raw["schema"] != REQUEST_SCHEMA
        or raw["protocol"] != PROTOCOL
        or not isinstance(operation, str)
        or operation not in _PARAMETER_FIELDS
        or not isinstance(raw["request_id"], str)
        or _REQUEST_ID.fullmatch(raw["request_id"]) is None
        or not isinstance(raw["lease_id"], str)
        or _ID.fullmatch(raw["lease_id"]) is None
    ):
        raise ProtocolError("request_identity_invalid")
    params = _exact_mapping(
        raw["parameters"], _PARAMETER_FIELDS[operation], "request_parameters"
    )
    if operation == "exec.start":
        _bounded_text(params["command"], maximum=MAX_COMMAND_BYTES, label="command")
        _bounded_text(params["cwd"], maximum=4096, label="cwd")
        if not isinstance(params["stdin_b64"], str):
            raise ProtocolError("stdin_invalid")
        try:
            stdin = base64.b64decode(params["stdin_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ProtocolError("stdin_invalid") from exc
        if len(stdin) > MAX_STDIN_BYTES:
            raise ProtocolError("stdin_invalid")
        timeout = params["timeout_seconds"]
        if type(timeout) is not int or not 1 <= timeout <= 300:
            raise ProtocolError("timeout_invalid")
    elif operation == "exec.poll":
        if (
            not isinstance(params["session_id"], str)
            or _ID.fullmatch(params["session_id"]) is None
            or type(params["wait_milliseconds"]) is not int
            or not 0 <= params["wait_milliseconds"] <= 1000
        ):
            raise ProtocolError("poll_parameters_invalid")
    elif not isinstance(params["session_id"], str) or _ID.fullmatch(
        params["session_id"]
    ) is None:
        raise ProtocolError("cancel_parameters_invalid")
    return raw


def _response(request: Mapping[str, Any], *, ok: bool, result: Mapping[str, Any]) -> bytes:
    value = {
        "schema": RESPONSE_SCHEMA,
        "protocol": PROTOCOL,
        "request_id": request["request_id"],
        "lease_id": request["lease_id"],
        "operation": request["operation"],
        "ok": ok,
        "result": dict(result),
    }
    payload = canonical_bytes(value)
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError("response_frame_too_large")
    return payload


def _read_frame(stream) -> bytes | None:
    frame = stream.readline(MAX_FRAME_BYTES + 2)
    if frame == b"":
        return None
    if len(frame) > MAX_FRAME_BYTES + 1 or not frame.endswith(b"\n"):
        raise ProtocolError("request_frame_invalid")
    return frame[:-1]


def _write_frame(stream, frame: bytes) -> None:
    stream.write(frame + b"\n")
    stream.flush()


def _peer_credentials(connection: socket.socket) -> tuple[int, int]:
    """Return peer uid/gid on Linux/BSD, or fail closed."""

    if hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _pid, uid, gid = struct.unpack("3i", raw)
        return uid, gid
    getpeereid = getattr(connection, "getpeereid", None)
    if callable(getpeereid):
        uid, gid = getpeereid()
        return int(uid), int(gid)
    raise ProtocolError("peer_credentials_unavailable")


def _verify_read_only_tree(
    root: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> None:
    """Reject mutable, linked, or special entries in a configured RO tree."""

    root = Path(root)
    root_state = os.lstat(root)
    if (
        not stat.S_ISDIR(root_state.st_mode)
        or stat.S_ISLNK(root_state.st_mode)
        or root_state.st_uid != expected_uid
        or root_state.st_gid != expected_gid
        or stat.S_IMODE(root_state.st_mode) & 0o222
        or root_state.st_nlink < 2
    ):
        raise ValueError("read_only_bind_identity_invalid")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            candidate = current_path / name
            item = os.lstat(candidate)
            if (
                stat.S_ISLNK(item.st_mode)
                or not (stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode))
                or item.st_uid != expected_uid
                or item.st_gid != expected_gid
                or stat.S_IMODE(item.st_mode) & 0o222
                or (stat.S_ISREG(item.st_mode) and item.st_nlink != 1)
            ):
                raise ValueError("read_only_bind_tree_not_sealed")


@dataclass(frozen=True)
class WorkerPolicy:
    """Owner-sealed mechanical policy for one unprivileged worker instance."""

    expected_peer_uid: int
    expected_peer_gid: int
    socket_uid: int
    socket_gid: int
    lease_base: Path
    lease_uid: int
    lease_gid: int
    network_isolated: bool
    bwrap_path: Path
    bwrap_sha256: str
    shell_sha256: str
    bwrap_uid: int = 0
    shell: Path = Path("/bin/bash")
    shell_uid: int = 0
    runtime_roots: tuple[Path, ...] = FIXED_RUNTIME_ROOTS
    maximum_timeout_seconds: int = 300
    maximum_output_bytes: int = MAX_OUTPUT_BYTES
    maximum_active_leases: int = 128
    maximum_active_jobs_per_lease: int = 8
    lease_ttl_seconds: int = 900
    lease_quota_bytes: int = 4 * 1024 * 1024 * 1024
    lease_quota_entries: int = 100_000
    global_quota_bytes: int = 4 * 1024 * 1024 * 1024
    global_quota_entries: int = 200_000
    read_only_binds: tuple[ReadOnlyBind, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "expected_peer_uid",
            "expected_peer_gid",
            "socket_uid",
            "socket_gid",
            "bwrap_uid",
            "shell_uid",
            "lease_uid",
            "lease_gid",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name}_invalid")
        if self.network_isolated is not True:
            raise ValueError("worker_network_namespace_not_attested")
        lease_base = Path(self.lease_base)
        if (
            not lease_base.is_absolute()
            or lease_base != Path(os.path.normpath(lease_base))
        ):
            raise ValueError("lease_base_invalid")
        base_state = os.lstat(lease_base)
        if (
            not stat.S_ISDIR(base_state.st_mode)
            or stat.S_ISLNK(base_state.st_mode)
            or base_state.st_uid != self.lease_uid
            or base_state.st_gid != self.lease_gid
            or stat.S_IMODE(base_state.st_mode) != 0o700
            or base_state.st_nlink < 2
        ):
            raise ValueError("lease_base_identity_invalid")
        bwrap = Path(self.bwrap_path)
        if not bwrap.is_absolute() or bwrap != Path(os.path.normpath(bwrap)):
            raise ValueError("bwrap_path_invalid")
        shell = Path(self.shell)
        if not shell.is_absolute() or shell != Path(os.path.normpath(shell)):
            raise ValueError("shell_path_invalid")
        _verify_regular_digest(
            bwrap,
            expected_sha256=self.bwrap_sha256,
            expected_uid=self.bwrap_uid,
        )
        _verify_regular_digest(
            shell,
            expected_sha256=self.shell_sha256,
            expected_uid=self.shell_uid,
        )
        if not 1 <= self.maximum_timeout_seconds <= 300:
            raise ValueError("worker_timeout_invalid")
        if not 4096 <= self.maximum_output_bytes <= MAX_OUTPUT_BYTES:
            raise ValueError("worker_output_limit_invalid")
        if not 1 <= self.maximum_active_leases <= MAX_LEASES_LIMIT:
            raise ValueError("maximum_active_leases_invalid")
        if not 1 <= self.maximum_active_jobs_per_lease <= MAX_ACTIVE_JOBS_PER_LEASE_LIMIT:
            raise ValueError("maximum_active_jobs_per_lease_invalid")
        if not 1 <= self.lease_ttl_seconds <= MAX_LEASE_TTL_SECONDS:
            raise ValueError("lease_ttl_invalid")
        if not 4096 <= self.lease_quota_bytes <= MAX_LEASE_QUOTA_BYTES:
            raise ValueError("lease_quota_bytes_invalid")
        if not 1 <= self.lease_quota_entries <= MAX_LEASE_QUOTA_ENTRIES:
            raise ValueError("lease_quota_entries_invalid")
        if not 4096 <= self.global_quota_bytes <= MAX_GLOBAL_QUOTA_BYTES:
            raise ValueError("global_quota_bytes_invalid")
        if not 1 <= self.global_quota_entries <= MAX_GLOBAL_QUOTA_ENTRIES:
            raise ValueError("global_quota_entries_invalid")
        if self.lease_quota_bytes > self.global_quota_bytes:
            raise ValueError("lease_quota_exceeds_global_bytes")
        # The service-wide entry accounting includes each lease directory.
        # Keep one full lease representable by the aggregate policy.
        if self.lease_quota_entries + 1 > self.global_quota_entries:
            raise ValueError("lease_quota_exceeds_global_entries")
        if tuple(self.runtime_roots) != FIXED_RUNTIME_ROOTS:
            raise ValueError("runtime_roots_not_exact")
        if not isinstance(self.read_only_binds, tuple) or any(
            not isinstance(item, ReadOnlyBind) for item in self.read_only_binds
        ):
            raise ValueError("read_only_binds_invalid")
        destinations: set[Path] = set()
        for item in self.read_only_binds:
            if item.source_uid == self.lease_uid:
                raise ValueError("read_only_bind_mutable_by_worker")
            try:
                item.source.relative_to(lease_base)
            except ValueError:
                pass
            else:
                raise ValueError("read_only_bind_source_is_lease")
            if item.destination in destinations:
                raise ValueError("read_only_bind_destination_duplicate")
            destinations.add(item.destination)
        object.__setattr__(self, "lease_base", lease_base)
        object.__setattr__(self, "bwrap_path", bwrap)
        object.__setattr__(self, "shell", shell)


def _verify_regular_digest(
    path: Path,
    *,
    expected_sha256: str,
    expected_uid: int,
) -> os.stat_result:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return _verify_open_regular_digest(
            path,
            descriptor,
            expected_sha256=expected_sha256,
            expected_uid=expected_uid,
        )
    finally:
        os.close(descriptor)


def _verify_open_regular_digest(
    path: Path,
    descriptor: int,
    *,
    expected_sha256: str,
    expected_uid: int,
) -> os.stat_result:
    """Verify the exact already-open descriptor later passed to bwrap."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("executable_digest_invalid")
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or stat.S_IMODE(before.st_mode) & 0o022
        or not stat.S_IMODE(before.st_mode) & 0o111
    ):
        raise ValueError("executable_identity_invalid")
    opened = os.fstat(descriptor)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = os.lstat(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise ValueError("executable_changed_during_verification")
    if digest != expected_sha256:
        raise ValueError("executable_digest_mismatch")
    return before


@dataclass
class _Lease:
    lease_id: str
    root: Path
    created_monotonic: float
    last_used_monotonic: float
    connections: int = 0
    jobs: int = 0


def canonical_lease_id(session_id: str) -> str:
    """Mechanically derive one stable, path-safe lease from a session id."""

    if not isinstance(session_id, str) or not session_id or len(session_id) > 4096:
        raise ValueError("session_id_invalid")
    return "lease-" + hashlib.sha256(session_id.encode("utf-8")).hexdigest()


@dataclass
class _Execution:
    lease: _Lease
    process: subprocess.Popen[bytes]
    timeout_seconds: int
    output_limit: int
    started_monotonic: float = field(default_factory=time.monotonic)
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    stdout_sent: int = 0
    stderr_sent: int = 0
    state: str = "running"
    lock: threading.Lock = field(default_factory=threading.Lock)
    complete: threading.Event = field(default_factory=threading.Event)
    stdout_complete: threading.Event = field(default_factory=threading.Event)
    stderr_complete: threading.Event = field(default_factory=threading.Event)

    def terminate(self, state: str) -> None:
        with self.lock:
            if self.state != "running":
                return
            self.state = state
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.process.kill()
            except OSError:
                pass


class IsolatedWorkerServer:
    """Threaded connection handler for one pre-created AF_UNIX listener."""

    def __init__(self, policy: WorkerPolicy):
        self.policy = policy
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        self._replay: dict[str, tuple[bytes, bytes]] = {}
        self._replay_lock = threading.Lock()
        self._leases: dict[str, _Lease] = {}
        self._leases_lock = threading.RLock()
        self._lease_base_fd = os.open(
            self.policy.lease_base,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        base = os.fstat(self._lease_base_fd)
        self._lease_base_identity = (base.st_dev, base.st_ino)
        self._read_only_bind_fds: list[
            tuple[ReadOnlyBind, int, tuple[int, int]]
        ] = []
        try:
            for bind in self.policy.read_only_binds:
                descriptor = os.open(
                    bind.source,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                state = os.fstat(descriptor)
                self._read_only_bind_fds.append(
                    (bind, descriptor, (state.st_dev, state.st_ino))
                )
        except BaseException:
            for _bind, descriptor, _identity in self._read_only_bind_fds:
                os.close(descriptor)
            os.close(self._lease_base_fd)
            raise
        self._validate_lease_base()

    def close(self) -> None:
        for _bind, descriptor, _identity in self._read_only_bind_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._read_only_bind_fds.clear()
        try:
            os.close(self._lease_base_fd)
        except OSError:
            pass

    def _validate_lease_base(self) -> None:
        path_state = os.lstat(self.policy.lease_base)
        opened = os.fstat(self._lease_base_fd)
        if (
            not stat.S_ISDIR(path_state.st_mode)
            or stat.S_ISLNK(path_state.st_mode)
            or (path_state.st_dev, path_state.st_ino) != self._lease_base_identity
            or (opened.st_dev, opened.st_ino) != self._lease_base_identity
            or path_state.st_uid != self.policy.lease_uid
            or path_state.st_gid != self.policy.lease_gid
            or stat.S_IMODE(path_state.st_mode) != 0o700
            or path_state.st_nlink < 2
        ):
            raise ProtocolError("lease_base_identity_drifted")

    def _validate_read_only_binds(self) -> None:
        for bind, descriptor, identity in self._read_only_bind_fds:
            _verify_read_only_tree(
                bind.source,
                expected_uid=bind.source_uid,
                expected_gid=bind.source_gid,
            )
            opened = os.fstat(descriptor)
            current = os.lstat(bind.source)
            if (
                (opened.st_dev, opened.st_ino) != identity
                or (current.st_dev, current.st_ino) != identity
            ):
                raise ProtocolError("read_only_bind_identity_drifted")

    @staticmethod
    def _canonical_dynamic_lease_id(lease_id: str) -> bool:
        return re.fullmatch(r"lease-[0-9a-f]{64}", lease_id) is not None

    def _lease_root_state(self, lease_id: str) -> os.stat_result:
        state = os.stat(lease_id, dir_fd=self._lease_base_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(state.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or state.st_uid != self.policy.lease_uid
            or state.st_gid != self.policy.lease_gid
            or stat.S_IMODE(state.st_mode) != 0o700
            or state.st_nlink < 2
        ):
            raise ProtocolError("lease_root_identity_invalid")
        return state

    def _load_existing_leases_locked(self, now: float) -> None:
        self._validate_lease_base()
        for name in os.listdir(self._lease_base_fd):
            if not self._canonical_dynamic_lease_id(name):
                raise ProtocolError("lease_base_contains_unmanaged_entry")
            state = self._lease_root_state(name)
            persisted_age = max(0.0, time.time() - state.st_mtime)
            self._leases.setdefault(
                name,
                _Lease(
                    lease_id=name,
                    root=self.policy.lease_base / name,
                    created_monotonic=now - persisted_age,
                    last_used_monotonic=now - persisted_age,
                ),
            )

    def _touch_lease_locked(self, lease: _Lease, now: float | None = None) -> None:
        self._lease_root_state(lease.lease_id)
        os.utime(
            lease.lease_id,
            None,
            dir_fd=self._lease_base_fd,
            follow_symlinks=False,
        )
        lease.last_used_monotonic = time.monotonic() if now is None else now

    def _ensure_lease(self, lease_id: str) -> _Lease:
        if not self._canonical_dynamic_lease_id(lease_id):
            raise ProtocolError("lease_id_not_canonical")
        now = time.monotonic()
        self.reap_expired(now_monotonic=now)
        with self._leases_lock:
            self._load_existing_leases_locked(now)
            existing = self._leases.get(lease_id)
            if existing is not None:
                self._global_usage_locked()
                self._lease_root_state(lease_id)
                self._touch_lease_locked(existing, now)
                return existing
            if len(self._leases) >= self.policy.maximum_active_leases:
                raise ProtocolError("lease_capacity_exhausted")
            # Account for the lease root before creating it so an empty lease
            # cannot push the service above its aggregate inode/entry bound.
            self._global_usage_locked(additional_entries=1)
            try:
                os.mkdir(lease_id, mode=0o700, dir_fd=self._lease_base_fd)
            except FileExistsError:
                pass
            os.chown(
                lease_id,
                self.policy.lease_uid,
                self.policy.lease_gid,
                dir_fd=self._lease_base_fd,
                follow_symlinks=False,
            )
            os.chmod(
                lease_id,
                0o700,
                dir_fd=self._lease_base_fd,
                follow_symlinks=False,
            )
            self._lease_root_state(lease_id)
            lease = _Lease(
                lease_id=lease_id,
                root=self.policy.lease_base / lease_id,
                created_monotonic=now,
                last_used_monotonic=now,
            )
            self._leases[lease_id] = lease
            self._touch_lease_locked(lease, now)
            return lease

    def reap_expired(self, *, now_monotonic: float | None = None) -> tuple[str, ...]:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        removed: list[str] = []
        with self._leases_lock:
            self._load_existing_leases_locked(now)
            for lease_id, lease in tuple(self._leases.items()):
                if (
                    lease.connections
                    or lease.jobs
                    or now - lease.last_used_monotonic < self.policy.lease_ttl_seconds
                ):
                    continue
                self._validate_lease_base()
                self._lease_root_state(lease_id)
                shutil.rmtree(lease_id, dir_fd=self._lease_base_fd)
                self._leases.pop(lease_id, None)
                removed.append(lease_id)
        return tuple(sorted(removed))

    def _validate_cwd(self, lease: _Lease, cwd: str) -> Path:
        candidate = Path(cwd)
        if not candidate.is_absolute() or candidate != Path(os.path.normpath(cwd)):
            raise ProtocolError("cwd_invalid")
        try:
            relative = candidate.relative_to(VIRTUAL_WORKSPACE_ROOT)
        except ValueError as exc:
            raise ProtocolError("cwd_outside_lease") from exc
        descriptor = os.open(
            lease.lease_id,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self._lease_base_fd,
        )
        try:
            for component in relative.parts:
                if component in {"", ".", ".."}:
                    raise ProtocolError("cwd_invalid")
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
        except OSError as exc:
            raise ProtocolError("cwd_symlink_or_not_directory") from exc
        finally:
            os.close(descriptor)
        return candidate

    def _lease_usage(self, lease: _Lease) -> tuple[int, int]:
        entries = 0
        total_bytes = 0
        self._lease_root_state(lease.lease_id)
        for current, directories, files in os.walk(
            lease.root, topdown=True, followlinks=False
        ):
            directories.sort()
            files.sort()
            for name in (*directories, *files):
                state = os.lstat(Path(current) / name)
                entries += 1
                if stat.S_ISREG(state.st_mode):
                    total_bytes += state.st_size
                elif stat.S_ISLNK(state.st_mode):
                    total_bytes += state.st_size
                elif stat.S_ISFIFO(state.st_mode) or stat.S_ISSOCK(state.st_mode):
                    total_bytes += state.st_size
                elif not stat.S_ISDIR(state.st_mode):
                    raise ProtocolError("lease_contains_special_file")
                if (
                    entries > self.policy.lease_quota_entries
                    or total_bytes > self.policy.lease_quota_bytes
                ):
                    raise ProtocolError("lease_quota_exceeded")
        return entries, total_bytes

    def _global_usage_locked(
        self,
        *,
        additional_entries: int = 0,
        additional_bytes: int = 0,
    ) -> tuple[int, int]:
        """Measure every managed lease under the service-wide quota.

        Callers hold ``_leases_lock`` so lease admission and aggregate
        accounting cannot race each other.  Workspace processes can still
        mutate files while they run; every scan error is therefore fail-closed
        in the same way as the existing per-lease monitor.
        """

        self._load_existing_leases_locked(time.monotonic())
        entries = additional_entries
        total_bytes = additional_bytes
        for lease_id in sorted(self._leases):
            lease_entries, lease_bytes = self._lease_usage(self._leases[lease_id])
            entries += lease_entries + 1  # include the lease directory itself
            total_bytes += lease_bytes
            if (
                entries > self.policy.global_quota_entries
                or total_bytes > self.policy.global_quota_bytes
            ):
                raise ProtocolError("global_quota_exceeded")
        return entries, total_bytes

    def _global_usage(self) -> tuple[int, int]:
        with self._leases_lock:
            return self._global_usage_locked()

    @staticmethod
    def _peer_uid(connection: socket.socket) -> int:
        return _peer_credentials(connection)[0]

    def _validate_listener(self, listener: socket.socket) -> None:
        path = listener.getsockname()
        if not isinstance(path, str) or not path:
            raise ProtocolError("worker_listener_path_invalid")
        item = os.lstat(path)
        if (
            not stat.S_ISSOCK(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != self.policy.socket_uid
            or item.st_gid != self.policy.socket_gid
            or stat.S_IMODE(item.st_mode) != 0o660
        ):
            raise ProtocolError("worker_listener_identity_invalid")

    def serve_connection(self, connection: socket.socket) -> None:
        executions: dict[str, _Execution] = {}
        bound_lease: _Lease | None = None
        try:
            peer_uid, peer_gid = _peer_credentials(connection)
            if (
                peer_uid != self.policy.expected_peer_uid
                or peer_gid != self.policy.expected_peer_gid
            ):
                raise ProtocolError("peer_uid_not_authorized")
            reader = connection.makefile("rb", buffering=0)
            writer = connection.makefile("wb", buffering=0)
            while True:
                raw = _read_frame(reader)
                if raw is None:
                    break
                request = parse_request(raw)
                request_lease_id = str(request["lease_id"])
                if bound_lease is None:
                    bound_lease = self._ensure_lease(request_lease_id)
                    with self._leases_lock:
                        bound_lease.connections += 1
                elif request_lease_id != bound_lease.lease_id:
                    raise ProtocolError("connection_lease_changed")
                with self._leases_lock:
                    self._touch_lease_locked(bound_lease)
                request_id = str(request["request_id"])
                with self._replay_lock:
                    cached = self._replay.get(request_id)
                    if cached is not None:
                        if cached[0] != raw:
                            raise ProtocolError("request_id_reused")
                        response = cached[1]
                    else:
                        response = b""
                if cached is not None:
                    _write_frame(writer, response)
                    continue
                response = self._dispatch(request, executions)
                with self._replay_lock:
                    if len(self._replay) >= MAX_REQUEST_CACHE:
                        self._replay.pop(next(iter(self._replay)))
                    self._replay[request_id] = (raw, response)
                _write_frame(writer, response)
        finally:
            for execution in tuple(executions.values()):
                execution.terminate("disconnected")
                execution.complete.wait(2)
            if bound_lease is not None:
                with self._leases_lock:
                    bound_lease.connections = max(0, bound_lease.connections - 1)
                    bound_lease.jobs = max(0, bound_lease.jobs - len(executions))
                    self._touch_lease_locked(bound_lease)
            try:
                connection.close()
            except OSError:
                pass

    def serve(self, listener: socket.socket, stop: threading.Event) -> None:
        self._validate_listener(listener)
        # Startup readiness is fail-closed when recent persisted workspaces are
        # already above the service-wide quota.  Expired leases are reclaimed
        # first so a clean restart does not require operator intervention.
        self.reap_expired()
        self._global_usage()
        listener.settimeout(0.1)
        reap_interval = min(60.0, max(1.0, self.policy.lease_ttl_seconds / 2))
        next_reap = time.monotonic() + reap_interval
        while not stop.is_set():
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                if time.monotonic() >= next_reap:
                    self.reap_expired()
                    next_reap = time.monotonic() + reap_interval
                continue

            with self._threads_lock:
                if len(self._threads) >= MAX_ACTIVE_CONNECTIONS:
                    connection.close()
                    continue

            def run_connection() -> None:
                try:
                    self.serve_connection(connection)
                finally:
                    with self._threads_lock:
                        self._threads.discard(threading.current_thread())

            thread = threading.Thread(target=run_connection, daemon=True)
            with self._threads_lock:
                self._threads.add(thread)
            thread.start()
        with self._threads_lock:
            threads = tuple(self._threads)
        for thread in threads:
            thread.join(timeout=2)

    def _dispatch(
        self,
        request: Mapping[str, Any],
        executions: dict[str, _Execution],
    ) -> bytes:
        try:
            operation = request["operation"]
            params = request["parameters"]
            if operation == "exec.start":
                result = self._start(request["lease_id"], params, executions)
            elif operation == "exec.poll":
                result = self._poll(request["lease_id"], params, executions)
            else:
                result = self._cancel(request["lease_id"], params, executions)
            return _response(request, ok=True, result=result)
        except ProtocolError as exc:
            return _response(
                request,
                ok=False,
                result={"error_code": str(exc) or "protocol_error"},
            )
        except OSError:
            return _response(request, ok=False, result={"error_code": "worker_os_error"})
        except ValueError:
            return _response(request, ok=False, result={"error_code": "worker_value_error"})

    def _start(
        self,
        lease_id: str,
        params: Mapping[str, Any],
        executions: dict[str, _Execution],
    ) -> Mapping[str, Any]:
        if len(executions) >= MAX_ACTIVE_JOBS_PER_CONNECTION:
            raise ProtocolError("active_job_limit_reached")
        lease = self._ensure_lease(lease_id)
        with self._leases_lock:
            if lease.jobs >= self.policy.maximum_active_jobs_per_lease:
                raise ProtocolError("lease_job_capacity_exhausted")
            self._global_usage_locked()
            lease.jobs += 1
            self._touch_lease_locked(lease)
        try:
            virtual_cwd = self._validate_cwd(lease, params["cwd"])
            timeout = min(
                params["timeout_seconds"], self.policy.maximum_timeout_seconds
            )
            stdin = base64.b64decode(params["stdin_b64"], validate=True)
            # Exact allowlist: never merge os.environ or skill-declared
            # environment/credential-file registration.
            environment = {
                "HOME": str(lease.root),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "TMPDIR": str(lease.root),
            }
            process = self._spawn_sandboxed(
                lease=lease,
                virtual_cwd=virtual_cwd,
                command=params["command"],
                environment=environment,
            )
        except BaseException:
            with self._leases_lock:
                lease.jobs = max(0, lease.jobs - 1)
            raise
        session_id = f"job-{uuid.uuid4().hex}"
        execution = _Execution(
            lease=lease,
            process=process,
            timeout_seconds=timeout,
            output_limit=self.policy.maximum_output_bytes,
        )
        executions[session_id] = execution

        def drain(stream, target: bytearray, done: threading.Event) -> None:
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        return
                    with execution.lock:
                        remaining = execution.output_limit - len(execution.stdout) - len(execution.stderr)
                        if remaining <= 0:
                            overflow = True
                        else:
                            target.extend(chunk[:remaining])
                            overflow = len(chunk) > remaining
                    if overflow:
                        execution.terminate("output_limit")
                        return
            finally:
                try:
                    stream.close()
                except OSError:
                    pass
                done.set()

        assert process.stdout is not None and process.stderr is not None
        threading.Thread(
            target=drain,
            args=(process.stdout, execution.stdout, execution.stdout_complete),
            daemon=True,
        ).start()
        threading.Thread(
            target=drain,
            args=(process.stderr, execution.stderr, execution.stderr_complete),
            daemon=True,
        ).start()

        def feed() -> None:
            assert process.stdin is not None
            try:
                process.stdin.write(stdin)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=feed, daemon=True).start()

        def monitor() -> None:
            try:
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        execution.terminate("timed_out")
                        break
                    try:
                        self._global_usage()
                    except (OSError, ProtocolError):
                        # A process can rename workspace entries while the
                        # scanner walks them. Treat every scan failure as a
                        # fail-closed quota/integrity result; otherwise the
                        # monitor thread could die while the child survives
                        # without timeout or quota enforcement.
                        execution.terminate("quota_exceeded")
                        break
                    time.sleep(0.05)
                process.wait()
                with execution.lock:
                    if execution.state == "running":
                        execution.state = "exited"
            finally:
                execution.stdout_complete.wait(2)
                execution.stderr_complete.wait(2)
                execution.complete.set()

        threading.Thread(target=monitor, daemon=True).start()
        return {"session_id": session_id, "state": "running"}

    def _spawn_sandboxed(
        self,
        *,
        lease: _Lease,
        virtual_cwd: Path,
        command: str,
        environment: Mapping[str, str],
    ) -> subprocess.Popen[bytes]:
        """Launch through the exact verified bwrap inode; never raw-fallback."""

        lease_descriptor = -1
        descriptor = -1
        shell_descriptor = -1
        try:
            lease_descriptor = os.open(
                lease.lease_id,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._lease_base_fd,
            )
            descriptor = os.open(
                self.policy.bwrap_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            shell_descriptor = os.open(
                self.policy.shell,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            self._validate_read_only_binds()
            opened = _verify_open_regular_digest(
                self.policy.bwrap_path,
                descriptor,
                expected_sha256=self.policy.bwrap_sha256,
                expected_uid=self.policy.bwrap_uid,
            )
            shell_opened = _verify_open_regular_digest(
                self.policy.shell,
                shell_descriptor,
                expected_sha256=self.policy.shell_sha256,
                expected_uid=self.policy.shell_uid,
            )
            current = os.lstat(self.policy.bwrap_path)
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise ProtocolError("bwrap_changed_before_exec")
            shell_current = os.lstat(self.policy.shell)
            if (shell_opened.st_dev, shell_opened.st_ino) != (
                shell_current.st_dev,
                shell_current.st_ino,
            ):
                raise ProtocolError("shell_changed_before_exec")
            executable = f"/proc/self/fd/{descriptor}"
            arguments = [
                executable,
                "--die-with-parent",
                "--new-session",
                "--unshare-all",
                "--cap-drop",
                "ALL",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--dir",
                "/run",
                "--dir",
                "/opt",
                "--dir",
                str(VIRTUAL_READ_ONLY_ROOT),
                "--ro-bind-fd",
                str(shell_descriptor),
                str(VIRTUAL_SHELL_PATH),
                "--tmpfs",
                "/tmp",
                "--dir",
                str(VIRTUAL_WORKSPACE_ROOT),
            ]
            for root in self.policy.runtime_roots:
                if root.exists():
                    arguments.extend(("--ro-bind", str(root), str(root)))
            for bind, bind_descriptor, _identity in self._read_only_bind_fds:
                arguments.extend(
                    (
                        "--ro-bind",
                        f"/proc/self/fd/{bind_descriptor}",
                        str(bind.destination),
                    )
                )
            arguments.extend(
                (
                    "--bind",
                    f"/proc/self/fd/{lease_descriptor}",
                    str(VIRTUAL_WORKSPACE_ROOT),
                    "--chdir",
                    str(virtual_cwd),
                    "--clearenv",
                )
            )
            for key, value in sorted(environment.items()):
                virtual_value = (
                    str(VIRTUAL_WORKSPACE_ROOT)
                    if value == str(lease.root)
                    else value
                )
                arguments.extend(("--setenv", key, virtual_value))
            arguments.extend(
                (
                    "--",
                    str(VIRTUAL_SHELL_PATH),
                    "--noprofile",
                    "--norc",
                    "-c",
                    command,
                )
            )
            return subprocess.Popen(
                arguments,
                env={},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(
                    descriptor,
                    shell_descriptor,
                    lease_descriptor,
                    *(item[1] for item in self._read_only_bind_fds),
                ),
                start_new_session=True,
            )
        finally:
            for opened_descriptor in (
                shell_descriptor,
                descriptor,
                lease_descriptor,
            ):
                if opened_descriptor >= 0:
                    os.close(opened_descriptor)

    def _poll(
        self,
        lease_id: str,
        params: Mapping[str, Any],
        executions: dict[str, _Execution],
    ) -> Mapping[str, Any]:
        session_id = params["session_id"]
        execution = executions.get(session_id)
        if execution is None or execution.lease.lease_id != lease_id:
            raise ProtocolError("session_not_authorized")
        execution.complete.wait(params["wait_milliseconds"] / 1000)
        with execution.lock:
            stdout_end = min(
                len(execution.stdout), execution.stdout_sent + MAX_POLL_CHUNK_BYTES
            )
            stderr_end = min(
                len(execution.stderr), execution.stderr_sent + MAX_POLL_CHUNK_BYTES
            )
            stdout = bytes(execution.stdout[execution.stdout_sent:stdout_end])
            stderr = bytes(execution.stderr[execution.stderr_sent:stderr_end])
            execution.stdout_sent = stdout_end
            execution.stderr_sent = stderr_end
            state = execution.state
            returncode = execution.process.poll()
            drained = stdout_end == len(execution.stdout) and stderr_end == len(execution.stderr)
            complete = execution.complete.is_set()
        result = {
            "session_id": session_id,
            "state": state,
            "returncode": returncode,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "drained": drained,
            "complete": complete,
        }
        if state != "running" and drained and complete:
            executions.pop(session_id, None)
            with self._leases_lock:
                execution.lease.jobs = max(0, execution.lease.jobs - 1)
                self._touch_lease_locked(execution.lease)
        return result

    def _cancel(
        self,
        lease_id: str,
        params: Mapping[str, Any],
        executions: dict[str, _Execution],
    ) -> Mapping[str, Any]:
        session_id = params["session_id"]
        execution = executions.get(session_id)
        if execution is None or execution.lease.lease_id != lease_id:
            raise ProtocolError("session_not_authorized")
        execution.terminate("cancelled")
        execution.complete.wait(1)
        return {"session_id": session_id, "state": execution.state}


class IsolatedWorkerClient:
    """No-fallback client for one lease-bound worker connection."""

    def __init__(
        self,
        socket_path: Path,
        *,
        lease_id: str,
        expected_server_uid: int,
        expected_server_gid: int,
        expected_socket_uid: int,
        expected_socket_gid: int,
    ):
        if _ID.fullmatch(lease_id) is None:
            raise ValueError("lease_id_invalid")
        self.socket_path = Path(socket_path)
        self.lease_id = lease_id
        self.expected_server_uid = expected_server_uid
        self.expected_server_gid = expected_server_gid
        self.expected_socket_uid = expected_socket_uid
        self.expected_socket_gid = expected_socket_gid
        self._socket: socket.socket | None = None
        self._reader = None
        self._writer = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        item = os.lstat(self.socket_path)
        if (
            not stat.S_ISSOCK(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != self.expected_socket_uid
            or item.st_gid != self.expected_socket_gid
            or stat.S_IMODE(item.st_mode) != 0o660
        ):
            raise ProtocolError("worker_socket_invalid")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(self.socket_path))
        uid, gid = _peer_credentials(connection)
        if uid != self.expected_server_uid or gid != self.expected_server_gid:
            connection.close()
            raise ProtocolError("worker_server_uid_invalid")
        self._socket = connection
        self._reader = connection.makefile("rb", buffering=0)
        self._writer = connection.makefile("wb", buffering=0)

    def close(self) -> None:
        if self._socket is not None:
            try:
                if self._writer is not None:
                    self._writer.close()
                if self._reader is not None:
                    self._reader.close()
                self._socket.close()
            finally:
                self._socket = None
                self._reader = None
                self._writer = None

    def request(self, operation: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation not in _PARAMETER_FIELDS:
            raise ValueError("operation_invalid")
        if self._socket is None:
            self.connect()
        request = {
            "schema": REQUEST_SCHEMA,
            "protocol": PROTOCOL,
            "request_id": uuid.uuid4().hex,
            "lease_id": self.lease_id,
            "operation": operation,
            "parameters": dict(parameters),
        }
        payload = canonical_bytes(request)
        parse_request(payload)
        with self._lock:
            _write_frame(self._writer, payload)
            raw = _read_frame(self._reader)
        if raw is None:
            raise ProtocolError("worker_disconnected")
        try:
            response = json.loads(raw.decode("ascii", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ProtocolError("response_invalid") from exc
        if canonical_bytes(response) != raw or set(response) != {
            "schema", "protocol", "request_id", "lease_id", "operation", "ok", "result"
        }:
            raise ProtocolError("response_invalid")
        if (
            response["schema"] != RESPONSE_SCHEMA
            or response["protocol"] != PROTOCOL
            or response["request_id"] != request["request_id"]
            or response["lease_id"] != self.lease_id
            or response["operation"] != operation
            or type(response["ok"]) is not bool
            or not isinstance(response["result"], Mapping)
        ):
            raise ProtocolError("response_identity_invalid")
        if not response["ok"]:
            raise ProtocolError(str(response["result"].get("error_code", "worker_error")))
        return dict(response["result"])

    def start(self, command: str, *, cwd: Path, timeout_seconds: int, stdin: bytes = b"") -> str:
        result = self.request(
            "exec.start",
            {
                "command": command,
                "cwd": str(cwd),
                "stdin_b64": base64.b64encode(stdin).decode("ascii"),
                "timeout_seconds": timeout_seconds,
            },
        )
        return str(result["session_id"])

    def poll(self, session_id: str, *, wait_milliseconds: int = 100) -> Mapping[str, Any]:
        return self.request(
            "exec.poll",
            {"session_id": session_id, "wait_milliseconds": wait_milliseconds},
        )

    def cancel(self, session_id: str) -> Mapping[str, Any]:
        return self.request("exec.cancel", {"session_id": session_id})


__all__ = [
    "IsolatedWorkerClient",
    "IsolatedWorkerServer",
    "MAX_ACTIVE_CONNECTIONS",
    "MAX_FRAME_BYTES",
    "PROTOCOL",
    "ProtocolError",
    "ReadOnlyBind",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "WorkerPolicy",
    "HOST_READ_ONLY_ROOT",
    "canonical_bytes",
    "canonical_lease_id",
    "parse_request",
]
