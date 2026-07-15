"""Fail-closed in-process readiness proof for the Canonical writer boundary.

The gateway performs this mechanical PING from its exact systemd MainPID
before it starts MCP discovery, platform adapters, cron, or any model-controlled
child.  The resulting receipt contains no credential material and is bound to
the current process identity and installed module bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
)
from gateway.canonical_writer_protocol import (
    MAX_SEQUENCE,
    CanonicalWriterOperation,
)


READINESS_RECEIPT_VERSION = "canonical-writer-readiness-v1"
WRITER_LIVENESS_RECEIPT_VERSION = "canonical-writer-liveness-v1"
DEFAULT_READINESS_RECEIPT_PATH = Path(
    "/run/hermes-cloud-gateway/canonical-writer-readiness.json"
)
DEFAULT_WRITER_LIVENESS_RECEIPT_PATH = Path(
    "/run/hermes-cloud-gateway/canonical-writer-liveness.json"
)
_MAX_RECEIPT_BYTES = 256 * 1024
_EXPECTED_PING_FIELDS = frozenset(
    {"request_id", "service", "protocol", "database_identity"}
)


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise RuntimeError("gateway readiness requires POSIX UID support")
    return int(getter())


def _process_start_time_ticks(pid: int) -> int:
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("gateway readiness PID is invalid")
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    suffix = raw.rsplit(")", 1)
    if len(suffix) != 2:
        raise RuntimeError("gateway process identity is unavailable")
    fields = suffix[1].strip().split()
    try:
        value = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("gateway process start time is unavailable") from exc
    if value <= 0:
        raise RuntimeError("gateway process start time is invalid")
    return value


def boot_identity() -> tuple[str, int]:
    raw_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    try:
        parsed = uuid.UUID(raw_boot_id)
    except ValueError as exc:
        raise RuntimeError("runtime boot identity is invalid") from exc
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is None:
        raise RuntimeError("runtime boottime clock is unavailable")
    boottime_ns = time.clock_gettime_ns(clock_id)
    if boottime_ns < 0:
        raise RuntimeError("runtime boottime clock is invalid")
    return hashlib.sha256(str(parsed).encode("ascii")).hexdigest(), boottime_ns


def _module_identity() -> tuple[str, str]:
    return module_file_identity(__file__)


def _absolute_runtime_path(value: Any) -> str | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not raw
        or not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or any(character in raw for character in ("\x00", "\n", "\r"))
    ):
        return None
    return raw


def current_python_runtime_identity() -> Mapping[str, Any]:
    """Capture the calling interpreter's complete path/module origin view."""

    import_paths: set[str] = set()
    unexpected_paths: set[str] = set()
    for value in sys.path:
        normalized = _absolute_runtime_path(value)
        if normalized is None:
            unexpected_paths.add(str(value))
        else:
            import_paths.add(normalized)

    module_origins: set[str] = set()
    unexpected_origins: set[str] = set()
    for module in tuple(sys.modules.values()):
        candidates: list[Any] = [getattr(module, "__file__", None)]
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            candidates.append(getattr(spec, "origin", None))
        for candidate in candidates:
            if candidate in {None, "built-in", "frozen"}:
                continue
            normalized = _absolute_runtime_path(candidate)
            if normalized is None:
                unexpected_origins.add(str(candidate))
            else:
                module_origins.add(normalized)
    environment = dict(os.environ)
    environment_names = sorted(environment)
    environment_value_sha256 = {
        name: hashlib.sha256(
            environment[name].encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        for name in environment_names
    }
    return {
        "effective_import_paths": sorted(import_paths),
        "unexpected_import_paths": sorted(unexpected_paths),
        "loaded_module_origins": sorted(module_origins),
        "unexpected_import_origins": sorted(unexpected_origins),
        "loaded_module_origins_complete": True,
        "effective_environment_variable_names": environment_names,
        "effective_environment_variable_value_sha256": environment_value_sha256,
    }


def _current_process_hardening_state() -> tuple[bool, int, int]:
    from gateway.canonical_writer_boundary import current_process_hardening_state

    return current_process_hardening_state()


def _remove_stale_receipt(path: Path) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode):
        raise RuntimeError("gateway readiness receipt path is a directory")
    path.unlink()


def remove_canonical_writer_liveness_receipt(
    path: Path = DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
) -> None:
    """Remove liveness evidence before/after a failed sentinel generation."""

    _remove_stale_receipt(Path(path))


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("gateway readiness receipt path must be absolute")
    parent = path.parent
    parent_stat = parent.stat()
    if not stat.S_ISDIR(parent_stat.st_mode) or parent.is_symlink():
        raise RuntimeError("gateway readiness runtime directory is invalid")
    encoded = (
        json.dumps(
            dict(receipt),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise RuntimeError("gateway readiness receipt is too large")
    temporary = parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags | nofollow, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("gateway readiness receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        observed = temporary.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_uid != _effective_uid()
        ):
            raise RuntimeError("gateway readiness receipt staging is unsafe")
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def readiness_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(receipt),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def notify_systemd_writer_readiness(
    receipt: Mapping[str, Any],
    *,
    ready: bool,
    _notify_socket: str | None = None,
) -> bool:
    """Publish a digest-bound status from the exact main process.

    NOTIFY_SOCKET is systemd's mechanical process contract, not Hermes
    configuration. The collector requires Type=notify and NotifyAccess=main
    before treating this status as activation evidence.
    """

    return notify_systemd_attestation(
        READINESS_RECEIPT_VERSION,
        readiness_receipt_sha256(receipt),
        ready=ready,
        _notify_socket=_notify_socket,
    )


def writer_liveness_status_text(
    startup_readiness_sha256: str,
    generation: int,
    liveness_receipt_sha256: str,
) -> str:
    """Return the one exact systemd status accepted as live authority.

    ``NotifyAccess=main`` makes systemd authenticate the sender as the unit's
    MainPID.  These digest-only fields bind that process to its immutable
    startup proof and to the newest atomically persisted writer PING receipt.
    """

    if (
        not isinstance(startup_readiness_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", startup_readiness_sha256) is None
    ):
        raise ValueError("startup readiness digest is invalid")
    if type(generation) is not int or not 1 <= generation <= MAX_SEQUENCE:
        raise ValueError("canonical writer liveness generation is invalid")
    if (
        not isinstance(liveness_receipt_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", liveness_receipt_sha256) is None
    ):
        raise ValueError("canonical writer liveness digest is invalid")
    return (
        f"{WRITER_LIVENESS_RECEIPT_VERSION}:"
        f"{startup_readiness_sha256}:{generation}:{liveness_receipt_sha256}"
    )


def notify_systemd_writer_liveness(
    startup_readiness_sha256: str,
    receipt: Mapping[str, Any],
    *,
    _notify_socket: str | None = None,
) -> bool:
    """Authenticate a persisted liveness generation as the systemd MainPID."""

    if receipt.get("version") != WRITER_LIVENESS_RECEIPT_VERSION:
        raise ValueError("canonical writer liveness receipt version is invalid")
    status_text = writer_liveness_status_text(
        startup_readiness_sha256,
        receipt.get("generation"),
        readiness_receipt_sha256(receipt),
    )
    return _notify_systemd_status(
        status_text,
        ready=False,
        _notify_socket=_notify_socket,
    )


def notify_systemd_attestation(
    schema: str,
    digest_sha256: str,
    *,
    ready: bool,
    _notify_socket: str | None = None,
) -> bool:
    """Send one bounded digest-only systemd status from the calling process."""

    if type(ready) is not bool:
        raise TypeError("systemd readiness state must be boolean")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", schema):
        raise ValueError("systemd attestation schema is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", digest_sha256):
        raise ValueError("systemd attestation digest is invalid")
    return _notify_systemd_status(
        f"{schema}:{digest_sha256}",
        ready=ready,
        _notify_socket=_notify_socket,
    )


def _notify_systemd_status(
    status_text: str,
    *,
    ready: bool,
    _notify_socket: str | None,
) -> bool:
    """Send an exact bounded status over systemd's inherited notify socket."""

    if type(ready) is not bool:
        raise TypeError("systemd readiness state must be boolean")
    try:
        encoded_status = status_text.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ValueError("systemd attestation status is invalid") from exc
    if (
        not encoded_status
        or len(encoded_status) > 512
        or any(character in status_text for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("systemd attestation status is invalid")
    raw_address = (
        os.environ.get("NOTIFY_SOCKET", "")
        if _notify_socket is None
        else _notify_socket
    )
    if not raw_address:
        return False
    if (
        not isinstance(raw_address, str)
        or len(raw_address.encode("utf-8")) > 100
        or any(character in raw_address for character in ("\x00", "\n", "\r"))
        or raw_address[0] not in {"/", "@"}
    ):
        raise RuntimeError("systemd notify socket is invalid")
    address = "\x00" + raw_address[1:] if raw_address.startswith("@") else raw_address
    fields = [f"STATUS={status_text}"]
    if ready:
        fields.insert(0, "READY=1")
    payload = ("\n".join(fields) + "\n").encode("utf-8")
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    channel.set_inheritable(False)
    try:
        sent = channel.sendto(payload, address)
    finally:
        channel.close()
    if sent != len(payload):
        raise RuntimeError("systemd readiness notification was incomplete")
    return True


def process_start_time_ticks(pid: int) -> int:
    return _process_start_time_ticks(pid)


def module_file_identity(path: str | os.PathLike[str]) -> tuple[str, str]:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise RuntimeError("runtime module origin must be absolute")
    supplied_stat = supplied.lstat()
    if (
        stat.S_ISLNK(supplied_stat.st_mode)
        or not stat.S_ISREG(supplied_stat.st_mode)
        or supplied_stat.st_nlink != 1
    ):
        raise RuntimeError("runtime module origin is invalid")
    origin = supplied.resolve(strict=True)
    if origin != supplied:
        raise RuntimeError("runtime module origin is not normalized")
    return str(origin), hashlib.sha256(origin.read_bytes()).hexdigest()


def write_runtime_attestation(
    path: str | os.PathLike[str],
    receipt: Mapping[str, Any],
) -> None:
    _write_receipt(Path(path), receipt)


def _validate_ping_response(response: Mapping[str, Any]) -> str:
    if (
        set(response) != _EXPECTED_PING_FIELDS
        or response.get("service") != "canonical_writer"
        or response.get("protocol") != "v1"
        or response.get("database_identity") != CANONICAL_WRITER_MIGRATION_OWNER
    ):
        raise RuntimeError("canonical writer readiness response is invalid")
    request_id = str(response.get("request_id") or "")
    try:
        parsed_request_id = uuid.UUID(request_id)
    except ValueError as exc:
        raise RuntimeError(
            "canonical writer readiness request ID is invalid"
        ) from exc
    if parsed_request_id.int == 0 or str(parsed_request_id) != request_id:
        raise RuntimeError("canonical writer readiness request ID is invalid")
    return request_id


def _writer_socket_identity(path: Path) -> Mapping[str, Any]:
    if not path.is_absolute():
        raise RuntimeError("canonical writer liveness socket path is invalid")
    observed = path.lstat()
    if (
        not stat.S_ISSOCK(observed.st_mode)
        or observed.st_dev <= 0
        or observed.st_ino <= 0
        or observed.st_uid < 0
        or observed.st_gid < 0
        or stat.S_IMODE(observed.st_mode) != 0o660
    ):
        raise RuntimeError("canonical writer liveness socket identity is invalid")
    return {
        "socket_path": str(path),
        "socket_device": observed.st_dev,
        "socket_inode": observed.st_ino,
        "socket_owner_uid": observed.st_uid,
        "socket_group_gid": observed.st_gid,
        "socket_mode": "0660",
    }


def attest_canonical_writer_liveness(
    generation: int,
    *,
    receipt_path: Path = DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
    _writer_call: Callable[..., Mapping[str, Any]] | None = None,
    _now_unix: Callable[[], float] = time.time,
    _pid: int | None = None,
    _boot_identity_provider: Callable[[], tuple[str, int]] = boot_identity,
    _process_start_time: Callable[[int], int] = _process_start_time_ticks,
    _socket_identity_provider: Callable[[Path], Mapping[str, Any]] = (
        _writer_socket_identity
    ),
    _deadline_monotonic_ns: int | None = None,
    _monotonic_ns: Callable[[], int] = time.monotonic_ns,
    _persist: bool = True,
) -> Mapping[str, Any]:
    """PING the writer and atomically publish one process-bound generation.

    The receipt is deliberately separate from the immutable startup readiness
    receipt.  The sentinel atomically persists it and only then publishes a
    digest-bound systemd ``StatusText`` from the exact MainPID.  A deadline is
    checked after the blocking PING and immediately before persistence, so a
    timed-out worker can never publish late evidence after the service has
    already failed closed.
    """

    if type(generation) is not int or not 1 <= generation <= MAX_SEQUENCE:
        raise ValueError("canonical writer liveness generation is invalid")
    if type(_persist) is not bool:
        raise TypeError("canonical writer liveness persistence flag is invalid")
    if _deadline_monotonic_ns is not None and (
        type(_deadline_monotonic_ns) is not int or _deadline_monotonic_ns <= 0
    ):
        raise ValueError("canonical writer liveness deadline is invalid")

    from gateway.canonical_writer_boundary import (
        canonical_writer_call,
        frozen_writer_boundary_config,
    )

    config = frozen_writer_boundary_config()
    if not config.enabled:
        raise RuntimeError("canonical writer liveness boundary is disabled")
    path = Path(receipt_path)
    if _persist:
        _remove_stale_receipt(path)
    before_socket = dict(_socket_identity_provider(Path(config.socket_path)))
    expected_socket_keys = {
        "socket_path",
        "socket_device",
        "socket_inode",
        "socket_owner_uid",
        "socket_group_gid",
        "socket_mode",
    }
    if set(before_socket) != expected_socket_keys:
        raise RuntimeError("canonical writer liveness socket identity is incomplete")

    call = canonical_writer_call if _writer_call is None else _writer_call
    response = dict(call(CanonicalWriterOperation.PING, {}))
    request_id = _validate_ping_response(response)
    if (
        _deadline_monotonic_ns is not None
        and _monotonic_ns() > _deadline_monotonic_ns
    ):
        raise TimeoutError("canonical writer liveness PING exceeded its deadline")
    after_socket = dict(_socket_identity_provider(Path(config.socket_path)))
    if after_socket != before_socket:
        raise RuntimeError("canonical writer liveness socket changed during PING")

    pid = os.getpid() if _pid is None else _pid
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("canonical writer liveness PID is invalid")
    start_time_ticks = _process_start_time(pid)
    boot_id_sha256, boottime_ns = _boot_identity_provider()
    observed_at_unix = int(_now_unix())
    if (
        not re.fullmatch(r"[0-9a-f]{64}", boot_id_sha256)
        or type(boottime_ns) is not int
        or boottime_ns < 0
        or observed_at_unix < 0
    ):
        raise RuntimeError("canonical writer liveness clock identity is invalid")
    receipt = {
        "version": WRITER_LIVENESS_RECEIPT_VERSION,
        "generation": generation,
        "observed_at_unix": observed_at_unix,
        "observed_at_boottime_ns": boottime_ns,
        "boot_id_sha256": boot_id_sha256,
        "gateway_pid": pid,
        "gateway_start_time_ticks": start_time_ticks,
        "writer_request_id": request_id,
        "writer_service": response["service"],
        "writer_protocol": response["protocol"],
        "database_identity": response["database_identity"],
        **before_socket,
    }
    if (
        _deadline_monotonic_ns is not None
        and _monotonic_ns() > _deadline_monotonic_ns
    ):
        raise TimeoutError("canonical writer liveness receipt exceeded its deadline")
    if _persist:
        write_canonical_writer_liveness_receipt(receipt, path=path)
    return receipt


def write_canonical_writer_liveness_receipt(
    receipt: Mapping[str, Any],
    *,
    path: Path = DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
) -> None:
    """Validate and atomically persist a completed liveness generation."""

    value = dict(receipt)
    expected_keys = {
        "version",
        "generation",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "boot_id_sha256",
        "gateway_pid",
        "gateway_start_time_ticks",
        "writer_request_id",
        "writer_service",
        "writer_protocol",
        "database_identity",
        "socket_path",
        "socket_device",
        "socket_inode",
        "socket_owner_uid",
        "socket_group_gid",
        "socket_mode",
    }
    socket_path = value.get("socket_path")
    try:
        request_id = uuid.UUID(str(value.get("writer_request_id") or ""))
    except ValueError as exc:
        raise RuntimeError("canonical writer liveness receipt is invalid") from exc
    if (
        set(value) != expected_keys
        or value.get("version") != WRITER_LIVENESS_RECEIPT_VERSION
        or type(value.get("generation")) is not int
        or not 1 <= value["generation"] <= MAX_SEQUENCE
        or type(value.get("observed_at_unix")) is not int
        or value["observed_at_unix"] < 0
        or type(value.get("observed_at_boottime_ns")) is not int
        or value["observed_at_boottime_ns"] < 0
        or not isinstance(value.get("boot_id_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["boot_id_sha256"]) is None
        or type(value.get("gateway_pid")) is not int
        or value["gateway_pid"] <= 1
        or type(value.get("gateway_start_time_ticks")) is not int
        or value["gateway_start_time_ticks"] <= 0
        or request_id.int == 0
        or str(request_id) != value.get("writer_request_id")
        or value.get("writer_service") != "canonical_writer"
        or value.get("writer_protocol") != "v1"
        or value.get("database_identity") != CANONICAL_WRITER_MIGRATION_OWNER
        or not isinstance(socket_path, str)
        or _absolute_runtime_path(socket_path) != socket_path
        or type(value.get("socket_device")) is not int
        or value["socket_device"] <= 0
        or type(value.get("socket_inode")) is not int
        or value["socket_inode"] <= 0
        or type(value.get("socket_owner_uid")) is not int
        or value["socket_owner_uid"] < 0
        or type(value.get("socket_group_gid")) is not int
        or value["socket_group_gid"] < 0
        or value.get("socket_mode") != "0660"
    ):
        raise RuntimeError("canonical writer liveness receipt is invalid")
    _write_receipt(Path(path), value)


def attest_canonical_writer_startup_readiness(
    *,
    receipt_path: Path = DEFAULT_READINESS_RECEIPT_PATH,
    _writer_call: Callable[..., Mapping[str, Any]] | None = None,
    _now_unix: Callable[[], float] = time.time,
    _pid: int | None = None,
    _boot_identity_provider: Callable[[], tuple[str, int]] = boot_identity,
    _process_start_time: Callable[[int], int] = _process_start_time_ticks,
    _module_identity_provider: Callable[[], tuple[str, str]] = _module_identity,
    _process_hardening_provider: Callable[[], tuple[bool, int, int]] = (
        _current_process_hardening_state
    ),
    _python_runtime_provider: Callable[[], Mapping[str, Any]] = (
        current_python_runtime_identity
    ),
) -> Mapping[str, Any] | None:
    """PING the enabled boundary and atomically persist a process-bound receipt.

    A disabled boundary is an exact no-op.  An enabled boundary raises on every
    transport, identity, response, or receipt failure so gateway startup cannot
    become ready with an unavailable Canonical source of truth.
    """

    from gateway.canonical_writer_boundary import (
        canonical_writer_call,
        frozen_writer_boundary_config,
    )

    config = frozen_writer_boundary_config()
    if not config.enabled:
        return None
    dumpable, core_soft, core_hard = _process_hardening_provider()
    if dumpable is not False or core_soft != 0 or core_hard != 0:
        raise RuntimeError("gateway process hardening attestation is invalid")
    python_runtime = dict(_python_runtime_provider())
    if set(python_runtime) != {
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "loaded_module_origins_complete",
        "effective_environment_variable_names",
        "effective_environment_variable_value_sha256",
    } or python_runtime.get("loaded_module_origins_complete") is not True:
        raise RuntimeError("gateway Python runtime attestation is incomplete")
    for name in (
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "effective_environment_variable_names",
    ):
        values = python_runtime.get(name)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError("gateway Python runtime attestation is invalid")
    environment_names = python_runtime["effective_environment_variable_names"]
    environment_value_sha256 = python_runtime.get(
        "effective_environment_variable_value_sha256"
    )
    if (
        not isinstance(environment_value_sha256, dict)
        or list(environment_value_sha256) != environment_names
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in environment_value_sha256.values()
        )
    ):
        raise RuntimeError("gateway Python environment attestation is invalid")
    call = canonical_writer_call if _writer_call is None else _writer_call
    path = Path(receipt_path)
    _remove_stale_receipt(path)
    response = dict(call(CanonicalWriterOperation.PING, {}))
    request_id = _validate_ping_response(response)
    pid = os.getpid() if _pid is None else _pid
    start_time_ticks = _process_start_time(pid)
    boot_id_sha256, boottime_ns = _boot_identity_provider()
    module_origin, module_sha256 = _module_identity_provider()
    observed_at_unix = int(_now_unix())
    if observed_at_unix < 0:
        raise RuntimeError("canonical writer readiness clock is invalid")
    receipt = {
        "version": READINESS_RECEIPT_VERSION,
        "observed_at_unix": observed_at_unix,
        "observed_at_boottime_ns": boottime_ns,
        "boot_id_sha256": boot_id_sha256,
        "gateway_pid": pid,
        "gateway_start_time_ticks": start_time_ticks,
        "writer_request_id": request_id,
        "writer_service": response["service"],
        "writer_protocol": response["protocol"],
        "database_identity": response["database_identity"],
        "gateway_module_origin": module_origin,
        "gateway_module_sha256": module_sha256,
        "gateway_dumpable": dumpable,
        "gateway_core_soft_limit": core_soft,
        "gateway_core_hard_limit": core_hard,
        **python_runtime,
    }
    _write_receipt(path, receipt)
    return receipt


__all__ = [
    "DEFAULT_READINESS_RECEIPT_PATH",
    "DEFAULT_WRITER_LIVENESS_RECEIPT_PATH",
    "READINESS_RECEIPT_VERSION",
    "WRITER_LIVENESS_RECEIPT_VERSION",
    "attest_canonical_writer_liveness",
    "attest_canonical_writer_startup_readiness",
    "boot_identity",
    "current_python_runtime_identity",
    "module_file_identity",
    "notify_systemd_attestation",
    "notify_systemd_writer_liveness",
    "notify_systemd_writer_readiness",
    "process_start_time_ticks",
    "readiness_receipt_sha256",
    "remove_canonical_writer_liveness_receipt",
    "write_canonical_writer_liveness_receipt",
    "write_runtime_attestation",
    "writer_liveness_status_text",
]
