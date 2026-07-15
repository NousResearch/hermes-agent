"""Config-only, socket-activated entrypoint for the isolated worker.

The service accepts no policy overrides on the command line and never creates
its own listener.  A privileged service manager seals the canonical JSON
configuration and passes exactly one AF_UNIX listening socket as descriptor 3.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import signal
import socket
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.isolated_worker import (
    PROTOCOL,
    IsolatedWorkerServer,
    ReadOnlyBind,
    WorkerPolicy,
    canonical_bytes,
)


CONFIG_SCHEMA = "muncho.isolated-worker.config.v2"
SOCKET_ACTIVATION_FD = 3
MAX_CONFIG_BYTES = 256 * 1024
_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "config_sha256",
        "listener_path",
        "expected_peer_uid",
        "expected_peer_gid",
        "socket_uid",
        "socket_gid",
        "lease_base",
        "lease_uid",
        "lease_gid",
        "network_isolated",
        "bwrap",
        "shell",
        "limits",
        "read_only_binds",
    }
)
_EXECUTABLE_FIELDS = frozenset({"path", "sha256", "uid"})
_LIMIT_FIELDS = frozenset(
    {
        "maximum_timeout_seconds",
        "maximum_output_bytes",
        "maximum_active_leases",
        "maximum_active_jobs_per_lease",
        "lease_ttl_seconds",
        "lease_quota_bytes",
        "lease_quota_entries",
        "global_quota_bytes",
        "global_quota_entries",
    }
)
_READ_ONLY_BIND_FIELDS = frozenset(
    {"source", "destination", "source_uid", "source_gid"}
)


class ServiceConfigError(RuntimeError):
    """Stable fail-closed service configuration error."""


@dataclass(frozen=True)
class ServiceConfig:
    listener_path: Path
    policy: WorkerPolicy


def _reject_constant(_value: str) -> None:
    raise ServiceConfigError("config_json_invalid")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ServiceConfigError("config_json_duplicate_key")
        result[key] = value
    return result


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ServiceConfigError(f"{label}_fields_not_exact")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ServiceConfigError(f"{label}_invalid")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ServiceConfigError(f"{label}_invalid")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ServiceConfigError(f"{label}_invalid")
    return path


def _read_sealed_config(
    path: Path,
    *,
    expected_owner_uid: int,
    expected_owner_gid: int,
) -> bytes:
    path = Path(path)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ServiceConfigError("config_path_invalid")
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != expected_owner_uid
        or before.st_gid != expected_owner_gid
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o440
        or not 0 < before.st_size <= MAX_CONFIG_BYTES
    ):
        raise ServiceConfigError("config_file_not_sealed")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = MAX_CONFIG_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(before) != identity(after)
        or len(payload) != before.st_size
    ):
        raise ServiceConfigError("config_changed_during_read")
    return payload


def parse_service_config(payload: bytes) -> ServiceConfig:
    """Parse exact canonical JSON and verify its self-digest."""

    if not payload or len(payload) > MAX_CONFIG_BYTES or b"\n" in payload:
        raise ServiceConfigError("config_frame_invalid")
    try:
        value = json.loads(
            payload.decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ServiceConfigError("config_json_invalid") from exc
    if canonical_bytes(value) != payload:
        raise ServiceConfigError("config_not_canonical")
    raw = _exact_mapping(value, _CONFIG_FIELDS, "config")
    if raw["schema"] != CONFIG_SCHEMA or raw["protocol"] != PROTOCOL:
        raise ServiceConfigError("config_identity_invalid")
    supplied_digest = raw["config_sha256"]
    if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
        raise ServiceConfigError("config_digest_invalid")
    unsigned = dict(raw)
    unsigned.pop("config_sha256")
    expected_digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise ServiceConfigError("config_digest_mismatch")

    bwrap = _exact_mapping(raw["bwrap"], _EXECUTABLE_FIELDS, "bwrap")
    shell = _exact_mapping(raw["shell"], _EXECUTABLE_FIELDS, "shell")
    limits = _exact_mapping(raw["limits"], _LIMIT_FIELDS, "limits")
    binds_value = raw["read_only_binds"]
    if not isinstance(binds_value, list):
        raise ServiceConfigError("read_only_binds_invalid")
    binds: list[ReadOnlyBind] = []
    for value in binds_value:
        item = _exact_mapping(value, _READ_ONLY_BIND_FIELDS, "read_only_bind")
        try:
            binds.append(
                ReadOnlyBind(
                    source=_absolute_path(item["source"], "read_only_bind_source"),
                    destination=_absolute_path(
                        item["destination"], "read_only_bind_destination"
                    ),
                    source_uid=_nonnegative_integer(
                        item["source_uid"], "read_only_bind_source_uid"
                    ),
                    source_gid=_nonnegative_integer(
                        item["source_gid"], "read_only_bind_source_gid"
                    ),
                )
            )
        except (OSError, ValueError) as exc:
            raise ServiceConfigError(
                str(exc) or "read_only_bind_invalid"
            ) from exc
    if type(raw["network_isolated"]) is not bool:
        raise ServiceConfigError("network_isolated_invalid")
    try:
        policy = WorkerPolicy(
            expected_peer_uid=_nonnegative_integer(
                raw["expected_peer_uid"], "expected_peer_uid"
            ),
            expected_peer_gid=_nonnegative_integer(
                raw["expected_peer_gid"], "expected_peer_gid"
            ),
            socket_uid=_nonnegative_integer(raw["socket_uid"], "socket_uid"),
            socket_gid=_nonnegative_integer(raw["socket_gid"], "socket_gid"),
            lease_base=_absolute_path(raw["lease_base"], "lease_base"),
            lease_uid=_nonnegative_integer(raw["lease_uid"], "lease_uid"),
            lease_gid=_nonnegative_integer(raw["lease_gid"], "lease_gid"),
            network_isolated=raw["network_isolated"],
            bwrap_path=_absolute_path(bwrap["path"], "bwrap_path"),
            bwrap_sha256=str(bwrap["sha256"]),
            bwrap_uid=_nonnegative_integer(bwrap["uid"], "bwrap_uid"),
            shell=_absolute_path(shell["path"], "shell_path"),
            shell_sha256=str(shell["sha256"]),
            shell_uid=_nonnegative_integer(shell["uid"], "shell_uid"),
            maximum_timeout_seconds=_nonnegative_integer(
                limits["maximum_timeout_seconds"], "maximum_timeout_seconds"
            ),
            maximum_output_bytes=_nonnegative_integer(
                limits["maximum_output_bytes"], "maximum_output_bytes"
            ),
            maximum_active_leases=_nonnegative_integer(
                limits["maximum_active_leases"], "maximum_active_leases"
            ),
            maximum_active_jobs_per_lease=_nonnegative_integer(
                limits["maximum_active_jobs_per_lease"],
                "maximum_active_jobs_per_lease",
            ),
            lease_ttl_seconds=_nonnegative_integer(
                limits["lease_ttl_seconds"], "lease_ttl_seconds"
            ),
            lease_quota_bytes=_nonnegative_integer(
                limits["lease_quota_bytes"], "lease_quota_bytes"
            ),
            lease_quota_entries=_nonnegative_integer(
                limits["lease_quota_entries"], "lease_quota_entries"
            ),
            global_quota_bytes=_nonnegative_integer(
                limits["global_quota_bytes"], "global_quota_bytes"
            ),
            global_quota_entries=_nonnegative_integer(
                limits["global_quota_entries"], "global_quota_entries"
            ),
            read_only_binds=tuple(binds),
        )
    except (OSError, ValueError) as exc:
        raise ServiceConfigError(str(exc) or "worker_policy_invalid") from exc
    return ServiceConfig(
        listener_path=_absolute_path(raw["listener_path"], "listener_path"),
        policy=policy,
    )


def load_service_config(
    path: Path,
    *,
    expected_owner_uid: int = 0,
    expected_owner_gid: int | None = None,
) -> ServiceConfig:
    owner_gid = os.getegid() if expected_owner_gid is None else expected_owner_gid
    if (
        type(expected_owner_uid) is not int
        or expected_owner_uid < 0
        or type(owner_gid) is not int
        or owner_gid < 0
    ):
        raise ServiceConfigError("config_owner_identity_invalid")
    return parse_service_config(
        _read_sealed_config(
            Path(path),
            expected_owner_uid=expected_owner_uid,
            expected_owner_gid=owner_gid,
        )
    )


def activated_listener(
    config: ServiceConfig,
    *,
    environment: Mapping[str, str] | None = None,
    process_id: int | None = None,
) -> socket.socket:
    """Take ownership of the one systemd-style socket activation descriptor."""

    source = os.environ if environment is None else environment
    pid = os.getpid() if process_id is None else process_id
    if (
        source.get("LISTEN_PID") != str(pid)
        or source.get("LISTEN_FDS") != "1"
        or source.get("LISTEN_FDNAMES") != "isolated-worker"
    ):
        raise ServiceConfigError("socket_activation_identity_invalid")
    try:
        listener = socket.socket(fileno=SOCKET_ACTIVATION_FD)
    except OSError as exc:
        raise ServiceConfigError("socket_activation_fd_invalid") from exc
    try:
        if (
            listener.family != socket.AF_UNIX
            or listener.type & socket.SOCK_STREAM != socket.SOCK_STREAM
            or listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1
            or Path(listener.getsockname()) != config.listener_path
        ):
            raise ServiceConfigError("socket_activation_listener_invalid")
    except BaseException:
        listener.detach()
        raise
    return listener


def run_service(config: ServiceConfig, listener: socket.socket) -> None:
    stop = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop.set()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    server = IsolatedWorkerServer(config.policy)
    try:
        server.serve(listener, stop)
    finally:
        server.close()
        listener.close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-isolated-worker")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_service_config(args.config)
    listener = activated_listener(config)
    run_service(config, listener)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by service manager
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "ServiceConfig",
    "ServiceConfigError",
    "activated_listener",
    "load_service_config",
    "main",
    "parse_service_config",
    "run_service",
]
