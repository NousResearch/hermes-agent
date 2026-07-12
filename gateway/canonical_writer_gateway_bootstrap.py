#!/usr/bin/env python3
"""Minimal credential-free gateway for the writer-only Cloud canary.

This entry point is intentionally separate from :mod:`gateway.run`.  It owns
only the mechanical writer PING/readiness contract and never imports platform
adapters, plugins, providers, models, MCP, cron, dotenv, or the general gateway
runner.  The process is synchronous: the writer client's socket deadlines are
the only blocking-operation bounds, so no worker process or thread can outlive
failed liveness evidence.
"""

from __future__ import annotations

import fcntl
import os
import signal
import stat
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Mapping

import yaml

from gateway.canonical_writer_boundary import (
    close_canonical_writer_clients,
    frozen_writer_boundary_config,
    harden_gateway_process_for_writer_boundary,
)
from gateway.canonical_writer_readiness import (
    attest_canonical_writer_liveness,
    attest_canonical_writer_startup_readiness,
    module_file_identity,
    notify_systemd_writer_liveness,
    notify_systemd_writer_readiness,
    process_start_time_ticks,
    readiness_receipt_sha256,
    remove_canonical_writer_liveness_receipt,
    write_canonical_writer_liveness_receipt,
)


DEFAULT_MANAGED_CONFIG_PATH = Path("/etc/hermes/config.yaml")
DEFAULT_GATEWAY_RUNTIME_DIR = Path("/run/hermes-cloud-gateway")
_LOCK_FILENAME = "gateway.lock"
_PID_FILENAME = "gateway.pid"
_MAX_CONFIG_BYTES = 32 * 1024
_LIVENESS_INTERVAL_SECONDS = 1.0
_MAX_LIVENESS_DEADLINE_SECONDS = 30.0
_POSIX_ACL_NAMES = frozenset({"system.posix_acl_access", "system.posix_acl_default"})


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise RuntimeError("writer-only gateway requires POSIX UID support")
    return int(getter())


def _effective_gid() -> int:
    getter = getattr(os, "getegid", None)
    if not callable(getter):
        raise RuntimeError("writer-only gateway requires POSIX GID support")
    return int(getter())


def _require_linux_secure_open_flags(platform: str) -> None:
    if platform != "linux":
        raise RuntimeError("writer-only gateway requires Linux")
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        if type(getattr(os, name, None)) is not int or getattr(os, name) <= 0:
            raise RuntimeError(f"writer-only gateway requires {name}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
    )


def _reject_posix_acl(target: int | Path, *, label: str) -> None:
    try:
        names = os.listxattr(target)
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"{label} ACL state is unavailable") from exc
    normalized = {
        name.decode("utf-8", errors="strict") if isinstance(name, bytes) else name
        for name in names
    }
    if normalized & _POSIX_ACL_NAMES:
        raise RuntimeError(f"{label} has a POSIX ACL")


class _StrictConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases and duplicate/non-string keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            raise ValueError("managed writer policy cannot contain YAML aliases")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ValueError("managed writer policy mapping is invalid")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or not key or key in result:
                raise ValueError(
                    "managed writer policy contains a duplicate or invalid key"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_StrictConfigLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictConfigLoader.construct_mapping,
)


def _validate_writer_only_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "canonical_brain",
        "plugins",
        "cron",
    }:
        raise RuntimeError("managed writer-only policy root is not exact")
    canonical = value.get("canonical_brain")
    plugins = value.get("plugins")
    cron = value.get("cron")
    if (
        not isinstance(canonical, Mapping)
        or set(canonical) != {"writer_boundary", "discord_edge"}
        or not isinstance(canonical.get("writer_boundary"), Mapping)
        or dict(canonical["writer_boundary"]) != {"enabled": True}
        or not isinstance(canonical.get("discord_edge"), Mapping)
        or dict(canonical["discord_edge"]) != {"enabled": False}
        or not isinstance(plugins, Mapping)
        or set(plugins) != {"enabled", "disabled"}
        or plugins.get("enabled") != []
        or plugins.get("disabled") != []
        or not isinstance(cron, Mapping)
        or dict(cron) != {"provider": "builtin"}
    ):
        raise RuntimeError("managed writer-only policy is not exact")
    return value


def load_strict_managed_writer_only_policy(
    path: Path = DEFAULT_MANAGED_CONFIG_PATH,
    *,
    _expected_uid: int = 0,
    _expected_gid: int = 0,
) -> Mapping[str, Any]:
    """Read one stable, root-controlled, duplicate-free managed policy."""

    path = Path(path)
    if not path.is_absolute() or path.name != "config.yaml":
        raise ValueError("managed writer policy path is invalid")
    parent = path.parent
    parent_before = os.lstat(parent)
    if (
        stat.S_ISLNK(parent_before.st_mode)
        or not stat.S_ISDIR(parent_before.st_mode)
        or parent_before.st_uid != _expected_uid
        or parent_before.st_gid != _expected_gid
        or stat.S_IMODE(parent_before.st_mode) & 0o022
    ):
        raise RuntimeError("managed writer policy directory is not trusted")

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(parent, directory_flags)
    try:
        parent_fd_stat = os.fstat(directory_fd)
        if _file_identity(parent_fd_stat) != _file_identity(parent_before):
            raise RuntimeError("managed writer policy directory changed")
        _reject_posix_acl(directory_fd, label="managed writer policy directory")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != _expected_uid
                or before.st_gid != _expected_gid
                or stat.S_IMODE(before.st_mode) & 0o022
                or not 0 < before.st_size <= _MAX_CONFIG_BYTES
            ):
                raise RuntimeError("managed writer policy file is not trusted")
            _reject_posix_acl(descriptor, label="managed writer policy file")
            chunks: list[bytes] = []
            remaining = _MAX_CONFIG_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(8192, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == 0 and os.read(descriptor, 1):
                raise RuntimeError("managed writer policy is oversized")
            after = os.fstat(descriptor)
            reachable = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                _file_identity(after) != _file_identity(before)
                or _file_identity(reachable) != _file_identity(before)
                or sum(len(chunk) for chunk in chunks) != before.st_size
            ):
                raise RuntimeError("managed writer policy changed during read")
        finally:
            os.close(descriptor)
        parent_after = os.lstat(parent)
        if _file_identity(parent_after) != _file_identity(
            parent_before
        ) or _file_identity(os.fstat(directory_fd)) != _file_identity(parent_before):
            raise RuntimeError("managed writer policy directory changed during read")
    finally:
        os.close(directory_fd)

    try:
        text = b"".join(chunks).decode("utf-8", errors="strict")
        loaded = yaml.load(text, Loader=_StrictConfigLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise RuntimeError("managed writer policy YAML is invalid") from exc
    return _validate_writer_only_policy(loaded)


class _RuntimeLease:
    def __init__(
        self,
        *,
        directory_fd: int,
        lock_fd: int,
        pid_payload: bytes,
        pid_identity: tuple[int, int, int, int, int, int],
    ) -> None:
        self.directory_fd = directory_fd
        self.lock_fd = lock_fd
        self.pid_payload = pid_payload
        self.pid_identity = pid_identity

    def close(self) -> None:
        try:
            try:
                observed = os.stat(
                    _PID_FILENAME,
                    dir_fd=self.directory_fd,
                    follow_symlinks=False,
                )
                descriptor = os.open(
                    _PID_FILENAME,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=self.directory_fd,
                )
                try:
                    payload = os.read(descriptor, 256)
                    identity = _file_identity(os.fstat(descriptor))
                finally:
                    os.close(descriptor)
                if (
                    _file_identity(observed) == self.pid_identity == identity
                    and payload == self.pid_payload
                ):
                    os.unlink(_PID_FILENAME, dir_fd=self.directory_fd)
                    os.fsync(self.directory_fd)
            except FileNotFoundError:
                pass
        finally:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)
                os.close(self.directory_fd)


def _acquire_runtime_lease(
    runtime_dir: Path,
    *,
    pid: int,
    process_start_ticks: int,
) -> _RuntimeLease:
    runtime_dir = Path(runtime_dir)
    if not runtime_dir.is_absolute():
        raise ValueError("writer-only gateway runtime path must be absolute")
    uid = _effective_uid()
    gid = _effective_gid()
    before = os.lstat(runtime_dir)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != uid
        or before.st_gid != gid
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise RuntimeError("writer-only gateway runtime directory is invalid")
    directory_fd = os.open(
        runtime_dir,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    lock_fd = -1
    pid_installed = False
    pid_payload = b""
    try:
        if _file_identity(os.fstat(directory_fd)) != _file_identity(before):
            raise RuntimeError("writer-only gateway runtime directory changed")
        _reject_posix_acl(directory_fd, label="writer-only gateway runtime directory")
        lock_fd = os.open(
            _LOCK_FILENAME,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        lock_stat = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_uid != uid
            or lock_stat.st_gid != gid
            or stat.S_IMODE(lock_stat.st_mode) != 0o600
        ):
            raise RuntimeError("writer-only gateway lock file is invalid")
        _reject_posix_acl(lock_fd, label="writer-only gateway lock file")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("writer-only gateway runtime lock is held") from exc
        reachable_lock = os.stat(
            _LOCK_FILENAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if _file_identity(reachable_lock) != _file_identity(lock_stat):
            raise RuntimeError("writer-only gateway lock identity changed")

        pid_payload = f"{pid}:{process_start_ticks}\n".encode("ascii")
        temporary_name = f".{_PID_FILENAME}.{pid}.{process_start_ticks}.tmp"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                offset = 0
                while offset < len(pid_payload):
                    written = os.write(temporary_fd, pid_payload[offset:])
                    if written <= 0:
                        raise OSError("writer-only gateway PID write made no progress")
                    offset += written
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)
            os.replace(
                temporary_name,
                _PID_FILENAME,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            pid_installed = True
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
        pid_stat = os.stat(
            _PID_FILENAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(pid_stat.st_mode)
            or pid_stat.st_nlink != 1
            or pid_stat.st_uid != uid
            or pid_stat.st_gid != gid
            or stat.S_IMODE(pid_stat.st_mode) != 0o600
        ):
            raise RuntimeError("writer-only gateway PID file is invalid")
        return _RuntimeLease(
            directory_fd=directory_fd,
            lock_fd=lock_fd,
            pid_payload=pid_payload,
            pid_identity=_file_identity(pid_stat),
        )
    except BaseException:
        if pid_installed:
            try:
                observed = os.stat(
                    _PID_FILENAME,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(observed.st_mode)
                    and observed.st_nlink == 1
                    and observed.st_uid == uid
                    and observed.st_gid == gid
                    and stat.S_IMODE(observed.st_mode) == 0o600
                ):
                    descriptor = os.open(
                        _PID_FILENAME,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        current = os.read(descriptor, 256)
                    finally:
                        os.close(descriptor)
                    if current == pid_payload:
                        os.unlink(_PID_FILENAME, dir_fd=directory_fd)
                        os.fsync(directory_fd)
            except (FileNotFoundError, OSError):
                pass
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
        os.close(directory_fd)
        raise


def _entry_module_identity() -> tuple[str, str]:
    return module_file_identity(__file__)


def run_writer_only_gateway(
    *,
    runtime_dir: Path = DEFAULT_GATEWAY_RUNTIME_DIR,
    liveness_interval_seconds: float = _LIVENESS_INTERVAL_SECONDS,
    _policy_loader: Callable[[], Mapping[str, Any]] = (
        load_strict_managed_writer_only_policy
    ),
    _hardener: Callable[[Mapping[str, Any]], bool] = (
        harden_gateway_process_for_writer_boundary
    ),
    _config_provider: Callable[[Mapping[str, Any]], Any] = (
        frozen_writer_boundary_config
    ),
    _startup_attestor: Callable[..., Mapping[str, Any] | None] = (
        attest_canonical_writer_startup_readiness
    ),
    _startup_notifier: Callable[..., bool] = notify_systemd_writer_readiness,
    _liveness_attestor: Callable[..., Mapping[str, Any]] = (
        attest_canonical_writer_liveness
    ),
    _liveness_writer: Callable[..., None] = (write_canonical_writer_liveness_receipt),
    _liveness_notifier: Callable[..., bool] = notify_systemd_writer_liveness,
    _liveness_remover: Callable[..., None] = (remove_canonical_writer_liveness_receipt),
    _process_start_provider: Callable[[int], int] = process_start_time_ticks,
    _stop_event: threading.Event | None = None,
    _install_signal_handlers: bool = True,
    _platform: str = sys.platform,
) -> bool:
    """Run the exact writer-only sentinel until signalled or first failure."""

    if not 0 < liveness_interval_seconds <= 60:
        raise ValueError("writer-only liveness interval is invalid")
    if type(_install_signal_handlers) is not bool:
        raise TypeError("writer-only signal installation flag is invalid")
    _require_linux_secure_open_flags(_platform)
    policy = _policy_loader()
    if not _hardener(policy):
        raise RuntimeError("writer-only gateway hardening was not activated")
    config = _config_provider(policy)
    if (
        config.enabled is not True
        or config.discord_edge_enabled is not False
        or config.model_tools_enabled is not False
    ):
        raise RuntimeError("writer-only gateway frozen policy is not isolated")

    pid = os.getpid()
    start_ticks = _process_start_provider(pid)
    if (
        type(pid) is not int
        or pid <= 1
        or type(start_ticks) is not int
        or start_ticks <= 0
    ):
        raise RuntimeError("writer-only gateway process identity is invalid")
    lease = _acquire_runtime_lease(
        runtime_dir,
        pid=pid,
        process_start_ticks=start_ticks,
    )
    stop_event = threading.Event() if _stop_event is None else _stop_event
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_stop(
        _received_signal: int,
        _frame: FrameType | None,
    ) -> None:
        stop_event.set()

    try:
        if _install_signal_handlers:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("writer-only gateway must own main-thread signals")
            for received_signal in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[received_signal] = signal.getsignal(received_signal)
                signal.signal(received_signal, request_stop)

        _liveness_remover()
        initial_receipt = _startup_attestor(
            _module_identity_provider=_entry_module_identity,
        )
        if initial_receipt is None or not _startup_notifier(
            initial_receipt,
            ready=False,
        ):
            raise RuntimeError("writer-only gateway pre-readiness PING failed")
        final_receipt = _startup_attestor(
            _module_identity_provider=_entry_module_identity,
        )
        if final_receipt is None or not _startup_notifier(
            final_receipt,
            ready=True,
        ):
            raise RuntimeError("writer-only gateway final readiness PING failed")
        startup_sha256 = readiness_receipt_sha256(final_receipt)

        generation = 1
        while not stop_event.is_set():
            deadline = time.monotonic_ns() + int(
                _MAX_LIVENESS_DEADLINE_SECONDS * 1_000_000_000
            )
            receipt = _liveness_attestor(
                generation,
                _deadline_monotonic_ns=deadline,
                _persist=False,
            )
            if stop_event.is_set():
                break
            if receipt.get("generation") != generation:
                raise RuntimeError("writer-only liveness generation is invalid")
            _liveness_writer(receipt)
            if not _liveness_notifier(startup_sha256, receipt):
                raise RuntimeError("writer-only liveness systemd notify failed")
            generation += 1
            stop_event.wait(liveness_interval_seconds)
        return True
    finally:
        try:
            _liveness_remover()
        finally:
            for received_signal, previous in previous_handlers.items():
                signal.signal(received_signal, previous)
            close_canonical_writer_clients()
            lease.close()


def main() -> int:
    if len(sys.argv) != 1:
        print("writer-only gateway accepts no command-line arguments", file=sys.stderr)
        return 2
    try:
        return 0 if run_writer_only_gateway() else 1
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print("writer-only gateway failed closed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GATEWAY_RUNTIME_DIR",
    "DEFAULT_MANAGED_CONFIG_PATH",
    "load_strict_managed_writer_only_policy",
    "main",
    "run_writer_only_gateway",
]
