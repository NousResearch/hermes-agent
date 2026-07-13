#!/usr/bin/env python3
"""Systemd-notifying bootstrap for the sealed full-canary Discord edge.

The underlying edge remains the only process that can read the Discord bot
credential.  This wrapper adds no operation or routing decision: it hardens the
process, assembles the existing fixed public-egress service, binds its Unix
socket, publishes a secret-free in-process readiness receipt, and then enters
the existing server loop.

Normal startup never creates or repairs the durable journal.  That remains the
separate explicit ``gateway.discord_edge_bootstrap --bootstrap-journal`` gate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import gateway.discord_edge_bootstrap as edge_bootstrap_module
import gateway.discord_edge_runtime as edge_runtime_module
import gateway.discord_edge_service as edge_service_module
import gateway.discord_rest_edge as edge_rest_module
from gateway.canonical_writer_boundary import harden_current_process_against_dumping
from gateway.canonical_writer_readiness import (
    boot_identity,
    module_file_identity,
    notify_systemd_attestation,
    process_start_time_ticks,
    readiness_receipt_sha256,
    write_runtime_attestation,
)
from gateway.discord_edge_bootstrap import (
    DiscordEdgeBootstrap,
    build_service,
    load_service_config,
    serve_service,
)
from gateway.discord_edge_protocol import DiscordPublicTargetType


EDGE_READINESS_SCHEMA = "muncho-discord-edge-readiness-v1"
DEFAULT_EDGE_READINESS_PATH = Path(
    "/run/muncho-discord-egress/runtime-attestation.json"
)
_MAX_CONFIG_BYTES = 64 * 1024
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "DISCORD_BOT_TOKEN",
        "MUNCHO_DISCORD_EDGE_PRIVATE_KEY",
        "MUNCHO_DISCORD_WRITER_PRIVATE_KEY",
    }
)


def _stable_regular_file_sha256(path: Path, *, maximum: int) -> str:
    """Hash one stable non-symlink file without following a replacement."""

    path = Path(path)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
    ):
        raise RuntimeError("Discord edge readiness input is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
        ):
            raise RuntimeError("Discord edge readiness input changed during open")
        while True:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("Discord edge readiness input is oversized")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = path.lstat()
    identity = lambda item: (  # noqa: E731 - compact immutable identity tuple
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(reachable):
        raise RuntimeError("Discord edge readiness input changed during hashing")
    if total != before.st_size:
        raise RuntimeError("Discord edge readiness input length changed")
    return digest.hexdigest()


def _file_provenance(path: Path, *, expected_uid: int, expected_mode: int) -> Mapping[str, Any]:
    """Return non-content credential provenance; never a secret digest."""

    item = Path(path).lstat()
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != expected_uid
        or stat.S_IMODE(item.st_mode) != expected_mode
        or item.st_size <= 0
    ):
        raise RuntimeError("Discord edge credential provenance is invalid")
    return {
        "path": str(path),
        "device": int(item.st_dev),
        "inode": int(item.st_ino),
        "uid": int(item.st_uid),
        "gid": int(item.st_gid),
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": int(item.st_size),
    }


def build_edge_readiness_receipt(
    bootstrap: DiscordEdgeBootstrap,
    *,
    config_path: Path,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Attest the exact live edge from inside its systemd MainPID."""

    if not isinstance(bootstrap, DiscordEdgeBootstrap):
        raise TypeError("Discord edge bootstrap is required")
    if bootstrap.server.fileno() < 0:
        raise RuntimeError("Discord edge listener is not active")
    config = bootstrap.config
    socket_item = config.socket_path.lstat()
    if (
        not stat.S_ISSOCK(socket_item.st_mode)
        or socket_item.st_uid != config.edge_uid
        or socket_item.st_gid != config.edge_gid
        or stat.S_IMODE(socket_item.st_mode) != 0o660
    ):
        raise RuntimeError("Discord edge socket readiness identity is invalid")
    environment_names = sorted(os.environ)
    forbidden = sorted(_FORBIDDEN_ENVIRONMENT_NAMES.intersection(environment_names))
    if forbidden:
        raise RuntimeError("Discord edge received forbidden secret environment")
    boot_sha256, boottime_ns = boot_identity()
    pid = os.getpid()
    observed_at = int(time.time()) if now_unix is None else now_unix
    if type(observed_at) is not int or observed_at < 0:
        raise RuntimeError("Discord edge readiness time is invalid")
    modules: dict[str, Mapping[str, str]] = {}
    module_paths = {
        "readiness_bootstrap": __file__,
        "edge_bootstrap": edge_bootstrap_module.__file__,
        "edge_runtime": edge_runtime_module.__file__,
        "edge_service": edge_service_module.__file__,
        "edge_rest_transport": edge_rest_module.__file__,
    }
    for name, raw_path in sorted(module_paths.items()):
        origin, digest = module_file_identity(raw_path)
        modules[name] = {"origin": origin, "sha256": digest}
    receipt = {
        "version": EDGE_READINESS_SCHEMA,
        "observed_at_unix": observed_at,
        "observed_at_boottime_ns": boottime_ns,
        "boot_id_sha256": boot_sha256,
        "edge_pid": pid,
        "edge_start_time_ticks": process_start_time_ticks(pid),
        "edge_uid": config.edge_uid,
        "edge_gid": config.edge_gid,
        "config_path": str(config_path),
        "config_sha256": _stable_regular_file_sha256(
            Path(config_path), maximum=_MAX_CONFIG_BYTES
        ),
        "socket_path": str(config.socket_path),
        "socket_device": int(socket_item.st_dev),
        "socket_inode": int(socket_item.st_ino),
        "socket_mode": "0660",
        "journal_path": str(config.journal_path),
        "token_file_provenance": _file_provenance(
            config.token_file,
            expected_uid=config.edge_uid,
            expected_mode=0o400,
        ),
        "edge_receipt_private_key_provenance": _file_provenance(
            config.edge_receipt_private_key_file,
            expected_uid=config.edge_uid,
            expected_mode=0o400,
        ),
        "writer_capability_public_key_id": config.writer_capability_public_key_id,
        "edge_receipt_public_key_id": config.edge_receipt_public_key_id,
        "allowed_target_types": sorted(item.value for item in DiscordPublicTargetType),
        "forbidden_target_types": [
            "direct_message",
            "dm",
            "group_dm",
            "private_channel",
            "private_thread",
        ],
        "effective_environment_variable_names": environment_names,
        "modules": modules,
    }
    return receipt


def run_edge(
    config_path: Path,
    *,
    readiness_path: Path = DEFAULT_EDGE_READINESS_PATH,
) -> None:
    """Start, attest, and serve the existing edge; close on every failure."""

    harden_current_process_against_dumping()
    config = load_service_config(config_path)
    bootstrap = build_service(config)
    try:
        bootstrap.server.start()
        receipt = build_edge_readiness_receipt(
            bootstrap,
            config_path=Path(config_path),
        )
        write_runtime_attestation(readiness_path, receipt)
        if not notify_systemd_attestation(
            EDGE_READINESS_SCHEMA,
            readiness_receipt_sha256(receipt),
            ready=True,
        ):
            raise RuntimeError("Discord edge requires systemd Type=notify")
        serve_service(bootstrap)
    except BaseException:
        bootstrap.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="absolute root-owned Discord edge JSON config",
    )
    arguments = parser.parse_args(argv)
    run_edge(Path(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EDGE_READINESS_PATH",
    "EDGE_READINESS_SCHEMA",
    "build_edge_readiness_receipt",
    "main",
    "run_edge",
]
