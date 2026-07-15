#!/usr/bin/env python3
"""Isolated, digest-attested launcher for the production cron cutover edge."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from pathlib import Path


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE = re.compile(r"^hermes-agent-[0-9a-f]{12}$")
_RUNTIME_RELATIVE = Path("gateway/production_cron_cutover_runtime.py")


def _pop_exact_flag(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1:
        raise RuntimeError("production_cron_cutover_launcher_identity_invalid")
    index = positions[0]
    if index + 1 >= len(argv):
        raise RuntimeError("production_cron_cutover_launcher_identity_invalid")
    value = argv[index + 1]
    del argv[index : index + 2]
    if _SHA256.fullmatch(value) is None:
        raise RuntimeError("production_cron_cutover_launcher_identity_invalid")
    return value


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(
                "production_cron_cutover_launcher_identity_invalid"
            )
        value = b""
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            value += chunk
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("production_cron_cutover_launcher_identity_changed")
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    argv = list(sys.argv[1:])
    expected_entrypoint = _pop_exact_flag(
        argv, "--expected-entrypoint-sha256"
    )
    expected_runtime = _pop_exact_flag(argv, "--expected-runtime-sha256")
    entrypoint = Path(__file__).resolve(strict=True)
    release_root = entrypoint.parents[2]
    runtime = release_root / _RUNTIME_RELATIVE
    if (
        _RELEASE.fullmatch(release_root.name) is None
        or _sha256_file(entrypoint) != expected_entrypoint
        or _sha256_file(runtime) != expected_runtime
    ):
        raise RuntimeError("production_cron_cutover_launcher_identity_drifted")
    sys.path.insert(0, str(release_root))
    from gateway.production_cron_cutover_runtime import main as runtime_main

    return runtime_main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
