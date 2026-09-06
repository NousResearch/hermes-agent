#!/usr/bin/env python3
"""Attest a generic Hermes test sandbox, then exec its test command."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import sys


def _mounts() -> dict[str, tuple[set[str], str]]:
    result: dict[str, tuple[set[str], str]] = {}
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        before, after = line.split(" - ", 1)
        fields = before.split()
        target = fields[4].replace("\\040", " ").replace("\\134", "\\")
        result[target] = (set(fields[5].split(",")), after.split()[0])
    return result


def main() -> None:
    if not sys.platform.startswith("linux"):
        raise SystemExit("generic Hermes tests have no attested sandbox on this OS")
    mounts = _mounts()
    failures: list[str] = []
    if Path("/proc/1/comm").read_text(encoding="utf-8").strip() != "bwrap":
        failures.append("PID namespace init is not bubblewrap")
    if socket.if_nameindex() != [(1, "lo")]:
        failures.append("network namespace exposes a non-loopback interface")
    if "ro" not in mounts.get("/", (set(), ""))[0]:
        failures.append("host root is not read-only")
    for path in ("/run", "/tmp", "/var/tmp"):
        if mounts.get(path, (set(), ""))[1] != "tmpfs":
            failures.append(f"{path} is not an isolated tmpfs")
    for raw_path in os.environ.get("HERMES_TEST_MASKED_DIRECTORIES", "").split(
        os.pathsep
    ):
        if (
            raw_path
            and mounts.get(str(Path(raw_path).resolve()), (set(), ""))[1] != "tmpfs"
        ):
            failures.append(f"sensitive directory is not masked: {raw_path}")
    for raw_path in os.environ.get("HERMES_TEST_MASKED_FILES", "").split(os.pathsep):
        if raw_path and str(Path(raw_path).resolve()) not in mounts:
            failures.append(f"sensitive file is not masked: {raw_path}")
    real_home = Path(os.environ["HERMES_TEST_REAL_HOME"]).resolve()
    live_hermes = real_home / ".hermes"
    if live_hermes.exists() and mounts.get(str(live_hermes), (set(), ""))[1] != "tmpfs":
        failures.append("the live Hermes home is not hidden")
    if failures:
        raise SystemExit("Hermes sandbox attestation failed: " + "; ".join(failures))
    if len(sys.argv) < 2:
        raise SystemExit("no test command supplied")
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
