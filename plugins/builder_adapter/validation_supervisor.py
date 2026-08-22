"""Minimal supervisor executed only as a transient launchd validation job."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _write_result(path: Path, value: dict) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    if len(sys.argv) != 3:
        return 64
    spec_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        argv = spec["argv"]
        cwd = spec["cwd"]
        env = spec["env"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(value, str) and value for value in argv)
            or not isinstance(cwd, str)
            or not isinstance(env, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in env.items()
            )
        ):
            return 65
        started = time.time()
        process = subprocess.Popen(argv, cwd=cwd, env=env)
        status = process.wait()
        _write_result(
            result_path,
            {
                "child_pid": process.pid,
                "exit_status": status,
                "started_at_epoch": started,
                "finished_at_epoch": time.time(),
            },
        )
    except Exception:
        return 70

    # Remain the launchd service leader until the owner removes the unit.
    # Descendants that survived their direct parent therefore remain attached
    # to a live OS-owned service during the mandatory bootout cleanup.
    signal.signal(signal.SIGTERM, lambda *_args: raise_exit())
    signal.signal(signal.SIGINT, lambda *_args: raise_exit())
    while True:
        signal.pause()


def raise_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())
