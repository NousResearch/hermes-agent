"""Contain one session-bridge backend process and kill it if the worker dies."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

from hermes_cli._subprocess_compat import kill_process_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        return 64
    child = subprocess.Popen(
        command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
        start_new_session=True,
    )
    stopping = threading.Event()

    def stop(*_unused) -> None:
        if stopping.is_set():
            return
        stopping.set()
        kill_process_tree(child)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while child.poll() is None:
        if os.getppid() != args.parent_pid:
            stop()
            break
        time.sleep(0.1)
    if child.poll() is None:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    return int(child.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
