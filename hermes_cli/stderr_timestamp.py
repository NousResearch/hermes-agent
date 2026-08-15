"""Run a child process while prefixing each stderr line with a timestamp."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Sequence, TextIO

from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV


_TIMESTAMP_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}(?:\s|$)"
)


def _timestamp() -> str:
    """Match logging.Formatter's default ``%(asctime)s`` timestamp shape."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:23]


def _write_timestamped_line(log_file: TextIO, line: str) -> None:
    rendered = line.rstrip("\r\n")
    prefix = "" if _TIMESTAMP_PREFIX.match(rendered) else f"{_timestamp()} "
    log_file.write(f"{prefix}{rendered}\n")
    log_file.flush()


def _copy_stderr_with_timestamps(stderr: BinaryIO, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        for raw_line in iter(stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace")
            _write_timestamped_line(log_file, line)


def _command_exit_code(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def _install_signal_forwarders(proc: subprocess.Popen[bytes]) -> dict[int, object]:
    def _forward(signum: int, _frame: object) -> None:
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous: dict[int, object] = {}
    for signum in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if signum is not None:
            try:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, _forward)
            except (OSError, RuntimeError, ValueError):
                previous.pop(signum, None)
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _child_env_for_command(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Preserve launchd supervision across this one-hop wrapper.

    launchd stamps ``XPC_SERVICE_NAME=<job label>`` only on its *direct*
    child. This module is that child when the generated plist wraps
    ``gateway run`` for timestamped stderr. The grandchild then sees
    ``XPC_SERVICE_NAME=0`` and ``_guard_supervised_gateway_conflict``
    treats the service's own spawn as a foreign supervised gateway
    (#86893). Forward the existing opt-in marker so the grandchild
    still recognizes itself as the supervised process.
    """
    env = os.environ if environ is None else environ
    xpc_service = str(env.get("XPC_SERVICE_NAME", "")).strip()
    if not xpc_service or xpc_service == "0":
        return None
    child_env = dict(env)
    child_env[EXTERNAL_GATEWAY_SUPERVISOR_ENV] = "1"
    return child_env


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and timestamp each stderr line into a log file."
    )
    parser.add_argument("--error-log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    log_path: Path = args.error_log

    try:
        proc = subprocess.Popen(
            args.command,
            stderr=subprocess.PIPE,
            env=_child_env_for_command(),
        )
    except OSError as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            _write_timestamped_line(
                log_file,
                f"failed to start stderr-timestamped command: {exc}",
            )
        return 127

    assert proc.stderr is not None
    previous_handlers = _install_signal_forwarders(proc)
    try:
        _copy_stderr_with_timestamps(proc.stderr, log_path)
    finally:
        proc.stderr.close()
        _restore_signal_handlers(previous_handlers)
    return _command_exit_code(proc.wait())


if __name__ == "__main__":
    sys.exit(main())
