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

EXTERNAL_SUPERVISOR_FLAG = "--external-supervisor"

_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}(?:\s|$)")


def _timestamp() -> str:
    """Match logging.Formatter's default ``%(asctime)s`` timestamp shape."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:23]


def _write_timestamped_line(log_file: TextIO, line: str) -> None:
    rendered = line.rstrip("\r\n")
    prefix = "" if _TIMESTAMP_PREFIX.match(rendered) else f"{_timestamp()} "
    log_file.write(f"{prefix}{rendered}\n")
    log_file.flush()


def _open_log(log_path: Path) -> TextIO:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)


def _copy_stderr_with_timestamps(stderr: BinaryIO, log_path: Path) -> None:
    with _open_log(log_path) as log_file:
        for raw_line in iter(stderr.readline, b""):
            _write_timestamped_line(log_file, raw_line.decode("utf-8", errors="replace"))


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


def _is_hermes_gateway_run_argv(command: Sequence[str]) -> bool:
    """True for Hermes ``gateway run`` argv this wrapper is allowed to upgrade.

    The wrapper is generic. Only historical/current Hermes gateway shapes get ``--external-
    supervisor``; an arbitrary launchd child must not be marked as gateway-supervised (#87005).
    """
    try:
        from gateway.status import looks_like_gateway_command_line
    except Exception:
        return False
    return bool(looks_like_gateway_command_line(" ".join(str(part) for part in command)))


def _prepare_child_command(command: Sequence[str], environ: Mapping[str, str] | None = None) -> list[str]:
    """Return the argv to exec, upgrading stale launchd-wrapped gateway commands.

    launchd stamps ``XPC_SERVICE_NAME=<job label>`` only on this wrapper (its direct child; an
    interactive shell has none, the grandchild sees ``XPC_SERVICE_NAME=0``). Newly generated
    plists put ``--external-supervisor`` on the inner ``gateway run`` so ``hermes update`` can see
    the flag on the live process argv.
    """
    argv = [str(part) for part in command]
    env = os.environ if environ is None else environ
    xpc_service = str(env.get("XPC_SERVICE_NAME", "")).strip()
    if EXTERNAL_SUPERVISOR_FLAG not in argv and xpc_service and xpc_service != "0" and _is_hermes_gateway_run_argv(argv):
        argv.append(EXTERNAL_SUPERVISOR_FLAG)
    return argv


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command and timestamp each stderr line into a log file.")
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
        proc = subprocess.Popen(_prepare_child_command(args.command), stderr=subprocess.PIPE)
    except OSError as exc:
        with _open_log(log_path) as log_file:
            _write_timestamped_line(log_file, f"failed to start stderr-timestamped command: {exc}")
        return 127

    assert proc.stderr is not None
    previous_handlers = _install_signal_forwarders(proc)
    try:
        _copy_stderr_with_timestamps(proc.stderr, log_path)
    finally:
        proc.stderr.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    returncode = proc.wait()
    return 128 + abs(returncode) if returncode < 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
