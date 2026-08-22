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
from typing import BinaryIO, Protocol, Sequence

EXTERNAL_SUPERVISOR_FLAG = "--external-supervisor"


class _LineWriter(Protocol):
    """Any object with a line ``write`` + ``flush`` (file or rotating writer)."""

    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...

# Same defaults as hermes_logging.setup_logging when config.yaml doesn't set
# logging.max_size_mb / logging.backup_count (config_defaults.py).
_DEFAULT_MAX_SIZE_MB = 5
_DEFAULT_BACKUP_COUNT = 3


def _rotation_config() -> tuple[int, int]:
    """Best-effort ``(max_size_mb, backup_count)`` for the stderr capture log.

    Delegates to hermes_logging's canonical config reader so gateway.error.log
    rotates under the SAME policy as agent.log / errors.log / gui.log (shared
    defaults 5 MiB / 3, overridable via logging.max_size_mb / backup_count).
    Any read failure falls back to the shared defaults.
    """
    max_mb = _DEFAULT_MAX_SIZE_MB
    backups = _DEFAULT_BACKUP_COUNT
    try:
        from hermes_logging import _read_logging_config

        _, cfg_max, cfg_backups = _read_logging_config()
        if cfg_max is not None:
            max_mb = int(cfg_max)
        if cfg_backups is not None:
            backups = int(cfg_backups)
    except Exception as exc:  # pragma: no cover - defensive
        # Signal rather than silently reverting to hardcoded defaults: if
        # hermes_logging renames its reader, rotation still works but no
        # longer tracks logging.max_size_mb / backup_count.
        print(f"stderr_timestamp: using default rotation sizes ({exc})", file=sys.stderr)
    return max(max_mb, 1), max(backups, 0)


_TIMESTAMP_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}(?:\s|$)"
)


def _timestamp() -> str:
    """Match logging.Formatter's default ``%(asctime)s`` timestamp shape."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:23]


def _write_timestamped_line(log_file: _LineWriter, line: str) -> None:
    rendered = line.rstrip("\r\n")
    prefix = "" if _TIMESTAMP_PREFIX.match(rendered) else f"{_timestamp()} "
    log_file.write(f"{prefix}{rendered}\n")
    log_file.flush()


class _RotatingWriter:
    """Size-based rotating writer for the launchd stderr capture log.

    ``gateway.error.log`` is written by the launchd ``StandardErrorPath`` /
    ``hermes_cli.stderr_timestamp`` wrapper and previously grew unbounded (a
    Slack Socket Mode reconnect flap filled one host's file to 141 MB). This
    mirrors stdlib ``RotatingFileHandler`` semantics — split the file at
    ``max_size_mb``, keep ``backup_count`` rotated copies, oldest first —
    without pulling the logging framework into a lightweight stderr wrapper.

    Config comes from ``logging.max_size_mb`` / ``logging.backup_count``
    (same keys the rest of Hermes logging honors); defaults 5 MiB / 3.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        max_mb, backup_count = _rotation_config()
        self._max_bytes = max_mb * 1024 * 1024
        self._backup_count = backup_count
        self._io: BinaryIO | None = None

    def _open(self) -> BinaryIO:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Recompute config each open so an admin changing logging.max_size_mb
        # while the gateway runs is honored on the next rotation.
        max_mb, backup_count = _rotation_config()
        self._max_bytes = max_mb * 1024 * 1024
        self._backup_count = backup_count
        self._io = self._path.open("ab", buffering=0)
        assert self._io is not None
        return self._io

    def _rotate(self) -> None:
        """Rename current file to .1, shift older backups up, start a fresh one."""
        self.close()
        for index in range(self._backup_count, 0, -1):
            target = self._path.with_name(f"{self._path.name}.{index}")
            source = self._path.with_name(f"{self._path.name}.{index - 1}") if index > 1 else self._path
            if source.exists():
                target.unlink(missing_ok=True)
                try:
                    source.rename(target)
                except OSError:
                    # Ignore races from another process; a fresh file is created
                    # below and the rotation retries on the next write.
                    pass
        self._open()

    def write(self, text: str) -> None:
        # A single line longer than max_bytes still lands whole (rotation only
        # happens BEFORE writing, and only when the file is non-empty): an
        # oversize line may exceed the cap by its own length. That trades a
        # bounded overshoot for never truncating a diagnostic mid-line.
        if self._io is None:
            self._open()
        assert self._io is not None
        data = text.encode("utf-8", errors="replace")
        if self._io.tell() + len(data) > self._max_bytes and self._io.tell() > 0:
            self._rotate()
        self._io.write(data)

    def flush(self) -> None:
        """No-op — writes are unbuffered (BinaryIO opened with buffering=0)."""

    def close(self) -> None:
        """Close the underlying handle. Rotated .N backups intentionally
        survive; this never deletes anything."""
        if self._io is not None:
            self._io.close()
            self._io = None


def _copy_stderr_with_timestamps(stderr: BinaryIO, log_path: Path) -> None:
    writer = _RotatingWriter(log_path)
    try:
        for raw_line in iter(stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace")
            _write_timestamped_line(writer, line)
    finally:
        writer.close()


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


def _is_launchd_supervised(environ: Mapping[str, str] | None = None) -> bool:
    """True when this process is launchd's direct child (not an interactive shell)."""
    env = os.environ if environ is None else environ
    xpc_service = str(env.get("XPC_SERVICE_NAME", "")).strip()
    return bool(xpc_service and xpc_service != "0")


def _is_hermes_gateway_run_argv(command: Sequence[str]) -> bool:
    """True for Hermes ``gateway run`` argv this wrapper is allowed to upgrade.

    The wrapper is generic. Only historical/current Hermes gateway shapes
    get ``--external-supervisor``; an arbitrary launchd child must not be
    marked as gateway-supervised (#87005).
    """
    try:
        from gateway.status import looks_like_gateway_command_line
    except Exception:
        return False
    return bool(looks_like_gateway_command_line(" ".join(str(part) for part in command)))


def _with_external_supervisor_flag(command: Sequence[str]) -> list[str]:
    argv = [str(part) for part in command]
    if EXTERNAL_SUPERVISOR_FLAG not in argv:
        argv.append(EXTERNAL_SUPERVISOR_FLAG)
    return argv


def _prepare_child_command(
    command: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the argv to exec, upgrading stale launchd-wrapped gateway commands.

    launchd stamps ``XPC_SERVICE_NAME=<job label>`` only on this wrapper.
    The grandchild sees ``XPC_SERVICE_NAME=0``. Newly generated plists put
    ``--external-supervisor`` on the inner ``gateway run`` so ``hermes update``
    can see the flag on the live process argv. Stale plists still wrap the
    historical ``gateway run --replace`` shape without that flag; append it
    here, and only for that shape.
    """
    argv = [str(part) for part in command]
    if not _is_launchd_supervised(environ):
        return argv
    if not _is_hermes_gateway_run_argv(argv):
        return argv
    return _with_external_supervisor_flag(argv)


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
            _prepare_child_command(args.command),
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        writer = _RotatingWriter(log_path)
        try:
            _write_timestamped_line(
                writer,
                f"failed to start stderr-timestamped command: {exc}",
            )
        finally:
            writer.close()
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
