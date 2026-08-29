"""Run a child process while prefixing each stderr line with a timestamp."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Protocol, Sequence

EXTERNAL_SUPERVISOR_FLAG = "--external-supervisor"
_DEFAULT_MAX_SIZE_MB = 5
_DEFAULT_BACKUP_COUNT = 3


class _LineWriter(Protocol):
    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...


def _rotation_config() -> tuple[int, int]:
    """Return the canonical Hermes file-log size and retention settings."""
    try:
        from hermes_logging import _read_logging_config

        _, configured_size, configured_backups = _read_logging_config()
    except Exception:
        configured_size = configured_backups = None
    return (
        max(int(configured_size or _DEFAULT_MAX_SIZE_MB), 1),
        max(int(configured_backups or _DEFAULT_BACKUP_COUNT), 0),
    )

_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}(?:\s|$)")


def _timestamp() -> str:
    """Match logging.Formatter's default ``%(asctime)s`` timestamp shape."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:23]


def _write_timestamped_line(log_file: _LineWriter, line: str) -> None:
    rendered = line.rstrip("\r\n")
    prefix = "" if _TIMESTAMP_PREFIX.match(rendered) else f"{_timestamp()} "
    log_file.write(f"{prefix}{rendered}\n")
    log_file.flush()


class _RotatingWriter:
    """Owner-only, size-bounded writer for launchd-captured output.

    Rotation happens between writes, matching ``RotatingFileHandler``: one
    indivisible line can exceed the configured size, but retained history is
    still bounded by that line plus the configured live/backup file count.
    """

    def __init__(
        self,
        log_path: Path,
        *,
        max_bytes: int | None = None,
        backup_count: int | None = None,
    ) -> None:
        configured_mb, configured_backups = _rotation_config()
        self._path = log_path
        self._max_bytes = max(max_bytes or configured_mb * 1024 * 1024, 1)
        self._backup_count = (
            configured_backups if backup_count is None else max(backup_count, 0)
        )
        self._io: BinaryIO | None = None
        self._prune_and_secure_backups()
        self._open()

    def _prune_and_secure_backups(self) -> None:
        """Apply a reduced retention setting immediately, not one rollover later."""
        if not self._path.parent.exists():
            return
        prefix = f"{self._path.name}."
        for candidate in self._path.parent.iterdir():
            suffix = candidate.name.removeprefix(prefix)
            if candidate.name == f"{prefix}{suffix}" and suffix.isdigit():
                try:
                    if int(suffix) > self._backup_count:
                        candidate.unlink()
                    elif candidate.is_file() and not candidate.is_symlink():
                        candidate.chmod(0o600)
                except FileNotFoundError:
                    pass

    def _open(self) -> BinaryIO:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self._path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:  # pragma: no cover - Windows compatibility for unit tests
                os.chmod(self._path, 0o600)
            self._io = os.fdopen(fd, "ab", buffering=0)
        except BaseException:
            os.close(fd)
            raise
        return self._io

    def _rotate(self) -> None:
        self.close()
        if self._backup_count == 0:
            self._path.unlink(missing_ok=True)
        else:
            for index in range(self._backup_count, 0, -1):
                target = self._path.with_name(f"{self._path.name}.{index}")
                source = (
                    self._path.with_name(f"{self._path.name}.{index - 1}")
                    if index > 1
                    else self._path
                )
                if source.exists():
                    target.unlink(missing_ok=True)
                    source.replace(target)
                    if not target.is_symlink():
                        target.chmod(0o600)
        self._open()

    def write(self, text: str) -> None:
        data = text.encode("utf-8", errors="replace")
        stream = self._io or self._open()
        if stream.tell() and stream.tell() + len(data) > self._max_bytes:
            self._rotate()
            assert self._io is not None
            stream = self._io
        stream.write(data)

    def flush(self) -> None:
        """Writes are unbuffered."""

    def close(self) -> None:
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


def _copy_stdout(stdout: BinaryIO, log_path: Path) -> None:
    writer = _RotatingWriter(log_path)
    try:
        for raw_line in iter(stdout.readline, b""):
            writer.write(raw_line.decode("utf-8", errors="replace"))
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
    parser = argparse.ArgumentParser(description="Run a command and capture its output in rotating log files.")
    parser.add_argument("--output-log", type=Path)
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
    output_path: Path | None = args.output_log

    try:
        proc = subprocess.Popen(
            _prepare_child_command(args.command),
            stdout=subprocess.PIPE if output_path else None,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        writer = _RotatingWriter(log_path)
        try:
            _write_timestamped_line(writer, f"failed to start stderr-timestamped command: {exc}")
        finally:
            writer.close()
        return 127

    assert proc.stderr is not None
    stdout_thread = None
    if output_path is not None:
        assert proc.stdout is not None
        stdout_thread = threading.Thread(
            target=_copy_stdout,
            args=(proc.stdout, output_path),
            name="gateway-stdout-capture",
        )
        stdout_thread.start()
    previous_handlers = _install_signal_forwarders(proc)
    try:
        _copy_stderr_with_timestamps(proc.stderr, log_path)
    finally:
        proc.stderr.close()
        if stdout_thread is not None:
            stdout_thread.join()
        if proc.stdout is not None:
            proc.stdout.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return _command_exit_code(proc.wait())


if __name__ == "__main__":
    sys.exit(main())
