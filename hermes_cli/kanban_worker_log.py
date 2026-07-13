"""Redacting stdout wrapper for kanban worker subprocesses."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


_PRIVATE_KEY_BEGIN_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")
_PRIVATE_KEY_END_RE = re.compile(r"-----END[A-Z ]*PRIVATE KEY-----")


def open_worker_log_file(log_path: Path):
    """Open a worker log for append with owner-only permissions."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.chmod(log_path, 0o600)
    except OSError:
        pass
    return os.fdopen(fd, "ab", buffering=0)


def copy_redacted_worker_log_stream(src, dst) -> None:
    """Copy worker stdout to *dst*, redacting secret-shaped output per line."""
    from agent.redact import redact_terminal_output

    sample_key = "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----"
    redact_private_blocks = redact_terminal_output(sample_key) != sample_key
    inside_private_key = False

    while True:
        raw = src.readline()
        if not raw:
            break
        text = raw.decode("utf-8", errors="replace")
        if redact_private_blocks:
            if inside_private_key:
                if _PRIVATE_KEY_END_RE.search(text):
                    dst.write(b"[REDACTED PRIVATE KEY]\n")
                    inside_private_key = False
                continue
            if (
                _PRIVATE_KEY_BEGIN_RE.search(text)
                and not _PRIVATE_KEY_END_RE.search(text)
            ):
                inside_private_key = True
                continue
        redacted = redact_terminal_output(text)
        dst.write(redacted.encode("utf-8", errors="replace"))

    if inside_private_key:
        dst.write(b"[REDACTED PRIVATE KEY]\n")


def _install_signal_forwarders(proc: subprocess.Popen) -> dict[int, object]:
    previous: dict[int, object] = {}

    def _forward(signum, _frame) -> None:
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _forward)
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        with contextlib.suppress(Exception):
            signal.signal(signum, handler)


def run_worker_with_redacted_log(log_path: Path, command: list[str]) -> int:
    """Run *command*, copying its combined stdout/stderr into a redacted log."""
    try:
        with open_worker_log_file(log_path) as log_f:
            try:
                proc = subprocess.Popen(  # noqa: S603 -- caller supplies argv
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except FileNotFoundError as exc:
                log_f.write(f"Worker launch failed: {exc}\n".encode("utf-8"))
                return 127

            previous_handlers = _install_signal_forwarders(proc)
            try:
                if proc.stdout is not None:
                    copy_redacted_worker_log_stream(proc.stdout, log_f)
                return int(proc.wait())
            finally:
                _restore_signal_handlers(previous_handlers)
                if proc.poll() is None:
                    with contextlib.suppress(Exception):
                        proc.terminate()
                if proc.stdout is not None:
                    with contextlib.suppress(Exception):
                        proc.stdout.close()
    except Exception as exc:
        with contextlib.suppress(Exception):
            sys.stderr.write(f"kanban worker log wrapper failed: {exc}\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.stderr.write("kanban worker log wrapper: missing command\n")
        return 2
    return run_worker_with_redacted_log(Path(args.log_path), command)


if __name__ == "__main__":
    raise SystemExit(main())
