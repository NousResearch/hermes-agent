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
_UNFINISHED_AUTH_HEADER_RE = re.compile(
    r"(?im)(?<!\S)(?:proxy-)?authorization:[ \t]+(?:bearer|basic|token)[ \t]+\S*$"
)

# Bounded pipe read + text buffer safety margin. 128 chars comfortably
# covers the longest PEM ``BEGIN``/``END`` marker and typical token-shaped
# secrets straddling a chunk boundary.
_READ_CHUNK_BYTES = 65536
_SAFETY_MARGIN = 128


def open_worker_log_file(log_path: Path):
    """Open a worker log for append with owner-only permissions."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.chmod(log_path, 0o600)
    except OSError:
        pass
    return os.fdopen(fd, "ab", buffering=0)


def _read_chunk(src) -> bytes:
    """Read a bounded chunk from *src* without waiting for a newline.

    Uses ``read1`` on buffered pipe readers so a payload with no newline
    still returns whatever bytes the OS has for us instead of blocking
    until a full ``_READ_CHUNK_BYTES`` are available.
    """
    reader = getattr(src, "read1", None)
    if reader is not None:
        return reader(_READ_CHUNK_BYTES)
    return src.read(_READ_CHUNK_BYTES)


def copy_redacted_worker_log_stream(src, dst) -> None:
    """Copy worker stdout to *dst* in bounded chunks with secret redaction.

    Buffers decoded text across reads so token-shaped secrets and PEM
    ``BEGIN/END PRIVATE KEY`` blocks are redacted even when split across
    chunk boundaries.
    """
    from agent.redact import redact_terminal_output

    sample_key = "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----"
    redact_private_blocks = redact_terminal_output(sample_key, force=True) != sample_key
    inside_private_key = False
    buffer = ""

    def _emit(text: str) -> None:
        if not text:
            return
        redacted = redact_terminal_output(text, force=True) if redact_private_blocks else text
        dst.write(redacted.encode("utf-8", errors="replace"))

    def _unfinished_secret_start(text: str) -> int | None:
        """Return the start of a secret whose delimiter is in a later read."""
        # Prefix patterns are intentionally unbounded. If a match reaches the
        # current end, retaining only a fixed tail would emit its body raw on
        # the next iteration. The auth-header case has the same failure mode
        # but does not require a vendor prefix.
        from agent.redact import _PREFIX_RE

        starts = [
            match.start(1)
            for match in _PREFIX_RE.finditer(text)
            if match.end(1) == len(text)
        ]
        starts.extend(
            match.start() for match in _UNFINISHED_AUTH_HEADER_RE.finditer(text)
        )
        return min(starts) if starts else None

    while True:
        raw = _read_chunk(src)
        if not raw:
            break
        buffer += raw.decode("utf-8", errors="replace")

        # Drain as much of the buffer as we can before requesting more input.
        while True:
            if redact_private_blocks and inside_private_key:
                end = _PRIVATE_KEY_END_RE.search(buffer)
                if end is None:
                    # END has not arrived yet; suppress everything.
                    buffer = ""
                    break
                dst.write(b"[REDACTED PRIVATE KEY]\n")
                nl = buffer.find("\n", end.end())
                buffer = buffer[nl + 1:] if nl != -1 else buffer[end.end():]
                inside_private_key = False
                continue

            if redact_private_blocks:
                begin = _PRIVATE_KEY_BEGIN_RE.search(buffer)
                if begin is not None:
                    line_start = buffer.rfind("\n", 0, begin.start()) + 1
                    _emit(buffer[:line_start])
                    end = _PRIVATE_KEY_END_RE.search(buffer, begin.end())
                    if end is not None:
                        dst.write(b"[REDACTED PRIVATE KEY]\n")
                        nl = buffer.find("\n", end.end())
                        buffer = buffer[nl + 1:] if nl != -1 else buffer[end.end():]
                        continue
                    inside_private_key = True
                    buffer = ""
                    break

            # No pending block: emit everything except a safety tail so a
            # secret or ``BEGIN`` marker straddling the next chunk still
            # gets redacted intact.
            if len(buffer) > _SAFETY_MARGIN:
                safe_end = len(buffer) - _SAFETY_MARGIN
                unfinished_start = _unfinished_secret_start(buffer)
                if unfinished_start is not None:
                    safe_end = min(safe_end, unfinished_start)
                emittable = buffer[:safe_end]
                if not emittable:
                    break
                _emit(emittable)
                buffer = buffer[safe_end:]
            break

    if inside_private_key:
        dst.write(b"[REDACTED PRIVATE KEY]\n")
    elif buffer:
        _emit(buffer)


def _kill_proc_group(proc: subprocess.Popen, signum: int) -> None:
    """Signal the child's whole process group, falling back to the child."""
    # os.killpg is POSIX-only; on Windows we can only signal the child.
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), signum)  # windows-footgun: ok — POSIX branch guarded above
            return
    with contextlib.suppress(Exception):
        proc.send_signal(signum)


def _install_signal_forwarders(proc: subprocess.Popen) -> dict[int, object]:
    previous: dict[int, object] = {}
    # Guard against re-entry when killpg-to-group also delivers the signal
    # back to us — one forward per signal is enough.
    forwarded: set[int] = set()

    def _forward(signum, _frame) -> None:
        if signum in forwarded:
            return
        forwarded.add(signum)
        if proc.poll() is None:
            _kill_proc_group(proc, signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _forward)
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        with contextlib.suppress(Exception):
            signal.signal(signum, handler)


def _lead_process_group() -> None:
    """Best-effort: make this wrapper a process-group leader on POSIX.

    Lets the dispatcher signal the wrapper + its child together by
    sending to the negative wrapper pid, so SIGKILL to the wrapper still
    reaches the child.
    """
    # os.setpgrp is POSIX-only; Windows has no process-group concept here.
    if hasattr(os, "setpgrp"):
        with contextlib.suppress(OSError):
            os.setpgrp()


def run_worker_with_redacted_log(log_path: Path, command: list[str]) -> int:
    """Run *command*, copying its combined stdout/stderr into a redacted log."""
    _lead_process_group()
    try:
        with open_worker_log_file(log_path) as log_f:
            try:
                proc = subprocess.Popen(  # noqa: S603 -- caller supplies argv
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=False,
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
                    _kill_proc_group(proc, signal.SIGTERM)
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
