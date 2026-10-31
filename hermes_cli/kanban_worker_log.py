"""Kanban worker log helpers: per-task log rotation and the redacting stdout
wrapper a worker subprocess runs under (``python -m hermes_cli.kanban_worker_log``)."""

from __future__ import annotations

# PM's bare interpreter must select its committed dependencies before registration imports.
import hermes_bootstrap

import argparse
import codecs
import contextlib
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    from hermes_cli.kanban_db_dispatch import _positive_int
    max_bytes = _positive_int(kanban_cfg.get("worker_log_rotate_bytes"), DEFAULT_LOG_ROTATE_BYTES, minimum=1)
    backup_count = _positive_int(kanban_cfg.get("worker_log_backup_count"), DEFAULT_LOG_BACKUP_COUNT, minimum=0)
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        from hermes_cli.kanban_db_dispatch import _positive_int
        backup_count = _positive_int(backup_count, DEFAULT_LOG_BACKUP_COUNT, minimum=0)
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            dst = _rotated_log_path(log_path, generation + 1)
            with contextlib.suppress(OSError):
                src.rename(dst)
                dst.chmod(0o600)
        first = _rotated_log_path(log_path, 1)
        log_path.rename(first)
        with contextlib.suppress(OSError):
            first.chmod(0o600)
    except OSError:
        pass


_PRIVATE_KEY_BEGIN_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")
_PRIVATE_KEY_END_RE = re.compile(r"-----END[A-Z ]*PRIVATE KEY-----")
_READ_CHUNK_BYTES = 65536


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
    """Read a bounded chunk without waiting for a newline."""
    reader = getattr(src, "read1", None)
    return reader(_READ_CHUNK_BYTES) if reader is not None else src.read(_READ_CHUNK_BYTES)


def _worker_log_record_continues(text: str) -> bool:
    """Keep quoted values, header whitespace and incomplete control-split tokens together."""
    from agent.redact import (
        _AUTH_HEADER_RE, _CONTROL_CHARS_RE, _PREFIX_RE, _PREFIX_SUBSTRINGS,
        _SECRET_HEADER_RE, _TOKEN_BODY_CHARS,
    )

    # Synthetic completions expose canonical headers awaiting a colon or credential.
    # Inspect matches only: lookahead never reaches the log or duplicates the grammar.
    if any(m.start(pattern.groups) == len(text) + len(completion) - 1
           for pattern, completion in ((_AUTH_HEADER_RE, "?"),
                                       (_SECRET_HEADER_RE, "?"),
                                       (_SECRET_HEADER_RE, ":?"))
           for m in pattern.finditer(text + completion)):
        return True
    if sum(m[0] == '"' for m in re.finditer(r'\\.|"', text, re.DOTALL)) % 2:
        return True

    start = len(text)
    while start and (text[start - 1] in _TOKEN_BODY_CHARS
                     or _CONTROL_CHARS_RE.fullmatch(text[start - 1])):
        start -= 1
    suffix = text[start:]
    # The canonical redactor deliberately does not join a self-matching token across newlines.
    if _PREFIX_RE.search(suffix):
        return False
    stripped = _CONTROL_CHARS_RE.sub("", suffix)
    return bool(stripped) and any(p in stripped or p.startswith(stripped) for p in _PREFIX_SUBSTRINGS)


def copy_redacted_worker_log_stream(src, dst) -> None:
    """Redact complete logical records before writing, with bounded reads, not bounded records.

    Unterminated lines/quoted values and continuations stay in memory until complete or EOF;
    cutting them at an arbitrary size can expose opaque credentials to the log.
    """
    from agent.redact import redact_terminal_output

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    inside_private_key = False
    buffer = ""
    record = ""

    def _without_private_keys(line: str) -> str:
        nonlocal inside_private_key
        public = ""
        while line:
            if inside_private_key:
                end = _PRIVATE_KEY_END_RE.search(line)
                if end is None:
                    break
                line = line[end.end():]
                inside_private_key = False
            else:
                begin = _PRIVATE_KEY_BEGIN_RE.search(line)
                if begin is None:
                    public += line
                    break
                public += line[:begin.start()] + "[REDACTED PRIVATE KEY]"
                line = line[begin.end():]
                inside_private_key = True
        return public

    while True:
        raw = _read_chunk(src)
        eof = not raw
        buffer += decoder.decode(raw, final=eof)
        while buffer:
            newline = buffer.find("\n")
            if newline == -1 and not eof:
                break
            end = newline + 1 if newline != -1 else len(buffer)
            line, buffer = buffer[:end], buffer[end:]
            # Frame PEM delimiters by full lines, so neither boundary can be lost between reads.
            record += _without_private_keys(line)
            if record.endswith("\n") and not _worker_log_record_continues(record):
                dst.write(redact_terminal_output(record, force=True).encode("utf-8"))
                record = ""
        if eof:
            break
    if record:
        dst.write(redact_terminal_output(record, force=True).encode("utf-8"))


def _kill_proc_group(proc: subprocess.Popen, signum: int) -> None:
    """Signal the child's whole process group, falling back to the child."""
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), signum)  # windows-footgun: ok -- guarded POSIX capability
            return
    with contextlib.suppress(Exception):
        proc.send_signal(signum)


def _install_signal_forwarders(proc: subprocess.Popen) -> dict[int, object]:
    previous: dict[int, object] = {}
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
    """Best-effort: make this wrapper a process-group leader on POSIX."""
    if hasattr(os, "setpgrp"):
        with contextlib.suppress(OSError):
            os.setpgrp()


def run_worker_with_redacted_log(log_path: Path, command: list[str]) -> int:
    """Run *command*, copying its combined stdout/stderr into a redacted log."""
    _lead_process_group()
    try:
        with open_worker_log_file(log_path) as log_f:
            from tools.kanban_tools import register_current_worker_from_env

            # Recovery must track the group-owning wrapper before its child can self-register.
            if not register_current_worker_from_env():
                log_f.write(b"Worker claim was reclaimed before wrapper startup\n")
                return 1
            try:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=False,
                )
            except FileNotFoundError as exc:
                log_f.write(f"Worker launch failed: {exc}\n".encode())
                return 127

            previous_handlers = _install_signal_forwarders(proc)
            try:
                if proc.stdout is not None:
                    copy_redacted_worker_log_stream(proc.stdout, log_f)
                # health: allow HX006 -- supervise the worker's lifetime; dispatcher owns runtime limits
                return int(proc.wait())
            finally:
                _restore_signal_handlers(previous_handlers)
                if proc.poll() is None:
                    _kill_proc_group(proc, signal.SIGTERM)
                if proc.stdout is not None:
                    with contextlib.suppress(Exception):
                        proc.stdout.close()
    except (OSError, ValueError) as exc:
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
