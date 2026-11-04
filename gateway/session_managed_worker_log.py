"""Bounded, redacted per-profile sink for managed-worker stderr.

The worker routes its own stderr AND its ordinary stdout (``os.dup2``) here, so tracebacks,
provider error echoes and tool prints land in this stream. The owner drains it on a dedicated
thread (an undrained pipe would block the child), redacts every line before it touches disk,
and keeps ``<home>/logs/managed-worker.log`` private (0600) and capped at
``MAX_LOG_BYTES`` plus one rotated copy.
"""
import logging
import os
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

LOG_NAME = 'managed-worker.log'
MAX_LOG_BYTES = 1024 * 1024
# A line longer than this is redacted whole, then cut; the rest of it is discarded.
MAX_LINE_BYTES = 64 * 1024
KEPT_LINE_CHARS = 4096
_locks = {}
_locks_guard = threading.Lock()


def worker_log_path(profile_id):
    """The owning profile's sink, or None when the authority has no home path (test doubles)."""
    home = Path(str(profile_id))
    return home / 'logs' / LOG_NAME if home.is_absolute() else None


def _lock_for(path):
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def _append(path, text):
    """Append under the per-file lock; rotate once the cap would be exceeded. 0600 always."""
    from hermes_constants import mkdir_under_hermes_home
    data = text.encode('utf-8', 'replace')
    with _lock_for(path):
        mkdir_under_hermes_home(path.parent)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(data) > MAX_LOG_BYTES:
            os.replace(path, path.with_name(LOG_NAME + '.1'))
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if hasattr(os, 'fchmod'):
                os.fchmod(fd, 0o600)
            os.write(fd, data[:MAX_LOG_BYTES])
        finally:
            os.close(fd)


def _redact(line, scrub):
    from agent.redact import redact_sensitive_text
    for value in sorted((v for v in scrub() if isinstance(v, str) and v), key=len, reverse=True):
        line = line.replace(value, '***')
    return redact_sensitive_text(line, force=True)


def drain_worker_stderr(stream, path, pid, scrub):
    """Thread body: read ``stream`` to EOF; ``scrub()`` returns this worker's exact secrets
    (launch key, assignment secret) known so far. Never stops reading on a write failure."""
    skipping = False
    warned = False
    try:
        while True:
            chunk = stream.readline(MAX_LINE_BYTES)
            if not chunk:
                return
            whole = chunk.endswith(b'\n')
            if skipping:
                skipping = not whole
                continue
            skipping = not whole
            try:
                line = _redact(chunk.decode('utf-8', 'replace'), scrub)
                _append(path, f'[pid {pid}] ' + (line if whole else line[:KEPT_LINE_CHARS] + ' [truncated]\n'))
            except (OSError, ValueError) as exc:
                # Keep draining regardless: a child blocked on a full stderr pipe would wedge.
                if not warned:
                    warned = True
                    logger.warning('managed worker log %s not written (%s); dropping its lines', path,
                                   type(exc).__name__)
    finally:
        stream.close()
