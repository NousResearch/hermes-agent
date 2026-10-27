"""Bounded, redacted per-profile sink for managed-worker stderr.

The worker routes its own stderr AND its ordinary stdout (``os.dup2``) here, so tracebacks,
provider error echoes and tool prints land in this stream. The owner drains it on a dedicated
thread (an undrained pipe would block the child), redacts every line before it touches disk,
and keeps ``<home>/logs/managed-worker.log`` private (0600) and capped at
``MAX_LOG_BYTES`` plus one rotated copy.

A worker bound to a Kanban task also mirrors each redacted line into the card log the dispatcher
opened for it (``kanban/logs/<task>.log``, the dashboard's "Worker log"): without the mirror a
managed worker's card log carries only the client's ``Session:`` line and the exit trailer, so a
running worker looks dead. The mirror reuses this module's redaction and per-line cap, and leaves
the card log's rotation and mode to the dispatcher (``hermes_cli.kanban_worker_log``).
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


def kanban_card_log_path(policy):
    """The card log of the Kanban task this managed worker executes, or None when it has none.

    ``policy.kanban_json`` is the dispatcher's task context; the dispatcher opened exactly this
    file for the worker (``hermes_cli.kanban_db.worker_log_path``), so the mirror appends to the
    log the dashboard already reads. A managed worker for an ordinary session carries no context
    and mirrors nothing.
    """
    if getattr(policy, 'kanban_json', None) is None:
        return None
    import json
    context = json.loads(policy.kanban_json) or {}
    task_id = str(context.get('task_id') or '')
    if not task_id:
        return None
    from hermes_cli.kanban_db import worker_log_path
    return worker_log_path(task_id, board=context.get('board'))


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


def _append_card_log(path, text):
    """Append one already-redacted line to the owning task's card log.

    The dispatcher created that file (and its directory) before spawning the worker, and owns its
    rotation (``hermes_cli.kanban_worker_log``); the drain only ever appends, so a rotation is a
    rename this never races. A missing directory means the dispatcher never opened this log: the
    mirror writes nothing rather than materializing a log at a path nothing reads.
    """
    if not path.parent.is_dir():
        raise FileNotFoundError(path)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, text.encode('utf-8', 'replace'))
    finally:
        os.close(fd)


def _redact(line, scrub):
    from agent.redact import redact_sensitive_text
    for value in sorted((v for v in scrub() if isinstance(v, str) and v), key=len, reverse=True):
        line = line.replace(value, '***')
    return redact_sensitive_text(line, force=True)


def drain_worker_stderr(stream, path, pid, scrub, card_log=None):
    """Thread body: read ``stream`` to EOF; ``scrub()`` returns this worker's exact secrets
    (launch key, assignment secret) known so far. Never stops reading on a write failure.

    ``card_log`` is the owning Kanban task's card log (the dashboard's "Worker log"). When set,
    every redacted line is mirrored there as well, so a running managed worker shows its progress
    instead of only the client's ``Session:`` line. It stays None for a worker bound to no task.
    """
    skipping = False
    warned = set()

    def warn(target, exc):
        # Keep draining regardless: a child blocked on a full stderr pipe would wedge.
        if target not in warned:
            warned.add(target)
            logger.warning('managed worker log %s not written (%s); dropping its lines', target,
                           type(exc).__name__)

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
                entry = f'[pid {pid}] ' + (line if whole else line[:KEPT_LINE_CHARS] + ' [truncated]\n')
            except (OSError, ValueError) as exc:
                warn(path, exc)
                continue
            for target, write in ((path, _append), (card_log, _append_card_log)):
                if target is None:
                    continue
                try:
                    write(target, entry)
                except (OSError, ValueError) as exc:
                    warn(target, exc)
    finally:
        stream.close()
