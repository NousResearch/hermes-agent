"""Bounded, redacted sinks for a managed worker's output.

The worker routes its own stderr AND its ordinary stdout (``os.dup2``) here, so tracebacks,
provider error echoes and tool prints land in this stream. The owner drains it on a dedicated
thread (an undrained pipe would block the child), redacts every line before it touches disk,
and keeps ``<home>/logs/managed-worker.log`` private (0600) and capped at
``MAX_LOG_BYTES`` plus one rotated copy.

A Kanban card has a second, per-task sink: the dispatcher opens ``<board logs>/<task_id>.log``
for the process it spawns and the dashboard's Worker log tab reads it, but that process is only
the submitter now, so it prints no tool feed. The owner, which already validates every tool
frame, appends one redacted line per call there (``TaskToolLog``) — append-only, never created
or rotated here.
"""
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

logger = logging.getLogger(__name__)

LOG_NAME = 'managed-worker.log'
MAX_LOG_BYTES = 1024 * 1024
# A line longer than this is redacted whole, then cut; the rest of it is discarded.
MAX_LINE_BYTES = 64 * 1024
KEPT_LINE_CHARS = 4096
_locks = {}
_locks_guard = threading.Lock()

# The Kanban card's tool feed, one line per tool call, in the per-task log the dispatcher opened for
# the card (the file its Worker log tab and `hermes kanban log` read). A `chat -q` worker printed this
# feed there itself; a managed card's dispatcher child is only the submitter, so the owner, which
# already receives every tool frame, writes it. Every line starts with the tag and holds no line
# break: tool output can never forge the submitter's exit trailer the dead-worker sweep reads.
TASK_LOG_TAG = '[tool]'
TASK_LOG_PREVIEW_CHARS = 240
_LINE_BREAKS = re.compile('[\x00-\x1f\x7f\x85\u2028\u2029]+')


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


def format_tool_line(frame, scrub):
    """One ``[tool]`` line for a frame ``publish_worker_tool_event`` accepted: a start shows the
    executor's own preview of the arguments, a complete the result and its verdict.

    The whole text is redacted before it is cut (pattern redaction such as a PEM block or
    ``KEY=value`` needs the original line structure), then every line break is collapsed so the
    line can never forge the submitter's exit trailer, then it is shortened.
    """
    from agent.display import build_tool_preview
    name = frame['name']
    if frame['type'] == 'tool.start':
        verb = 'start'
        text = build_tool_preview(name, frame['args'], max_len=0) or json.dumps(
            frame['args'], ensure_ascii=False, default=str)
    else:
        verb = 'error' if frame['is_error'] else 'done'
        text = frame['result']
    body = _LINE_BREAKS.sub(' ', _redact(f'{name}: {text}', scrub)).strip()
    if len(body) > TASK_LOG_PREVIEW_CHARS:
        body = body[:TASK_LOG_PREVIEW_CHARS] + ' [truncated]'
    return f'{TASK_LOG_TAG} {time.strftime("%Y-%m-%d %H:%M:%S")} {verb} {body}\n'


class TaskToolLog:
    """The Kanban card's tool feed for ONE run: ``write`` appends a line per validated tool frame
    to the per-task log the dispatcher opened, off the event loop.

    The dispatcher owns that file's lifecycle (create, rotate at spawn, GC) and the submitter
    holds it open for its own output and exit trailer, so this only ever appends: no create, no
    chmod, no rotation, no lock. ``for_policy`` returns None for every non-Kanban session.
    """

    def __init__(self, kanban_json, scrub):
        self._kanban_json = kanban_json
        self._scrub = scrub
        self._path = None
        self._written = 0
        self._capped = False
        self._warned = False

    @classmethod
    def for_policy(cls, policy, scrub):
        kanban_json = getattr(policy, 'kanban_json', None)
        return cls(kanban_json, scrub) if kanban_json else None

    def write(self, frame):
        """Append one line; never raises, so a log failure cannot touch the turn."""
        if self._capped:
            return
        try:
            if self._path is None:
                from hermes_cli import kanban_db as kb
                context = json.loads(self._kanban_json)
                self._path = kb.worker_log_path(context['task_id'], board=context['board'])
            data = format_tool_line(frame, self._scrub).encode('utf-8', 'replace')
            if self._written + len(data) > MAX_LOG_BYTES:
                data = (f'{TASK_LOG_TAG} {time.strftime("%Y-%m-%d %H:%M:%S")} feed capped at '
                        f'{MAX_LOG_BYTES} bytes for this run; the session transcript keeps every '
                        'tool call\n').encode()
                self._capped = True
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            self._written += len(data)
        except FileNotFoundError:
            # No log for this run (a direct kanban.run caller): nothing to feed, quietly.
            self._capped = True
            logger.debug('kanban tool feed %s not written (no per-task log)', self._path)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning('kanban tool feed %s not written (%s); dropping its lines',
                               self._path, type(exc).__name__)


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
