"""Durable, gapless per-session event log (opt-in ``sessions.event_log``).

``SessionEvents.publish`` keeps its in-memory replay ring (bounded, reset by a restart) for
WebSocket reconnects. When a profile turns this log on, every published record frame is also
appended here with a per-session ``seq`` that never repeats and never skips, survives a gateway
restart, and lets an HTTP-only client (a watch, a shortcut, a proxy that only passes
request/response) long-poll ``GET /api/sessions/{id}/events?after=<seq>``.

Token deltas are not records: a turn's ``message.complete`` / ``reasoning.available`` /
``tool.complete`` already carries the full text, so the log stores those and skips the deltas.

Storage is a sibling ``session_events.db`` in the owning profile's home, not a ``state.db`` table:
it is append-only, written by one background thread off the event loop, never joins the
transcript's writer lock or migrations, and deleting the file loses nothing the transcript needs.
"""
import atexit
import contextlib
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CLOSE = object()

FILE_NAME = 'session_events.db'
# Streaming fragments: each one's text arrives again in the completion record that ends it.
DELTA_TYPES = frozenset({'message.delta', 'reasoning.delta', 'thinking.delta', 'tool.generating',
                         'subagent.text', 'subagent.thinking'})
# Rows kept per session; older ones are trimmed and ``first_seq`` tells a reader what it missed.
KEEP_PER_SESSION = 20000
# How long a failing write is retried before the log gives up (see ``EventLog.failed``).
FAIL_AFTER_S = 60.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    turn_id TEXT,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
) WITHOUT ROWID;
"""


class EventLog:
    """One profile's log: ``append`` never blocks the publisher; one writer thread assigns ``seq``."""

    def __init__(self, home):
        self.path = Path(home) / FILE_NAME
        self._queue = queue.Queue()
        self._next = {}
        # session -> highest committed seq, so an idle long-poll waits on memory, not SQLite.
        self.committed = {}
        # Set when SQLite kept refusing writes for FAIL_AFTER_S: the log stops taking records (no
        # unbounded backlog) and readers get ``event_log_unavailable`` instead of a replay with holes.
        self.failed = False
        self._conn = self._connect()  # the table exists before any reader can open the file
        # health: allow HX012 -- writes one explicit file path; reads no profile-scoped state
        self._thread = threading.Thread(target=self._run, name='session-event-log', daemon=True)
        self._thread.start()

    def append(self, session_id, event_type, payload, execution=None):
        if event_type in DELTA_TYPES or not session_id or self.failed:
            return
        turn_id = (execution or {}).get('admission_id')
        self._queue.put((str(session_id), event_type, turn_id, time.time(),
                         json.dumps(payload, ensure_ascii=False, default=str)))

    def forget(self, session_id):
        """Drop a deleted session's events (queued behind any append still in flight)."""
        self._queue.put((str(session_id), None, None, None, None))

    def head(self, session_id):
        """The highest committed ``seq`` for *session_id* (0 when it has none). Reads the file only
        for a session not seen since this process started; only sessions with events are cached."""
        if session_id not in self.committed:
            last = self.read(session_id, 0, limit=0)[1]
            if last:
                self.committed.setdefault(session_id, last)
            return last
        return self.committed[session_id]

    def flush(self, timeout=2.0):
        """Wait (bounded) until every appended event is committed; True when the queue drained."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self._queue.unfinished_tasks

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript(_SCHEMA)
        return conn

    def _run(self):
        conn = self._conn
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            closing = batch[-1] is _CLOSE
            batch = [item for item in batch if item is not _CLOSE]
            self._commit(conn, batch)
            for _ in range(len(batch) + closing):
                self._queue.task_done()
            if closing:
                conn.close()
                return

    def _commit(self, conn, batch):
        """Commit *batch* in order, retrying a failed write: dropping it would leave a hole the
        numbering hides (the next event takes the lost one's ``seq``)."""
        if self.failed:
            # Recording stopped, but a deleted session's events must still go: try its deletes.
            batch = [item for item in batch if item[1] is None]
            with contextlib.suppress(sqlite3.Error):
                if batch:
                    self._write(conn, batch)
            return
        delay, deadline = 0.05, time.monotonic() + FAIL_AFTER_S
        while batch and not self.failed:
            try:
                self._write(conn, batch)
                return
            except sqlite3.Error:
                logger.warning('session event log write failed for %s; retrying', self.path, exc_info=True)
                self._next.clear()  # a failed batch committed nothing: re-read each head
                if time.monotonic() >= deadline:
                    logger.error('session event log %s disabled until restart: writes keep failing', self.path)
                    self.failed = True
                    return
                time.sleep(delay)
                delay = min(delay * 2, 5.0)

    def close(self, timeout=None):
        """Commit what is queued, then stop the writer and close the file. ``timeout=None`` returns
        at once (the writer finishes on its own thread); a number waits that long for it."""
        self._queue.put(_CLOSE)
        if timeout is not None:
            self._thread.join(timeout)

    def _write(self, conn, batch):
        touched = set()
        with conn:
            for session_id, event_type, turn_id, created_at, payload in batch:
                if event_type is None:
                    conn.execute('DELETE FROM session_events WHERE session_id=?', (session_id,))
                    self._next.pop(session_id, None)
                    touched.discard(session_id)
                    continue
                seq = self._next.get(session_id)
                if seq is None:
                    seq = conn.execute('SELECT COALESCE(MAX(seq), 0) + 1 FROM session_events WHERE session_id=?',
                                       (session_id,)).fetchone()[0]
                conn.execute('INSERT INTO session_events VALUES (?,?,?,?,?,?)',
                             (session_id, seq, event_type, turn_id, created_at, payload))
                self._next[session_id] = seq + 1
                touched.add(session_id)
            for session_id in touched:
                conn.execute('DELETE FROM session_events WHERE session_id=? AND seq<=?',
                             (session_id, self._next[session_id] - 1 - KEEP_PER_SESSION))
        for session_id, event_type, *_ in batch:
            if event_type is None:
                self.committed.pop(session_id, None)
        self.committed.update({session_id: self._next[session_id] - 1 for session_id in touched})

    def read(self, session_id, after, *, limit=100, types=(), turn_id=None):
        """``(events, last_seq, first_seq)``: events with ``seq > after`` matching the filters."""
        where, args = ['session_id=?', 'seq>?'], [session_id, after]
        if types:
            where.append('(' + ' OR '.join('type GLOB ?' for _ in types) + ')')
            args.extend(types)
        if turn_id:
            where.append('turn_id=?')
            args.append(turn_id)
        conn = sqlite3.connect(f'file:{self.path}?mode=ro', uri=True, timeout=10, isolation_level=None)
        try:
            conn.execute('BEGIN')  # rows and bounds from one snapshot
            rows = conn.execute(
                f'SELECT seq, type, turn_id, created_at, payload_json FROM session_events WHERE {" AND ".join(where)}'
                ' ORDER BY seq LIMIT ?', (*args, limit)).fetchall()
            first, last = conn.execute('SELECT MIN(seq), MAX(seq) FROM session_events WHERE session_id=?',
                                       (session_id,)).fetchone()
        finally:
            conn.close()
        events = [{'seq': seq, 'type': kind, 'turn_id': turn, 'created_at': at, 'payload': json.loads(payload)}
                  for seq, kind, turn, at, payload in rows]
        return events, last or 0, first or 0


class LiveSessions(dict):
    """``SessionAuthority.sessions``: binds each ``LiveSession`` put here to the profile's log, so
    every creation path (there are many) feeds it without each one knowing the log exists."""

    def __init__(self, log=None):
        super().__init__()
        self.log = log

    def __setitem__(self, key, live):
        # Bound with the authority's own key: some publishers pass a route's chat id instead.
        live.event_stream.log = None if self.log is None else (self.log, key)
        super().__setitem__(key, live)

    def setdefault(self, key, live=None):
        if key not in self:
            self[key] = live
        return self[key]


_LOGS = {}
_LOGS_LOCK = threading.Lock()


def log_for_home(home, enabled):
    """The profile's shared ``EventLog`` when ``sessions.event_log`` is on, else None."""
    if not enabled:
        return None
    key = str(Path(home).resolve())
    with _LOGS_LOCK:
        if key not in _LOGS:
            _LOGS[key] = EventLog(key)
        return _LOGS[key]


def release_home(home):
    """Close *home*'s log (its profile stopped being served); a later ``log_for_home`` reopens it.
    Never waits for the writer: the caller may be the gateway loop."""
    with _LOGS_LOCK:
        log = _LOGS.pop(str(Path(home).resolve()), None)
    if log is not None:
        log.close()


def close_all():
    """Gateway shutdown: commit every queued event before the process exits."""
    with _LOGS_LOCK:
        logs = list(_LOGS.values())
        _LOGS.clear()
    for log in logs:
        log.close(timeout=5.0)


atexit.register(close_all)


def enabled_in_home(home):
    """``sessions.event_log`` from the owning profile's config.yaml (legacy default: off)."""
    from hermes_cli.config_effective import load_user_config_effective
    try:
        config = load_user_config_effective(Path(home) / 'config.yaml')
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the legacy default
        return False
    sessions = config.get('sessions') if isinstance(config, dict) else None
    return isinstance(sessions, dict) and sessions.get('event_log') is True
