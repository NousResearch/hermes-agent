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
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FILE_NAME = 'session_events.db'
# Streaming fragments: each one's text arrives again in the completion record that ends it.
DELTA_TYPES = frozenset({'message.delta', 'reasoning.delta', 'thinking.delta', 'tool.generating',
                         'subagent.text', 'subagent.thinking'})
# Rows kept per session; older ones are trimmed and ``first_seq`` tells a reader what it missed.
KEEP_PER_SESSION = 20000

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
        self._conn = self._connect()  # the table exists before any reader can open the file
        # health: allow HX012 -- writes one explicit file path; reads no profile-scoped state
        self._thread = threading.Thread(target=self._run, name='session-event-log', daemon=True)
        self._thread.start()

    def append(self, session_id, event_type, payload, execution=None):
        if event_type in DELTA_TYPES or not session_id:
            return
        turn_id = (execution or {}).get('admission_id')
        self._queue.put((str(session_id), event_type, turn_id, time.time(),
                         json.dumps(payload, ensure_ascii=False, default=str)))

    def forget(self, session_id):
        """Drop a deleted session's events (queued behind any append still in flight)."""
        self._queue.put((str(session_id), None, None, None, None))

    def head(self, session_id):
        """The highest committed ``seq`` for *session_id* (0 when it has none)."""
        if session_id not in self.committed and self.path.exists():
            self.committed.setdefault(session_id, self.read(session_id, 0, limit=0)[1])
        return self.committed.get(session_id, 0)

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
            try:
                self._write(conn, batch)
            except sqlite3.Error:
                logger.warning('session event log write failed for %s', self.path, exc_info=True)
                self._next.clear()  # re-read the head: a failed batch committed nothing
            finally:
                for _ in batch:
                    self._queue.task_done()

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
        conn = sqlite3.connect(f'file:{self.path}?mode=ro', uri=True, timeout=10)
        try:
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


def enabled_in_home(home):
    """``sessions.event_log`` from the owning profile's config.yaml (legacy default: off)."""
    from hermes_cli.config_effective import load_user_config_effective
    try:
        config = load_user_config_effective(Path(home) / 'config.yaml')
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the legacy default
        return False
    sessions = config.get('sessions') if isinstance(config, dict) else None
    return isinstance(sessions, dict) and sessions.get('event_log') is True
