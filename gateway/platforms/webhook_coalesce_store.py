"""Bounded, profile-local pending coalesced work; receipts alone cannot recover payloads."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager

MAX_PENDING = 1024
MAX_MEMBERS = 10000
MAX_BYTES = 16 * 1024 * 1024


def _key(value):
    return hashlib.sha256(value.encode()).hexdigest()


class CoalesceStore:
    def __init__(self, home: Path):
        self.path = home / 'webhook_coalescing.db'

    @contextmanager
    def connection(self, *, write=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('Unsupported webhook coalescing schema')
            if version == 0:
                conn.executescript('''
                    CREATE TABLE IF NOT EXISTS batches (
                        id TEXT PRIMARY KEY, group_key TEXT UNIQUE NOT NULL,
                        work TEXT NOT NULL, bytes INTEGER NOT NULL,
                        first_at REAL NOT NULL, due REAL NOT NULL,
                        count INTEGER NOT NULL, sealed INTEGER NOT NULL DEFAULT 0,
                        held TEXT, failures INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE IF NOT EXISTS members (
                        identity TEXT PRIMARY KEY, batch_id TEXT NOT NULL, expires REAL);
                    CREATE INDEX IF NOT EXISTS members_batch ON members(batch_id);
                    PRAGMA user_version=1;
                ''')
            if write:
                conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def accept(self, identity: str, group: str, work: dict, window: float, max_wait: float):
        encoded = json.dumps(work, sort_keys=True, separators=(',', ':'))
        size = len(encoded.encode())
        with self.connection() as conn:
            now = time.time()
            conn.execute('DELETE FROM members WHERE expires IS NOT NULL AND expires < ?', (now,))
            if conn.execute('SELECT 1 FROM members WHERE identity=?', (_key(identity),)).fetchone():
                return 'duplicate'
            row = conn.execute('SELECT * FROM batches WHERE group_key=?', (_key(group),)).fetchone()
            if row and (row['sealed'] or json.loads(row['work']).get('binding') != work.get('binding')):
                return 'busy'
            quota = conn.execute('SELECT count(*), COALESCE(SUM(bytes),0) FROM batches').fetchone()
            members = conn.execute('SELECT count(*) FROM members').fetchone()[0]
            if (members >= MAX_MEMBERS or (not row and quota[0] >= MAX_PENDING)
                    or quota[1] - (row['bytes'] if row else 0) + size > MAX_BYTES):
                return 'full'
            first = row['first_at'] if row else now
            batch_id = row['id'] if row else uuid.uuid4().hex
            due = min(now + window, first + max_wait)
            if row:
                conn.execute('UPDATE batches SET work=?, bytes=?, due=?, count=count+1 WHERE id=?',
                             (encoded, size, due, batch_id))
            else:
                conn.execute('INSERT INTO batches(id,group_key,work,bytes,first_at,due,count) VALUES(?,?,?,?,?,?,1)',
                             (batch_id, _key(group), encoded, size, first, due))
            conn.execute('INSERT INTO members(identity,batch_id) VALUES(?,?)', (_key(identity), batch_id))
            return 'coalesced'

    def seen(self, identity):
        with self.connection(write=False) as conn:
            return conn.execute('SELECT 1 FROM members WHERE identity=? AND (expires IS NULL OR expires>?)',
                                (_key(identity), time.time())).fetchone() is not None

    def schedule(self):
        with self.connection(write=False) as conn:
            return [dict(row) for row in conn.execute('SELECT id,due FROM batches ORDER BY due')]

    def pending(self):
        """Explicit diagnostic read, not part of the hot-path scheduler."""
        with self.connection(write=False) as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM batches ORDER BY first_at')]

    def claim(self, batch_id, force=False):
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
            if row and (force or row['due'] <= time.time()):
                conn.execute('UPDATE batches SET sealed=1 WHERE id=?', (batch_id,))
                return dict(row)
        return None

    def defer(self, batch_id, delay, reason):
        with self.connection() as conn:
            row = conn.execute('SELECT held,failures FROM batches WHERE id=?', (batch_id,)).fetchone()
            backoff = min(60, delay * 2 ** min(row['failures'] if row else 0, 6))
            conn.execute('UPDATE batches SET due=?, held=?, failures=failures+1 WHERE id=?',
                         (time.time() + backoff, reason, batch_id))
            return row is not None and row['held'] != reason

    def finish(self, batch_id):
        with self.connection() as conn:
            conn.execute('UPDATE members SET expires=? WHERE batch_id=?', (time.time() + 3600, batch_id))
            conn.execute('DELETE FROM batches WHERE id=?', (batch_id,))
