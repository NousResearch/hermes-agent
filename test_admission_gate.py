#!/usr/bin/env python3
"""Minimal acceptance test for the admission_conflict gate in hermes_state_runtime.py.

Verifies the idempotent-replay path and the conflict path of admit_session_input.
"""
import os, sys, sqlite3, json
sys.path.insert(0, '/home/ubuntu/.hermes/hermes-agent')
import hermes_state_runtime as HSR

DB_SCHEMA = '''
CREATE TABLE session_admissions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    admission_id TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    target_session_id TEXT NOT NULL,
    lineage_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN ('queue','steer','redirect')),
    status TEXT NOT NULL CHECK (status IN ('queued','started','unknown','terminal')),
    outcome TEXT, owner_epoch INTEGER, generation INTEGER
);
CREATE TABLE sessions (id TEXT PRIMARY KEY);
CREATE TABLE runtime_epoch (singleton INTEGER PRIMARY KEY, epoch INTEGER, instance_id TEXT);
CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
'''

class StubDB:
    def __init__(self, conn):
        self.conn = conn
    def _read_ctx(self):
        from contextlib import contextmanager
        @contextmanager
        def _ctx():
            try:
                yield self.conn
            finally:
                pass
        return _ctx()
    def _execute_write(self, write):
        try:
            return write(self.conn)
        finally:
            self.conn.commit()

def seed(conn, principal_id, session_id, request_id, payload, intent='queue', status='queued'):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = HSR.admission_fingerprint(canonical_target=session_id, payload={'input': payload, 'intent': intent})
    conn.execute('INSERT INTO session_admissions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (1, 'a1', request_id, principal_id, session_id, '["sess-1"]', encoded,
         digest, intent, status, None, None, None))
    conn.execute('INSERT INTO sessions VALUES (?)', (session_id,))
    conn.execute('INSERT INTO runtime_epoch VALUES (1,1,?)', ('x',))

def main():
    fails = 0
    def check(name, cond, extra=''):
        nonlocal fails
        print(('PASS' if cond else 'FAIL'), name, extra)
        fails += (not cond)

    # A: idempotent replay returns the existing row and inserts NO new admission.
    con = sqlite3.connect(':memory:'); con.row_factory = sqlite3.Row
    con.executescript(DB_SCHEMA)
    seed(con, 'p1', 'sess-1', 'req-1', {'q': 'a'})
    got = HSR.admit_session_input(StubDB(con), epoch=1, principal_id='p1', session_id='sess-1',
                                  request_id='req-1', payload={'q': 'a'})
    n = con.execute('SELECT COUNT(*) c FROM session_admissions').fetchone()['c']
    check('A idempotent replay returns stored row', got['admission_id'] == 'a1', str(got))
    check('A idempotent replay adds no row', n == 1, f'count={n}')

    # B: A reused key with a DIFFERENT payload raises admission_conflict (fail-closed).
    con2 = sqlite3.connect(':memory:'); con2.row_factory = sqlite3.Row
    con2.executescript(DB_SCHEMA)
    seed(con2, 'p1', 'sess-1', 'req-1', {'q': 'a'})
    try:
        HSR.admit_session_input(StubDB(con2), epoch=1, principal_id='p1', session_id='sess-1',
                                request_id='req-1', payload={'q': 'b'})
        check('B conflict raises admission_conflict', False, 'no exception raised')
    except HSR.RuntimeStoreError as e:
        check('B conflict raises admission_conflict', e.reason == 'admission_conflict', repr(e))

    # C: Different intent, same key + same digest => is it idempotent or conflict?
    # The digest fingerprint includes intent, so a 'steer' of the same payload has a DIFFERENT digest.
    con3 = sqlite3.connect(':memory:'); con3.row_factory = sqlite3.Row
    con3.executescript(DB_SCHEMA)
    seed(con3, 'p1', 'sess-1', 'req-1', {'q': 'a'}, intent='queue')
    try:
        HSR.admit_session_input(StubDB(con3), epoch=1, principal_id='p1', session_id='sess-1',
                                request_id='req-1', payload={'q': 'a'}, intent='steer')
        check('C steer same-payload not blocked', False, 'no exception')
    except HSR.RuntimeStoreError as e:
        check('C steer same-payload raises admission_conflict', e.reason == 'admission_conflict', repr(e))

    print('FAILS=', fails)
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
