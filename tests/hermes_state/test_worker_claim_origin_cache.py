"""``worker_claim_origin`` remembers its answer per execution, and the memory is exact.

Every worker RPC asks for the claim origin. Re-scanning all of an execution's receipts each time
parsed every full-transcript read receipt a running execution keeps, so the cost grew with the run,
and on the live gateway one call took 12 s on the event loop (py-spy, 2026-11-09). The cache only
holds because receipts are append-only in sequence order; these tests pin that the cached answer
always equals the full scan it replaces.
"""
import contextlib
import json
import random
import sqlite3

import pytest

import hermes_state_worker_compression as wc
from hermes_state_worker_compression import worker_claim_origin

_FULL_SCAN = ("SELECT result_json FROM worker_receipts WHERE execution_id=? "
              "AND json_extract(result_json,'$.worker_assignment.parent') IS NOT NULL ORDER BY sequence LIMIT 1")


class _Db:
    """The one table the claim reads, with the real schema, and a count of reads taken."""

    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn.execute('CREATE TABLE worker_receipts (execution_id TEXT NOT NULL, sequence INTEGER NOT NULL '
                          'CHECK (sequence > 0), payload_digest TEXT NOT NULL, result_json TEXT NOT NULL, '
                          'PRIMARY KEY (execution_id, sequence))')
        self.reads = 0

    @contextlib.contextmanager
    def _read_ctx(self):
        self.reads += 1
        yield self.conn

    def append(self, execution_id, result, *, commit=True):
        sequence = (self.conn.execute('SELECT max(sequence) FROM worker_receipts WHERE execution_id=?',
                                      (execution_id,)).fetchone()[0] or 0) + 1
        self.conn.execute('INSERT INTO worker_receipts VALUES (?,?,?,?)',
                          (execution_id, sequence, 'digest', json.dumps(result)))
        if commit:
            self.conn.commit()

    def full_scan(self, params):
        row = self.conn.execute(_FULL_SCAN, (params['execution_id'],)).fetchone()
        return json.loads(row[0])['worker_assignment']['parent'] if row else params['session_id']


def _params(execution_id='w1', session_id='local-child'):
    return {'execution_id': execution_id, 'session_id': session_id}


def _read_receipt(n):
    """A running execution's read receipt: a whole transcript, the thing that made the scan slow."""
    return {'messages': [{'role': 'user', 'content': f'turn {n} ' + 'x' * 200}]}


def _publish(parent, child):
    return {'value': None, 'worker_assignment': {'parent': parent, 'session_id': child}, 'annotations': []}


def test_no_rotation_answers_the_session_and_remembers_how_far_it_scanned():
    db = _Db()
    for n in range(3):
        db.append('w1', _read_receipt(n))
    assert worker_claim_origin(db, _params()) == 'local-child'
    assert db._worker_claim_origin_cache[0]['w1'] == (3, None)


def test_the_first_rotation_is_final_even_after_a_later_one():
    db = _Db()
    db.append('w1', _read_receipt(0))
    assert worker_claim_origin(db, _params()) == 'local-child'
    db.append('w1', _publish('local-root', 'local-child'))
    assert worker_claim_origin(db, _params()) == 'local-root'
    db.append('w1', _publish('local-child', 'local-grandchild'))
    assert worker_claim_origin(db, _params(session_id='local-grandchild')) == 'local-root'


def test_a_found_origin_reads_nothing_again():
    db = _Db()
    db.append('w1', _publish('local-root', 'local-child'))
    assert worker_claim_origin(db, _params()) == 'local-root'
    reads = db.reads
    for _ in range(5):
        assert worker_claim_origin(db, _params()) == 'local-root'
    assert db.reads == reads


def test_it_always_equals_the_full_scan_it_replaces():
    """Random interleavings of appends, rotations and claims across executions."""
    rng = random.Random(20261109)
    db = _Db()
    executions = ['w1', 'w2', 'w3']
    for step in range(400):
        execution_id = rng.choice(executions)
        if rng.random() < 0.6:
            rotate = rng.random() < 0.08
            db.append(execution_id, _publish(f'parent-{step}', f'child-{step}') if rotate else _read_receipt(step))
        params = _params(execution_id, session_id=f'session-{step}')
        assert worker_claim_origin(db, params) == db.full_scan(params)


def test_an_uncommitted_answer_is_not_remembered():
    db = _Db()
    db.append('w1', _publish('local-root', 'local-child'), commit=False)
    assert db.conn.in_transaction
    assert worker_claim_origin(db, _params()) == 'local-root'
    assert 'w1' not in db.__dict__.get('_worker_claim_origin_cache', ({},))[0]
    db.conn.rollback()
    assert worker_claim_origin(db, _params()) == 'local-child'


def test_each_database_keeps_its_own_answers():
    one, two = _Db(), _Db()
    one.append('w1', _publish('root-one', 'child'))
    two.append('w1', _read_receipt(0))
    assert worker_claim_origin(one, _params()) == 'root-one'
    assert worker_claim_origin(two, _params()) == 'local-child'


def test_the_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(wc, '_CLAIM_ORIGIN_CACHE_MAX', 4)
    db = _Db()
    for n in range(10):
        db.append(f'w{n}', _read_receipt(n))
        worker_claim_origin(db, _params(f'w{n}'))
    cache = db._worker_claim_origin_cache[0]
    assert len(cache) == 4 and list(cache) == ['w6', 'w7', 'w8', 'w9']


def test_an_execution_with_no_receipts_answers_the_session():
    db = _Db()
    assert worker_claim_origin(db, _params()) == 'local-child'
    db.append('w1', _publish('local-root', 'local-child'))
    assert worker_claim_origin(db, _params()) == 'local-root'


@pytest.mark.parametrize('upto', [None, 0])
def test_a_connection_that_reports_no_newest_receipt_is_a_miss(upto):
    """A stub connection (or an empty table) answering no row for ``max`` must not crash the claim."""
    class _Conn:
        sql = ''

        def execute(self, sql, *_a):
            self.sql = sql
            return self

        def fetchone(self):
            if 'max(sequence)' in self.sql:
                return None if upto is None else (upto,)
            return None

    class _StubDb:
        def _read_ctx(self):
            return contextlib.nullcontext(_Conn())

    assert worker_claim_origin(_StubDb(), _params()) == 'local-child'
