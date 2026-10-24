"""Real socket and SQLite budget acceptance; helper owner is synthetic and private."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_worker_receipt_reconciliation import owner, committed
from tests.waiting import wait_until


def test_legitimate_storage_wait_longer_than_twenty_seconds(owner):
    def wait_before_storage(params):
        if not params.get('receipt_only'):
            assert owner.release.wait(40)
    owner.before_request = wait_before_storage
    timer = threading.Timer(21, owner.release.set)
    timer.start()
    start = time.monotonic()
    try:
        owner.store.touch_session_activity('owned', ts=1234567890.0)
        elapsed = time.monotonic() - start
        assert 20 < elapsed < 40
        assert len(owner.requests) == 1
        assert len(committed(owner)[1]) == 1
        assert not owner.store.journal['pending'] and owner.store.failure is None
        print('VALID_WAIT_SECONDS', elapsed)
    finally:
        owner.release.set(); timer.join(2)


def test_true_hang_is_bounded_then_late_commit_is_not_replayed(owner):
    owner.rpc.budget = SimpleNamespace(rpc_timeout_seconds=1.2, recovery_timeout_seconds=.3,
                                      progress_interval_seconds=.05, close_timeout_seconds=.1)
    def wait_before_storage(params):
        if not params.get('receipt_only'):
            owner.entered.set()
            assert owner.release.wait(5)
    owner.before_request = wait_before_storage
    start = time.monotonic()
    try:
        with pytest.raises(Exception):
            owner.store.touch_session_activity('owned', ts=1234567890.0)
        assert time.monotonic() - start < 3
        assert owner.entered.is_set() and committed(owner) == (0, [])
        pending = owner.store.path.read_bytes()
        assert owner.store.failure and len(owner.store.journal['pending']) == 1
        assert len(owner.requests) == 2 and owner.requests[1]['receipt_only'] is True
    finally:
        owner.release.set()
    wait_until(lambda: len(committed(owner)[1]) == 1, timeout=3)
    assert owner.store.path.read_bytes() == pending and owner.store.failure
    assert len(owner.requests) == 2


def test_postcommit_delay_recovers_exact_receipt_with_no_new_mutation(owner, monkeypatch):
    owner.rpc.budget = SimpleNamespace(rpc_timeout_seconds=1.2, recovery_timeout_seconds=.3,
                                      progress_interval_seconds=.05, close_timeout_seconds=.1)
    def hold():
        owner.entered.set()
        assert owner.release.wait(5)
    monkeypatch.setattr(owner.db, '_try_wal_checkpoint', hold)
    owner.db._write_count = owner.db._CHECKPOINT_EVERY_N_WRITES - 1
    try:
        rows = [{'role':'assistant','content':'synthetic already finished result'}]
        assert owner.store.append_messages_batch('owned', rows) == 1
        assert owner.entered.is_set() and not owner.release.is_set()
        count, receipts = committed(owner)
        assert count == len(receipts) == 1 and rows[0]['_row_id']
        assert owner.requests[1] == dict(owner.requests[0], receipt_only=True)
        assert not owner.store.journal['pending'] and owner.store.failure is None
    finally:
        owner.release.set()
        assert owner.returned.wait(3)


def test_explicit_retry_after_late_commit_is_read_only(owner):
    owner.rpc.budget = SimpleNamespace(rpc_timeout_seconds=.8, recovery_timeout_seconds=.2,
                                      progress_interval_seconds=.05, close_timeout_seconds=.1)
    def wait_before_storage(params):
        if not params.get('receipt_only'):
            owner.entered.set()
            assert owner.release.wait(5)
    owner.before_request = wait_before_storage
    try:
        with pytest.raises(Exception):
            owner.store.touch_session_activity('owned', ts=1234567890.0)
    finally:
        owner.release.set()
    wait_until(lambda: len(committed(owner)[1]) == 1, timeout=3)
    assert owner.store.failure
    before = len(owner.requests)
    assert owner.store.retry_pending() == [{'value': None}]
    assert owner.requests[before:] == [dict(owner.requests[0], receipt_only=True)]
    assert len(committed(owner)[1]) == 1 and owner.store.failure is None
