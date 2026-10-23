"""Non-gateway setup processes can reserve an existing adapter lock."""

import os

from gateway import status
from gateway.status_scoped_locks import _scoped_lock_record_is_stale


def test_live_process_owner_does_not_need_gateway_argv():
    record = status._build_pid_record()
    record["metadata"] = {"owner_kind": "process"}
    assert not _scoped_lock_record_is_stale(record, os.getpid())


def test_default_gateway_owner_still_requires_gateway_argv():
    record = status._build_pid_record()
    assert _scoped_lock_record_is_stale(record, os.getpid())


def test_recycled_process_owner_is_stale():
    record = status._build_pid_record()
    assert record["start_time"] is not None
    record["start_time"] -= 10
    record["metadata"] = {"owner_kind": "process"}
    assert _scoped_lock_record_is_stale(record, os.getpid())


def test_unknown_live_process_identity_remains_reserved(monkeypatch):
    record = status._build_pid_record()
    record["start_time"] = None
    record["metadata"] = {"owner_kind": "process"}
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    assert not _scoped_lock_record_is_stale(record, os.getpid())


def test_invalid_metadata_does_not_bypass_gateway_validation():
    record = status._build_pid_record()
    record["metadata"] = ["process"]
    assert _scoped_lock_record_is_stale(record, os.getpid())
