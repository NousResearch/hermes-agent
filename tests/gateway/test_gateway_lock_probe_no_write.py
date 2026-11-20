"""Probing ``gateway.lock`` never writes into it (Windows).

``_try_acquire_file_lock`` wrote a ``"\\n"`` placeholder into a lock file it saw empty and only then
tried the byte lock. ``ProfileOwnership.reserve`` truncates the file before it writes its record, so a
probe that looked in that instant wrote ``\\r\\n`` over the first bytes of the new record. The record no
longer parsed, and every strict reader refused with "gateway PID or lock metadata is malformed", which
aborted ``hermes update``. Windows locks a byte past EOF, so nothing needs to be written at all.
"""
import json

import pytest

from gateway import status

pytestmark = pytest.mark.platforms("windows")

_RECORD = {"pid": 4242, "kind": "hermes-gateway", "argv": ["gateway", "run"], "start_time": 1}


def test_probes_leave_a_held_lock_untouched(tmp_path):
    lock_path = tmp_path / "gateway.lock"
    lock_path.touch()
    with open(lock_path, "r+", encoding="utf-8") as holder:
        assert status._try_acquire_file_lock(holder)
        try:
            holder.seek(0)
            holder.truncate()  # reserve(): the file stays empty until the record is written

            assert status._is_gateway_runtime_lock_active_strict(lock_path) is True
            assert status.is_gateway_runtime_lock_active(lock_path) is True
            assert lock_path.stat().st_size == 0

            json.dump(_RECORD, holder)
            holder.flush()
            assert json.loads(lock_path.read_text(encoding="utf-8")) == _RECORD
        finally:
            status._release_file_lock(holder)


def test_lock_excludes_contenders_until_released(tmp_path):
    lock_path = tmp_path / "gateway.lock"
    lock_path.touch()
    with open(lock_path, "r+", encoding="utf-8") as holder, open(lock_path, "r+", encoding="utf-8") as contender:
        assert status._try_acquire_file_lock(holder)
        assert not status._try_acquire_file_lock(contender)
        status._release_file_lock(holder)
        assert status._try_acquire_file_lock(contender)
        status._release_file_lock(contender)
    assert lock_path.stat().st_size == 0
