"""Test coverage for gateway/delivery_ledger.py — LOW coverage.

Tests the pure helpers: owner stamp, obligation ID computation, and
process liveness check. All database access uses tmp_path.
"""

import os
from unittest.mock import patch

import pytest

from gateway.delivery_ledger import (
    _owner_stamp,
    _owner_alive,
    compute_obligation_id,
)


class TestComputeObligationId:
    def test_deterministic(self):
        a = compute_obligation_id("session-1", "msg-ref-1", "content-here")
        b = compute_obligation_id("session-1", "msg-ref-1", "content-here")
        assert a == b

    def test_different_inputs_different_ids(self):
        a = compute_obligation_id("session-1", "msg-ref-1", "content-a")
        b = compute_obligation_id("session-1", "msg-ref-1", "content-b")
        assert a != b

    def test_returns_string(self):
        result = compute_obligation_id("s", "m", "c")
        assert isinstance(result, str)
        assert len(result) > 0


class TestOwnerStamp:
    def test_returns_tuple(self):
        result = _owner_stamp()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestOwnerAlive:
    def test_invalid_pid_not_alive(self):
        assert _owner_alive(-1, None) is False
        assert _owner_alive(0, None) is False
        assert _owner_alive(None, None) is False
