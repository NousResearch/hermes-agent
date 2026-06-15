"""Tests for pairing code display — issue #46580.

Verify that ``list_pending`` returns the original pairing code (not the
hash prefix) so that ``hermes pairing list`` shows a value the operator
can pass directly to ``hermes pairing approve``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.pairing import PairingStore


@pytest.fixture()
def store(tmp_path):
    """PairingStore rooted in a temp directory."""
    with patch("gateway.pairing.PAIRING_DIR", tmp_path):
        yield PairingStore()


class TestListPendingShowsActualCode:
    """``list_pending`` must return the original code, not the hash prefix."""

    def test_code_is_approvable(self, store):
        """The code returned by list_pending can be fed to approve_code."""
        actual_code = store.generate_code("telegram", "u1", "Alice")
        pending = store.list_pending("telegram")

        assert len(pending) == 1
        displayed_code = pending[0]["code"]
        # Must be the actual code, not a hash prefix
        assert displayed_code == actual_code

        # Approving with the displayed code must succeed
        result = store.approve_code("telegram", displayed_code)
        assert result is not None
        assert result["user_id"] == "u1"

    def test_code_not_hash_prefix(self, store):
        """Displayed code must be 8 uppercase alpha chars from ALPHABET."""
        store.generate_code("signal", "u2", "Bob")
        pending = store.list_pending("signal")

        assert len(pending) == 1
        code = pending[0]["code"]
        assert len(code) == 8
        assert all(ch in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" for ch in code)

    def test_legacy_entry_without_code_field(self, store, tmp_path):
        """Entries from before the fix (no ``code`` key) fall back to hash[:8]."""
        # Manually write a legacy pending entry (no "code" key)
        platform_dir = tmp_path / "telegram-pending.json"
        legacy_entry = {
            "abc12345": {
                "hash": "aabbccdd11223344" * 4,
                "salt": os.urandom(16).hex(),
                "user_id": "u3",
                "user_name": "Charlie",
                "created_at": time.time(),
            }
        }
        platform_dir.write_text(json.dumps(legacy_entry))

        pending = store.list_pending("telegram")
        assert len(pending) == 1
        # Should fall back to hash[:8]
        assert pending[0]["code"] == "aabbccdd"

    def test_multiple_platforms(self, store):
        """Codes are correct across multiple platforms."""
        code_tg = store.generate_code("telegram", "u1", "Alice")
        code_sig = store.generate_code("signal", "u2", "Bob")

        pending_tg = store.list_pending("telegram")
        pending_sig = store.list_pending("signal")

        assert pending_tg[0]["code"] == code_tg
        assert pending_sig[0]["code"] == code_sig
