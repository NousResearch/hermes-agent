"""Approving a user must never rewrite an approved store it could not read.

``PairingStore._load_json`` returns ``{}`` both when a pairing file does not
exist yet and when it exists but could not be read or parsed. For a lookup that
is the correct fail-closed answer. For ``_approve_user`` it was destructive: the
one new entry was added to that empty dict and ``_save_json`` atomically
replaced the file, so every previously approved user was gone from disk.

The read failure alone is recoverable; the write made it permanent. Two real
triggers land in the same branch — a root-owned 0600 file under a hermes-owned
directory (issue #10270, the ``docker exec`` symptom) is unreadable yet still
atomically replaceable, and a truncated restore leaves readable bytes that are
invalid JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gateway.pairing as pairing_mod

# Resolved lazily, never bound at import time: a sibling suite
# (test_pairing_allowlist_bypass.py) calls importlib.reload on this module, and
# a class captured here would go stale and stop matching the raised instance.
def _unreadable_error():
    return pairing_mod.PairingStoreUnreadableError


EXISTING = {
    "111": {"user_name": "alice", "approved_at": 1.0},
    "222": {"user_name": "bob", "approved_at": 2.0},
    "333": {"user_name": "carol", "approved_at": 3.0},
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    s = pairing_mod.PairingStore()
    # Pin the store directory per test: PairingStore resolves it through
    # get_hermes_dir(), which is shared enough that the per-user pairing
    # rate limit would leak between tests.
    s._dir = tmp_path / "pairing"
    s._dir.mkdir(parents=True, exist_ok=True)
    return s


def _seed_approved(store) -> Path:
    path = store._approved_path("telegram")
    path.write_text(json.dumps(EXISTING, indent=2), encoding="utf-8")
    return path


class TestApproveRefusesUnreadableApprovedStore:
    def test_approve_code_refuses_and_leaves_the_file_byte_identical(self, store):
        path = _seed_approved(store)
        code = store.generate_code("telegram", "999", "dave")
        assert code

        path.write_text('{"111": {"user_name": "alice",', encoding="utf-8")
        before = path.read_bytes()

        with pytest.raises(_unreadable_error()):
            store.approve_code("telegram", code)

        assert path.read_bytes() == before

    def test_refusal_does_not_burn_the_one_time_code(self, store):
        """The pending entry must survive so a repaired store can still pair."""
        path = _seed_approved(store)
        code = store.generate_code("telegram", "999", "dave")
        path.write_text("not json at all", encoding="utf-8")

        with pytest.raises(_unreadable_error()):
            store.approve_code("telegram", code)

        # Repair the store — the same code must still work.
        path.write_text(json.dumps(EXISTING, indent=2), encoding="utf-8")
        assert store.approve_code("telegram", code) == {
            "user_id": "999",
            "user_name": "dave",
        }
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert set(saved) == {"111", "222", "333", "999"}

    def test_approve_request_is_guarded_too(self, store):
        path = _seed_approved(store)
        store.generate_code("telegram", "999", "dave")
        pending = store._load_json(store._pending_path("telegram"))
        request_id = next(iter(pending))

        path.write_text("{oops", encoding="utf-8")
        before = path.read_bytes()

        with pytest.raises(_unreadable_error()):
            store.approve_request("telegram", request_id)

        assert path.read_bytes() == before


class TestUnaffectedPaths:
    def test_first_ever_approval_still_creates_the_file(self, store):
        path = store._approved_path("telegram")
        assert not path.exists()

        code = store.generate_code("telegram", "999", "dave")
        assert store.approve_code("telegram", code) is not None
        assert set(json.loads(path.read_text(encoding="utf-8"))) == {"999"}

    def test_normal_approval_preserves_existing_users(self, store):
        path = _seed_approved(store)
        code = store.generate_code("telegram", "999", "dave")

        assert store.approve_code("telegram", code) is not None

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert set(saved) == {"111", "222", "333", "999"}

    def test_lookups_still_fail_closed_without_raising(self, store):
        """An unreadable whitelist must deny, not raise, on the read path."""
        path = _seed_approved(store)
        path.write_text("{broken", encoding="utf-8")

        assert store.is_approved("telegram", "222") is False
        assert store.list_approved("telegram") == []

    def test_revoke_on_unreadable_store_reports_not_found_and_writes_nothing(
        self, store
    ):
        path = _seed_approved(store)
        path.write_text("{broken", encoding="utf-8")
        before = path.read_bytes()

        assert store.revoke("telegram", "222") is False
        assert path.read_bytes() == before
