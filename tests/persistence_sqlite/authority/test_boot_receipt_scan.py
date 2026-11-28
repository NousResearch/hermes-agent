"""The boot receipt scan reads the policy prefix through the state_meta key index (GLOB), not a
LIKE-powered full scan of every state_meta row — and so matches the prefix case-exactly."""

import json

from persistence.sqlite.authority import _local
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX


def _seed(store, *keys):
    def write(conn):
        conn.execute("INSERT OR REPLACE INTO sessions(id, source, started_at) VALUES('live', 'cli', 1)")
        for key in keys:
            conn.execute("INSERT OR REPLACE INTO state_meta(key, value) VALUES(?, ?)",
                         (key, json.dumps({"entry": {"session_id": "live"}})))
    store.holder()._execute_write(write)


def test_boot_receipt_scan_skips_keys_that_differ_in_case(store):
    _seed(store, POLICY_PREFIX + "live", POLICY_PREFIX.upper() + "decoy")
    # LIKE would match the decoy case-insensitively and restore a bogus session id at boot.
    assert _local.local_receipt_ids(store.holder()) == ["live"]


def test_boot_receipt_scan_uses_the_state_meta_key_index(store):
    with store.holder()._read_ctx() as conn:
        plan = [row[3] for row in conn.execute(
            "EXPLAIN QUERY PLAN " + _local._BOOT_RECEIPTS_SQL,
            (POLICY_PREFIX + "*", len(POLICY_PREFIX) + 1, CLOSE_PREFIX, len(POLICY_PREFIX) + 1))]
    assert plan and not any("SCAN state_meta" in step for step in plan), plan
