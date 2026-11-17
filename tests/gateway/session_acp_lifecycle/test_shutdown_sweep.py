"""The stop's ACP sweep costs the number of ACP sessions, not the number of local sessions.

``end_idle_acp_sessions`` used to list every ``state_meta`` policy receipt and parse each one to find
the few with ``policy.source='acp'``. A policy receipt carries the whole launch policy (tens of KB on
a real profile), so a store with thousands of local sessions spent seconds of the stop's watchdog
budget in that loop, on the gateway's event loop.
"""
import json

import pytest

from gateway import session_acp_lifecycle as lifecycle
from hermes_state_local import POLICY_PREFIX
from tests.gateway.test_owner_loop_writes_class import _local, _owner

pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")  # symbolic profile ids are relative homes


def _seed_foreign_policies(db, count):
    """Policy receipts of sessions that are not ACP (kanban workers, cron, messaging...)."""
    rows = [(POLICY_PREFIX + f"foreign-{i}", json.dumps({"policy": {"source": "kanban"}, "pad": "x" * 4096}))
            for i in range(count)]
    db._execute_write(lambda conn: conn.executemany("INSERT INTO state_meta(key,value) VALUES(?,?)", rows))


@pytest.mark.asyncio
async def test_the_sweep_reads_receipts_of_acp_candidates_only(tmp_path, monkeypatch):
    async def handle(event):
        return "ok"
    store, authority, owner = _owner(tmp_path, monkeypatch, handle)
    try:
        acp = [_local(authority, owner, tmp_path, source="acp", request_id=f"acp-{i}") for i in range(2)]
        cli = [_local(authority, owner, tmp_path, source="cli", request_id=f"cli-{i}") for i in range(20)]
        _seed_foreign_policies(store._db, 200)
        seen = []
        real = lifecycle.local_receipt

        def recorded(db, session_id):
            seen.append(session_id)
            return real(db, session_id)
        monkeypatch.setattr(lifecycle, "local_receipt", recorded)

        assert lifecycle.end_idle_acp_sessions(authority) == 2
        assert sorted(seen) == sorted(ref.session_id for ref in acp), (
            f"{len(seen)} receipts read for 2 ACP sessions among {len(cli) + 200} others")
        for ref in acp:
            assert store._db.get_session(ref.session_id)["end_reason"] == lifecycle.ACP_END_REASON
        for ref in cli:
            assert store._db.get_session(ref.session_id)["end_reason"] is None
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_peer_refreshed_acp_row_is_still_found(tmp_path, monkeypatch):
    """A peer refresh relabels an ACP row ``source='local'``; ``created_source`` keeps it findable."""
    async def handle(event):
        return "ok"
    store, authority, owner = _owner(tmp_path, monkeypatch, handle)
    try:
        ref = _local(authority, owner, tmp_path, source="acp")
        store._db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET source='local' WHERE id=?", (ref.session_id,)))
        assert store._db.get_session(ref.session_id)["created_source"] == "acp"

        assert lifecycle.end_idle_acp_sessions(authority) == 1
        assert store._db.get_session(ref.session_id)["end_reason"] == lifecycle.ACP_END_REASON
    finally:
        await owner.close()
        store.close_all_db_handles()
