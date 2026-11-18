"""The holder helper runs raw SQL on the store the interface handle reads."""
import importlib
import json
import sqlite3
import time

from tests.fixtures import storage_holder as h
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_raw_holder_writes_and_reads_the_handles_store(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    holder = raw_holder(store)
    holder._execute_write(lambda conn: conn.execute("UPDATE sessions SET title = ? WHERE id = ?", ("raw", "s")))
    with holder._read_ctx() as conn:
        assert conn.execute("SELECT title FROM sessions WHERE id = ?", ("s",)).fetchone()[0] == "raw"
    assert store.sessions.get("s").title == "raw"


def test_key_builders_prefix_their_suffix():
    for build in (h.policy_key, h.api_binding_key, h.terminal_admission_key, h.admission_result_key,
                  h.retired_session_key, h.retired_route_key, h.history_binding_key, h.retired_media_key):
        assert build("x") == build() + "x" and build().startswith("gateway.")


def test_result_codec_round_trips():
    result = {"result": {"final_response": "ok", "messages": [{"role": "assistant", "content": "ok"}]}}
    assert h.load_result(json.dumps(h.encode_result(result))) == result
    assert h.result_decoder()(json.dumps(h.encode_result(result))) == result


def test_retirement_helpers_write_list_and_reclaim_markers(tmp_path):
    store = storage_handle(tmp_path)
    h.retire_routes(store, ["gone"], expiring=["gone"])
    assert store.state_meta.get(h.retired_session_key("gone")) is not None
    assert h.expiring_retirements(store, ["gone", "other"]) == {"gone"}
    h.reclaim_expired(store, h.retired_session_key(), time.time() + 1)
    assert store.state_meta.get(h.retired_session_key("gone")) is None
    assert h.guarded_ids(store, ["gone"]) == set()
    assert h.create_conflict(store, "fresh") is None


def test_browse_helpers_reach_the_names_the_browse_reads():
    reconcile = importlib.import_module("persistence.sqlite.admin._reconcile")
    assert h.browse_integrity_scan() is reconcile._integrity_problem
    assert h.browse_budgets() == (reconcile.BROWSE_INTEGRITY_BUDGET_S, reconcile.BROWSE_RECONCILE_BUDGET_S)
    reconcile._scan_too_slow["probe"] = (0, 0)
    h.forget_slow_scans()
    assert reconcile._scan_too_slow == {}


def test_recovery_orphan_cleanup_clears_a_dangling_parent(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "child", "cli")
    conn = sqlite3.connect(str(tmp_path / "state.db"), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("UPDATE sessions SET parent_session_id = 'gone' WHERE id = 'child'")
        assert h.recovery_orphan_cleanup(conn)["sessions_parent_cleared"] == 1
        assert conn.execute("SELECT parent_session_id FROM sessions WHERE id = 'child'").fetchone()[0] is None
    finally:
        conn.close()
