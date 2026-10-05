import json
from contextlib import closing

import pytest

from hermes_cli.oneshot import _attach_auxiliary_usage, _auxiliary_usage, _write_usage_file
from hermes_state import SessionDB


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
def test_resumed_fork_ledger_keeps_its_compressed_usage_only(tmp_path, marker):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        assert db.import_sessions([
            {"id": "original", "source": "cli"},
            {"id": "fork", "source": "cli", "parent_session_id": "original",
             "model_config": {marker: "original"}},
        ])["ok"]
        db.record_auxiliary_usage("original", "vision", model="v", input_tokens=100)
        db.record_auxiliary_usage("fork", "vision", model="v", input_tokens=10)
        before = _auxiliary_usage(db, "fork")
        # Another turn on the original must not change this fork's ledger.
        db.record_auxiliary_usage("original", "vision", model="v", input_tokens=1000)
        db.record_auxiliary_usage("fork", "vision", model="v", input_tokens=7)
        assert db.try_acquire_compression_lock("fork", "test-holder")
        db.publish_compression_child(
            parent_session_id="fork", child_session_id="tip", source="cli",
            model_config={marker: "original"}, compression_lock_holder="test-holder",
            messages=[{"role": "user", "content": "Continue the summarized task"}],
        )
        db.record_auxiliary_usage("tip", "vision", model="v", input_tokens=3)
        result = {"session_id": "tip", "estimated_cost_usd": 0, "total_tokens": 20, "api_calls": 1}
        _attach_auxiliary_usage(result, db, before)
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), result)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["auxiliary"]["input_tokens"] == 10
    assert report["auxiliary"]["api_calls"] == 2
    assert report["total_including_auxiliary"]["total_tokens"] == 30
