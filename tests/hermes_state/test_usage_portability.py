import json
from contextlib import closing

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("export_kind", ["single", "all", "lineage"])
def test_usage_survives_json_export_import_without_double_counting(tmp_path, export_kind):
    source_path, target_path = tmp_path / "source.db", tmp_path / "target.db"
    with closing(SessionDB(source_path)) as source:
        source.create_session("root", "cli")
        source.append_message("root", "user", "original request")
        source.update_token_counts("root", model="first", billing_provider="provider-a",
                                   input_tokens=100, output_tokens=20, api_call_count=1,
                                   estimated_cost_usd=0.25, actual_cost_usd=0.2)
        source.update_token_counts("root", model="second", billing_provider="provider-b",
                                   input_tokens=50, output_tokens=10, api_call_count=1,
                                   estimated_cost_usd=0.125)
        source.record_auxiliary_usage("root", "vision", model="aux", input_tokens=12,
                                      output_tokens=3, estimated_cost_usd=0.0625)
        if export_kind == "lineage":
            assert source.try_acquire_compression_lock("root", "holder")
            source.publish_compression_child(
                parent_session_id="root", child_session_id="tip", source="cli",
                compression_lock_holder="holder", messages=[{"role": "user", "content": "summary"}],
            )
            source.record_auxiliary_usage("tip", "compression", model="aux", input_tokens=7)
        expected_rows = [dict(row) for row in source._read_all(
            "SELECT * FROM session_model_usage ORDER BY session_id, model, task")]
        expected_total = source.get_session("root")["input_tokens"]
        if export_kind == "single":
            payload = [source.export_session("root")]
        elif export_kind == "all":
            payload = source.export_all()
        else:
            payload = source.export_session_lineage("tip")["segments"]
    payload = json.loads(json.dumps(payload))
    with closing(SessionDB(target_path)) as target:
        assert target.import_sessions(payload)["ok"]
    with closing(SessionDB(target_path)) as target:
        assert [dict(row) for row in target._read_all(
            "SELECT * FROM session_model_usage ORDER BY session_id, model, task")] == expected_rows
        assert target.get_session("root")["input_tokens"] == expected_total
        assert target.auxiliary_usage_by_task("root")["vision"]["input_tokens"] == 12
        assert target.import_sessions(payload)["imported"] == 0
        assert [dict(row) for row in target._read_all(
            "SELECT * FROM session_model_usage ORDER BY session_id, model, task")] == expected_rows
        # Old exports without usage detail remain importable.
        assert target.import_sessions([{"id": "legacy", "input_tokens": 42}])["ok"]
        assert target.get_session("legacy")["input_tokens"] == 42


@pytest.mark.parametrize("usage", [
    {}, [None], [{"model": 12}], [{"model": "m", "input_tokens": "bad"}],
    [{"model": "m", "estimated_cost_usd": float("inf")}],
    [{"model": "m", "session_id": "bystander"}],
    [{"model": "m"}, {"model": "m"}],
])
def test_invalid_usage_rejects_whole_import_before_writes(tmp_path, usage):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("bystander", "cli")
        db.record_auxiliary_usage("bystander", "vision", model="v", input_tokens=9)
        result = db.import_sessions([
            {"id": "valid", "messages": [{"role": "user", "content": "must roll back"}]},
            {"id": "invalid", "model_usage": usage},
        ])
        assert result["ok"] is False
        assert result["imported"] == 0
        assert db.get_session("valid") is None
        assert db.get_session("invalid") is None
        assert db.auxiliary_usage_by_task("bystander")["vision"]["input_tokens"] == 9
