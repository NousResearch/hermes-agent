import json

from agent.token_optimization_metrics import (
    build_tool_result_metric,
    estimate_tokens,
    metrics_path,
    record_tool_result_metric,
)


def test_metrics_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TOKEN_OPTIMIZATION_METRICS", raising=False)

    record_tool_result_metric(
        tool_name="terminal",
        function_args={"command": "echo secret"},
        result='{"output":"secret"}',
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        duration_ms=3,
    )

    assert not metrics_path().exists()


def test_truthy_gate_writes_profile_aware_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_TOKEN_OPTIMIZATION_METRICS", "yes")

    record_tool_result_metric(
        tool_name="terminal",
        function_args={"command": "printf", "token": "sk-secret-value"},
        result='{"output":"SECRET_PAYLOAD","truncated":true}',
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        duration_ms=7,
    )

    path = tmp_path / "token-optimization" / "metrics.jsonl"
    assert metrics_path() == path
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1

    raw_row = rows[0]
    row = json.loads(raw_row)
    assert row["schema_version"] == 2
    assert row["event_type"] == "tool_result"
    assert row["session_id"] == "session-1"
    assert row["task_id"] == "task-1"
    assert row["tool_call_id"] == "call-1"
    assert row["tool_name"] == "terminal"
    assert row["duration_ms"] == 7
    assert row["arg_keys"] == ["command", "token"]
    assert row["is_json"] is True
    assert row["top_level_type"] == "object"
    assert row["has_error_key"] is False
    assert row["has_non_empty_error"] is False
    assert row["truncated_by_tool"] is True
    assert row["result_class"] == "truncated"
    assert "sha256" in row

    assert "SECRET_PAYLOAD" not in raw_row
    assert "sk-secret-value" not in raw_row
    assert "printf" not in raw_row


def test_build_metric_records_shape_and_identifier_not_payload():
    result = '{"error":"raw failure text","data":[1,2,3]}'

    row = build_tool_result_metric(
        tool_name="read_file",
        function_args={"path": "/tmp/secret.txt"},
        result=result,
        task_id=None,
        session_id=None,
        tool_call_id=None,
        duration_ms=0,
    )

    assert row["result_char_count"] == len(result)
    assert row["result_byte_count"] == len(result.encode("utf-8"))
    assert row["estimated_tokens"] == estimate_tokens(len(result))
    assert row["top_level_type"] == "object"
    assert row["has_error_key"] is True
    assert row["has_non_empty_error"] is True
    assert row["result_class"] == "runtime_error"
    assert row["arg_keys"] == ["path"]
    assert len(row["sha256"]) == 64
    # sha256 is an identifier for equality/correlation, not anonymization.
    assert result not in json.dumps(row)
    assert "/tmp/secret.txt" not in json.dumps(row)


def test_invalid_json_shape_is_metadata_only():
    row = build_tool_result_metric(
        tool_name="terminal",
        function_args=["not", "a", "dict"],
        result="plain text result",
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        duration_ms=None,
    )

    assert row["is_json"] is False
    assert row["top_level_type"] == "invalid_json"
    assert row["has_error_key"] is False
    assert row["has_non_empty_error"] is False
    assert row["truncated_by_tool"] is False
    assert row["result_class"] == "invalid_json"
    assert row["arg_keys"] == []
    assert row["duration_ms"] is None


def test_metrics_fail_open(monkeypatch, tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file blocks mkdir", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(blocked_parent))
    monkeypatch.setenv("HERMES_TOKEN_OPTIMIZATION_METRICS", "on")

    record_tool_result_metric(
        tool_name="terminal",
        function_args={},
        result="{}",
        task_id=None,
        session_id=None,
        tool_call_id=None,
        duration_ms=1,
    )


def test_null_error_field_is_not_counted_as_actual_error():
    row = build_tool_result_metric(
        tool_name="terminal",
        function_args={"command": "true"},
        result='{"output":"ok","error":null}',
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        duration_ms=2,
    )

    assert row["has_error_key"] is True
    assert row["has_non_empty_error"] is False
    assert row["result_class"] == "success"
