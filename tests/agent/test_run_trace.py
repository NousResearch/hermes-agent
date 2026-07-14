"""Tests for metadata-only run trace persistence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hermes_constants import get_hermes_home
from agent.run_trace import (
    RunTrace,
    RunTraceToolCall,
    append_run_trace,
    finish_trace_for_turn,
    record_tool_batch,
    start_trace_for_turn,
    trace_enabled,
)


def _secret() -> str:
    return "sk-test_" + "a" * 32


def test_trace_disabled_by_default_and_enabled_by_config():
    assert trace_enabled(config={}) is False
    assert trace_enabled(config={"observability": {"run_trace_enabled": False}}) is False
    assert trace_enabled(config={"observability": {"run_trace_enabled": True}}) is True
    assert trace_enabled(config={"observability": {"run_trace_enabled": "yes"}}) is True


def test_run_trace_metadata_only_omits_raw_prompt_args_and_tool_output():
    trace = RunTrace(
        run_id="run_test",
        session_id="session_1",
        turn_id="turn_1",
        task_id="task_1",
        model="test-model",
        provider="test-provider",
        source="cli",
    )
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({"command": "cat /tmp/private_prompt.txt", "token": _secret()}),
        ),
    )

    record_tool_batch(trace, [tool_call], status="requested")
    trace.tool_calls.append(
        RunTraceToolCall(
            name="terminal",
            tool_call_id="call_2",
            status="error",
            error_type="RuntimeError",
            error_message=f"tool failed with {_secret()}",
        )
    )
    data = trace.to_dict()
    encoded = json.dumps(data, ensure_ascii=False)

    assert "private_prompt" not in encoded
    assert "cat /tmp" not in encoded
    assert "command" not in encoded
    assert "arguments" not in encoded
    assert "result" not in encoded
    assert "output" not in encoded
    assert _secret() not in encoded
    tool_data = data["tool_calls"][0]
    assert tool_data["name"] == "terminal"
    assert tool_data["tool_call_id"].startswith("sha256:")
    assert tool_data["status"] == "requested"
    assert tool_data["duration_ms"] is None
    assert tool_data["error_type"] == ""
    assert tool_data["error_message"] == ""
    assert "call_1" not in encoded


def test_run_trace_redacts_secret_like_metadata_values():
    trace = RunTrace(
        run_id="run_test",
        session_id=f"session {_secret()}",
        turn_id="turn_1",
        task_id="task_1",
        model="test-model",
        provider="test-provider",
        source="cli",
        exit_reason=f"error_near_max_iterations(private prompt {_secret()})",
        tool_calls=[
            RunTraceToolCall(
                name="terminal",
                tool_call_id="call_1",
                status="error",
                error_type="RuntimeError",
                error_message=f"OPENAI_API_KEY={_secret()}",
            )
        ],
    )

    data = trace.to_dict()
    encoded = json.dumps(data, ensure_ascii=False)

    assert _secret() not in encoded
    assert "private prompt" not in encoded
    assert "OPENAI_API_KEY=" not in encoded
    assert data["exit_reason"] == "error_near_max_iterations"
    assert data["tool_calls"][0]["error_message"] == "other"


def test_run_trace_hashes_free_text_identifiers():
    trace = RunTrace(
        run_id="run_test",
        session_id="session_1",
        turn_id="turn_1",
        task_id="customer asked about a private acquisition plan",
    )

    data = trace.to_dict()
    encoded = json.dumps(data, ensure_ascii=False)

    assert "private acquisition" not in encoded
    assert data["task_id"].startswith("sha256:")


def test_run_trace_hashes_slug_and_underscore_identifiers():
    identifiers = (
        "customer-private-acquisition-plan",
        "customer_private_acquisition_plan",
    )

    for identifier in identifiers:
        trace = RunTrace(
            run_id="run_test",
            session_id="session_1",
            turn_id=f"session_1:{identifier}:deadbeef",
            task_id=identifier,
        )

        data = trace.to_dict()
        encoded = json.dumps(data, ensure_ascii=False)

        assert identifier not in encoded
        assert data["task_id"].startswith("sha256:")
        assert data["turn_id"].startswith("sha256:")


def test_append_run_trace_persists_jsonl_under_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    trace = RunTrace(run_id="run_test", session_id="sess", turn_id="turn", task_id="task")
    finish_trace_for_turn(
        trace,
        status="completed",
        exit_reason="text_response(finish_reason=stop)",
        api_call_count=1,
        config={"observability": {"run_trace_enabled": True}},
    )

    path = get_hermes_home() / "run_traces" / "run_traces.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["schema_version"] == "hermes_run_trace_v1"
    assert entry["run_id"].startswith("sha256:")
    assert "run_test" not in json.dumps(entry, ensure_ascii=False)
    assert entry["status"] == "completed"
    assert entry["api_call_count"] == 1


def test_append_run_trace_disabled_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    trace = RunTrace(run_id="run_test")

    wrote = append_run_trace(trace, config={"observability": {"run_trace_enabled": False}})

    assert wrote is False
    assert not (get_hermes_home() / "run_traces" / "run_traces.jsonl").exists()


def test_append_run_trace_write_failure_is_non_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    blocked_parent = get_hermes_home() / "run_traces"
    blocked_parent.parent.mkdir(parents=True, exist_ok=True)
    blocked_parent.write_text("not a directory", encoding="utf-8")
    trace = RunTrace(run_id="run_test")

    wrote = append_run_trace(trace, config={"observability": {"run_trace_enabled": True}})

    assert wrote is False


def test_start_trace_for_turn_builds_metadata_from_agent_when_enabled():
    agent = SimpleNamespace(
        session_id="session_1",
        model="test-model",
        provider="test-provider",
        platform="gateway",
    )

    trace = start_trace_for_turn(
        agent,
        turn_id="turn_1",
        task_id="task_1",
        config={"observability": {"run_trace_enabled": True}},
    )

    assert trace is not None
    assert trace.session_id == "session_1"
    assert trace.turn_id == "turn_1"
    assert trace.task_id == "task_1"
    assert trace.model == "test-model"
    assert trace.provider == "test-provider"
    assert trace.source == "gateway"


def test_start_trace_for_turn_returns_none_when_disabled():
    agent = SimpleNamespace(session_id="session_1")

    assert start_trace_for_turn(
        agent,
        turn_id="turn_1",
        task_id="task_1",
        config={"observability": {"run_trace_enabled": False}},
    ) is None
