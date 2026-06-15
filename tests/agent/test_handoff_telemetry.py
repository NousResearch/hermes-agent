import json
from types import SimpleNamespace

from agent.handoff_telemetry import (
    build_handoff_telemetry_event,
    record_handoff_telemetry,
    telemetry_log_path,
)
from tools.delegate_tool import _run_single_child


def test_handoff_telemetry_writes_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    event = build_handoff_telemetry_event(
        trace_id="handoff-test",
        subagent_id="sa-0-test",
        parent_session_id="sess-parent",
        parent_task_id="task-parent",
        parent_subagent_id="sa-parent",
        task_index=0,
        status="completed",
        exit_reason="completed",
        model="anthropic/claude-sonnet-4.6",
        provider="openrouter",
        api_mode="chat_completions",
        api_calls=2,
        duration_seconds=1.23456,
        input_tokens=100,
        output_tokens=40,
        cache_read_tokens=25,
        cache_write_tokens=5,
        reasoning_tokens=7,
        estimated_cost_usd=0.012345678,
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        result={"completed": True, "summary_chars": 12},
    )

    path = record_handoff_telemetry(event)

    assert path is not None
    assert path == telemetry_log_path()
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["trace_id"] == "handoff-test"
    assert payload["parent_subagent_id"] == "sa-parent"
    assert payload["status"] == "completed"
    assert payload["provider"] == "openrouter"
    assert payload["api_mode"] == "chat_completions"
    assert payload["tokens"] == {
        "input": 100,
        "output": 40,
        "cache_read": 25,
        "cache_write": 5,
        "reasoning": 7,
        "prompt": 130,
        "total": 170,
    }
    assert payload["cost"]["estimated_usd"] == 0.01234568


def test_run_single_child_records_handoff_telemetry(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class FakeChild:
        _subagent_id = "sa-0-unit"
        _handoff_trace_id = "handoff-unit"
        _parent_subagent_id = "sa-parent-unit"
        _delegate_depth = 1
        _delegate_role = "leaf"
        _delegate_saved_tool_names = []
        tool_progress_callback = None
        model = "gpt-4o-mini"
        provider = "openai"
        api_mode = "chat_completions"
        session_input_tokens = 11
        session_output_tokens = 13
        session_cache_read_tokens = 5
        session_cache_write_tokens = 7
        session_reasoning_tokens = 2
        session_prompt_tokens = 23
        session_completion_tokens = 13
        session_total_tokens = 36
        session_estimated_cost_usd = 0.00042
        session_cost_status = "estimated"
        session_cost_source = "official_docs_snapshot"

        def run_conversation(self, user_message, task_id):
            assert user_message == "summarize telemetry"
            assert task_id == self._subagent_id
            return {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
                "turn_exit_reason": "text_response(finish_reason=stop)",
                "messages": [],
            }

        def close(self):
            pass

    parent = SimpleNamespace(session_id="parent-session", _current_task_id="parent-task")

    result = _run_single_child(0, "summarize telemetry", child=FakeChild(), parent_agent=parent)

    assert result["status"] == "completed"
    path = telemetry_log_path()
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "agent_handoff"
    assert payload["trace_id"] == "handoff-unit"
    assert payload["subagent_id"] == "sa-0-unit"
    assert payload["parent_session_id"] == "parent-session"
    assert payload["parent_task_id"] == "parent-task"
    assert payload["parent_subagent_id"] == "sa-parent-unit"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["provider"] == "openai"
    assert payload["api_mode"] == "chat_completions"
    assert payload["exit_reason"] == "text_response(finish_reason=stop)"
    assert payload["api_calls"] == 1
    assert payload["tokens"]["input"] == 11
    assert payload["tokens"]["output"] == 13
    assert payload["tokens"]["cache_read"] == 5
    assert payload["tokens"]["cache_write"] == 7
    assert payload["tokens"]["reasoning"] == 2
    assert payload["tokens"]["prompt"] == 23
    assert payload["tokens"]["total"] == 36
    assert payload["cost"]["estimated_usd"] == 0.00042
    assert payload["cost"]["status"] == "estimated"
    assert payload["cost"]["source"] == "official_docs_snapshot"
    assert payload["result"]["completed"] is True


def test_handoff_telemetry_redacts_result_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    event = build_handoff_telemetry_event(
        trace_id="handoff-redact",
        subagent_id="sa-redact",
        parent_session_id="sess-parent",
        parent_task_id="task-parent",
        parent_subagent_id=None,
        task_index=0,
        status="error",
        exit_reason="error",
        model="test-model",
        provider="test-provider",
        api_mode="chat_completions",
        api_calls=0,
        duration_seconds=0,
        result={
            "completed": False,
            "error": "contains sensitive markers",
            "nested": {"authorization": "Bearer token"},
            "values": ["safe", "value"],
        },
    )
    path = record_handoff_telemetry(event)

    assert path is not None
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw.strip())
    assert payload["result"]["completed"] is False
    assert payload["result"]["error"] == "contains sensitive markers"
    assert payload["result"]["nested"]["authorization"] == "Bearer token"

def test_record_handoff_telemetry_ignores_write_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class BrokenPath:
        parent = tmp_path

        def open(self, *_args, **_kwargs):
            raise OSError("disk full")

    monkeypatch.setattr("agent.handoff_telemetry.telemetry_log_path", lambda: BrokenPath())

    assert record_handoff_telemetry({"trace_id": "handoff-fail"}) is None
