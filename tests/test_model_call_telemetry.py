import json
from pathlib import Path
from types import SimpleNamespace

from agent.chat_completion_helpers import _emit_model_call_event


def _last_event(tmp_path: Path) -> dict:
    event_files = sorted((tmp_path / "ops" / "events").glob("*.jsonl"))
    assert event_files
    lines = event_files[-1].read_text(encoding="utf-8").splitlines()
    assert lines
    return json.loads(lines[-1])


def test_model_call_event_hashes_provider_model_and_keeps_usage_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(provider="RAW_PROVIDER_DO_NOT_STORE", model="RAW_MODEL_DO_NOT_STORE", api_mode="chat_completions")
    api_kwargs = {
        "model": "RAW_MODEL_DO_NOT_STORE",
        "messages": [{"role": "user", "content": "RAW_PROMPT_DO_NOT_STORE"}],
        "tools": [{"name": "RAW_TOOL_SCHEMA_DO_NOT_STORE"}],
    }
    response = SimpleNamespace(
        id="RAW_RESPONSE_ID_DO_NOT_STORE",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )

    _emit_model_call_event(agent, api_kwargs, response=response, duration_ms=123, streaming=True)

    event = _last_event(tmp_path)
    assert event["event_type"] == "model_call"
    assert event["source"] == "agent.chat_completion_helpers"
    assert event["status"] == "ok"
    payload = event["payload"]
    assert payload["provider_hash"]
    assert payload["model_hash"]
    assert payload["api_mode"] == "chat_completions"
    assert payload["streaming"] is True
    assert payload["partial_response"] is False
    assert payload["duration_ms"] == 123
    assert payload["message_count"] == 1
    assert payload["tool_count"] == 1
    assert payload["context_units"] == 11
    assert payload["output_units"] == 7
    assert payload["total_units"] == 18
    assert payload["response_id_hash"]
    assert payload["error_fingerprint"] is None

    raw_file = next((tmp_path / "ops" / "events").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "RAW_PROVIDER_DO_NOT_STORE" not in raw_file
    assert "RAW_MODEL_DO_NOT_STORE" not in raw_file
    assert "RAW_PROMPT_DO_NOT_STORE" not in raw_file
    assert "RAW_TOOL_SCHEMA_DO_NOT_STORE" not in raw_file
    assert "RAW_RESPONSE_ID_DO_NOT_STORE" not in raw_file


def test_model_call_event_groups_errors_without_raw_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(provider="RAW_PROVIDER_DO_NOT_STORE", model="RAW_MODEL_DO_NOT_STORE", api_mode="anthropic_messages")
    api_kwargs = {"model": "RAW_MODEL_DO_NOT_STORE", "messages": []}
    error = RuntimeError("RAW_ERROR_DO_NOT_STORE")

    _emit_model_call_event(agent, api_kwargs, error=error, duration_ms=5, streaming=False)

    event = _last_event(tmp_path)
    assert event["event_type"] == "model_call"
    assert event["status"] == "error"
    payload = event["payload"]
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_fingerprint"]
    assert payload["streaming"] is False
    raw_file = next((tmp_path / "ops" / "events").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "RAW_ERROR_DO_NOT_STORE" not in raw_file
    assert "RAW_PROVIDER_DO_NOT_STORE" not in raw_file
    assert "RAW_MODEL_DO_NOT_STORE" not in raw_file
