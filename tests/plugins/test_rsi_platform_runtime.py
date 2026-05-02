"""Tests for the RSI native workflow runtime plugin."""

import json

import pytest

from plugins import rsi_platform_runtime as runtime


def _write_context(path, *, execution_id="hexec-1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "execution_id": execution_id,
                "operation_id": "op-1",
                "trace_id": "trace-1",
                "workflow_id": "wf-1",
                "hermes_session_id": "rsi-prod-conversation-1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _install_runtime_env(monkeypatch, context_path, envelope_path, *, execution_id="hexec-1"):
    monkeypatch.setenv("RSI_RUNTIME_CONTEXT_PATH", str(context_path))
    monkeypatch.setenv("RSI_RUNTIME_ENVELOPE_PATH", str(envelope_path))
    monkeypatch.setenv("RSI_EXECUTION_ID", execution_id)
    monkeypatch.setenv("RSI_OPERATION_ID", "op-1")
    monkeypatch.setenv("RSI_TRACE_ID", "trace-1")
    monkeypatch.setenv("RSI_WORKFLOW_ID", "wf-1")


def test_platform_runtime_writes_execution_scoped_redacted_envelope(tmp_path, monkeypatch):
    context_path = tmp_path / "rsi_runtime" / "context" / "executions" / "hexec-1.json"
    envelope_path = tmp_path / "rsi_runtime" / "envelopes" / "hexec-1.json"
    _write_context(context_path)
    _install_runtime_env(monkeypatch, context_path, envelope_path)
    runtime._STATE_BY_EXECUTION.clear()

    runtime.post_tool_call(
        tool_name="send_message",
        args={"message": "Final reply", "api_key": "should-not-leak"},
        result=json.dumps(
            {
                "success": True,
                "platform": "slack",
                "chat_id": "C123456789",
                "thread_id": "171000001.000100",
                "message_id": "171000001.000200",
                "message_link": "https://slack.example/messages/171000001.000200",
                "idempotency_key": "idem-send-1",
            },
            sort_keys=True,
        ),
        tool_call_id="call-send-1",
        duration_ms=12,
    )
    runtime.post_llm_call("Final reply with sk-secret-openrouter")

    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["contract_version"] == "execution-envelope/v1"
    assert envelope["producer"] == "rsi_platform_runtime"
    assert envelope["execution_id"] == "hexec-1"
    assert envelope["operation_id"] == "op-1"
    assert envelope["trace_id"] == "trace-1"
    assert envelope["workflow_id"] == "wf-1"
    assert envelope["final_response"] == "Final reply with [redacted]"
    assert envelope["ledger_events"][0]["kind"] == "tool.call.completed"
    assert envelope["ledger_events"][0]["payload"]["args"]["api_key"] == "[redacted]"
    assert envelope["deliveries"][0]["send_status"] == "posted"
    assert envelope["deliveries"][0]["idempotency_key"] == "idem-send-1"
    assert not list(envelope_path.parent.glob("*.tmp"))


def test_platform_runtime_rejects_mismatched_execution_context(tmp_path, monkeypatch):
    context_path = tmp_path / "rsi_runtime" / "context" / "executions" / "hexec-1.json"
    envelope_path = tmp_path / "rsi_runtime" / "envelopes" / "hexec-1.json"
    _write_context(context_path, execution_id="hexec-other")
    _install_runtime_env(monkeypatch, context_path, envelope_path)
    runtime._STATE_BY_EXECUTION.clear()

    with pytest.raises(RuntimeError, match="execution_id mismatch"):
        runtime.post_llm_call("Final reply")

    assert not envelope_path.exists()
