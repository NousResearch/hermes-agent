"""No-network turn-loop coverage for a provider-minted oversized checkpoint."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.encrypted_content import MAX_ENCRYPTED_CONTENT_CHARS


@pytest.mark.parametrize("response_kind", ["checkpoint_only", "tool_call", "final_answer"])
def test_new_oversized_checkpoint_preserves_response_and_stops_before_replay(monkeypatch, response_kind):
    import run_agent

    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [{
        "type": "function", "function": {"name": "terminal", "description": "Synthetic tool.",
                                              "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model="gpt-5.6", provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", api_key="synthetic-unused",
        quiet_mode=True, max_iterations=4, skip_context_files=True, skip_memory=True,
        skip_background_review=True, session_db=None,
    )
    agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    agent.compression_checkpoint_required = False
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._dump_api_request_debug = MagicMock()
    agent._try_activate_fallback = MagicMock(return_value=False)
    agent._disable_codex_reasoning_replay = MagicMock()
    agent._compress_context = MagicMock(side_effect=AssertionError("Must not rewrite oversized opaque context"))

    blob = "x" * (MAX_ENCRYPTED_CONTENT_CHARS + 1)
    output = [SimpleNamespace(type="compaction", encrypted_content=blob)]
    if response_kind == "tool_call":
        output.append(SimpleNamespace(type="function_call", status="completed", id="fc_synthetic",
                                      call_id="call_synthetic", name="terminal", arguments="{}"))
    elif response_kind == "final_answer":
        output.append(SimpleNamespace(type="message", role="assistant", status="completed",
                                      content=[SimpleNamespace(type="output_text", text="Synthetic answer.")]))
    response = SimpleNamespace(output=output, status="completed", model="gpt-5.6",
                               usage=SimpleNamespace(input_tokens=9, output_tokens=1, total_tokens=10))
    calls = []

    def synthetic_call(kwargs):
        calls.append(kwargs)
        assert len(calls) == 1, "Oversized checkpoint must never trigger another provider call"
        assert "context_management" in kwargs
        # Even if stale usage or a plugin asks to compress, an oversized checkpoint
        # must stop first. No auxiliary model or local rewrite may substitute for it.
        agent.context_compressor.should_compress = lambda *a, **kw: True
        agent.context_compressor.should_defer_preflight_to_real_usage = lambda *a, **kw: False
        return response

    agent._interruptible_api_call = synthetic_call
    # Execute no real tools. The real tool-round code still appends/pairs the result.
    monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: '{"result": "Synthetic tool evidence."}')
    result = agent.run_conversation("Complete the synthetic task.")
    if response_kind == "final_answer":
        assert result["completed"] is True
        assert result["final_response"] == "Synthetic answer."
        # Eligible follow-ups must still stop before compression or replay.
        history = result["messages"]
        agent._persist_session.reset_mock()
        result = agent.run_conversation("Complete the synthetic task.", conversation_history=history)
        assert result["messages"] is history
        assert result["api_calls"] == 0
        # Repeating the old prompt must not label that historical row as this turn.
        assert "current_turn_user_idx" not in result
        agent._persist_session.assert_not_called()
    assert result["failed"] is True
    assert result["failure_reason"] == "encrypted_content_too_large"
    assert result["failure_retryable"] is False
    assert len(calls) == 1
    carriers = [item for msg in result["messages"] for item in msg.get("codex_reasoning_items", [])]
    assert any(item["encrypted_content"] is blob for item in carriers)
    assert any(msg.get("content") == "Complete the synthetic task." for msg in result["messages"])
    if response_kind == "tool_call":
        tool_rows = [msg for msg in result["messages"] if msg.get("role") == "tool"]
        assert len(tool_rows) == 1
        assert tool_rows[0]["tool_call_id"] == "call_synthetic"
        assert "Synthetic tool evidence." in tool_rows[0]["content"]
        assert any(tc["id"] == "call_synthetic" for msg in result["messages"] for tc in msg.get("tool_calls", []))
    agent._disable_codex_reasoning_replay.assert_not_called()
    agent._try_activate_fallback.assert_not_called()
    agent._dump_api_request_debug.assert_not_called()
    agent._compress_context.assert_not_called()


@pytest.mark.parametrize("model,native,capable,replay,issuer", [
    ("gpt-6-astra-900k", False, False, True, "codex_backend"),
    ("gpt-5.6", False, True, True, "codex_backend"),
    ("gpt-5.6", True, False, True, "codex_backend"),
    ("gpt-5.6", True, True, False, "codex_backend"),
    ("gpt-5.6", True, True, True, "openai"),
])
def test_omitted_oversized_checkpoint_allows_complete_history_continuation(
    monkeypatch, model, native, capable, replay, issuer,
):
    import run_agent

    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model=model, provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", api_key="synthetic-unused",
        quiet_mode=True, max_iterations=2, skip_context_files=True, skip_memory=True,
        skip_background_review=True, session_db=None,
    )
    agent.codex_responses_native_compaction = native
    agent.runtime_capabilities = {"native_compaction": capable}
    agent._codex_reasoning_replay_enabled = replay
    agent.compression_checkpoint_required = False
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = MagicMock()
    agent._save_trajectory = MagicMock()
    agent.context_compressor.should_compress = lambda *a, **kw: False
    agent.context_compressor.should_defer_preflight_to_real_usage = lambda *a, **kw: False
    blob = "x" * 23_849_740
    history = [
        {"role": "user", "content": "Synthetic complete task: remember evidence A."},
        {"role": "assistant", "content": "I will check.",
         "codex_reasoning_items": [{"type": "compaction", "encrypted_content": blob,
                                    "_issuer_kind": issuer}],
         "tool_calls": [{"id": "call_probe", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_probe", "content": "Synthetic evidence A."},
        {"role": "assistant", "content": "Evidence A is retained."},
    ]
    calls = []

    def synthetic_call(kwargs):
        calls.append(kwargs)
        wire = kwargs["input"]
        assert not any(item.get("type") == "compaction" for item in wire)
        assert any(item.get("content") == history[0]["content"] for item in wire)
        assert any(item.get("type") == "function_call" and item.get("call_id") == "call_probe" for item in wire)
        assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call_probe"
                   and item.get("output") == "Synthetic evidence A." for item in wire)
        return SimpleNamespace(
            status="completed", model=model,
            output=[SimpleNamespace(type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text="Synthetic continuation succeeded.")])],
            usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
        )

    agent._interruptible_api_call = synthetic_call
    result = agent.run_conversation("Continue using evidence A.", conversation_history=history)
    assert result["completed"] is True
    assert len(calls) == 1
    assert result["final_response"] == "Synthetic continuation succeeded."
    assert history[1]["codex_reasoning_items"][0]["encrypted_content"] is blob
    assert any(msg.get("tool_call_id") == "call_probe" for msg in result["messages"])
