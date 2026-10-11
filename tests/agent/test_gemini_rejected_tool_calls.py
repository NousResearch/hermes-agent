"""Regression for #83869: rejected native tool calls must not become successful turns.

Salvaged from closed PR #129274; drive the native HTTP adapter and public turn loop.
"""

import json
from unittest.mock import patch

import pytest

from tests.agent.test_run_agent import (
    TestRetryExhaustion as _RetrySetup,
    _mock_response,
    _mock_plugin_discovery as _mock_plugin_discovery,
    agent as agent,
)

def test_malformed_function_call_surfaced_not_retried(agent):
    """Gemini MALFORMED_FUNCTION_CALL must fail the turn, not succeed empty.

    Regression #83869: native Gemini returns HTTP 200 with
    ``finish_reason=malformed_function_call`` and empty content. That used
    to fall through to empty-content retries and be saved as a successful
    ``(empty)`` trajectory. Surface it as a failed, non-completed turn
    without treating it as a safety refusal.
    """
    _RetrySetup()._setup_agent(agent)
    malformed_resp = _mock_response(
        content=None, finish_reason="malformed_function_call",
    )
    agent.client.chat.completions.create.return_value = malformed_resp
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("please call a tool")
    assert result.get("completed") is False
    assert result.get("failed") is True
    assert "malformed_function_call" in result.get("error", "")
    assert result.get("turn_exit_reason") == "malformed_function_call"
    final = result.get("final_response") or ""
    assert "malformed function call" in final.lower()
    assert "not a safety refusal" in final.lower()
    assert "content_policy_blocked" not in result.get("error", "")
    assert "(empty)" not in final
    # Deterministic for the unchanged prompt — not retried as empty.
    assert agent.client.chat.completions.create.call_count == 1

def test_malformed_function_call_tries_fallback_once(agent):
    """A pending fallback may recover from a Gemini malformed tool call."""
    _RetrySetup()._setup_agent(agent)
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.7"},
    ]
    agent._fallback_index = 0
    malformed_resp = _mock_response(
        content=None, finish_reason="malformed_function_call",
    )
    fallback_resp = _mock_response(
        content="Recovered on fallback", finish_reason="stop",
    )
    agent.client.chat.completions.create.side_effect = [
        malformed_resp, fallback_resp,
    ]

    def _fake_activate(reason=None):
        agent._fallback_index = len(agent._fallback_chain)
        return True

    with (
        patch.object(agent, "_try_activate_fallback", side_effect=_fake_activate) as mock_fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("please call a tool")

    assert result.get("completed") is True
    assert result.get("final_response") == "Recovered on fallback"
    mock_fallback.assert_called_once_with()
    assert agent.client.chat.completions.create.call_count == 2

@pytest.mark.parametrize("raw_reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
def test_native_malformed_turn_is_not_success(agent, raw_reason):
    from agent.gemini_native_adapter import translate_gemini_response

    _RetrySetup()._setup_agent(agent)
    agent.client.chat.completions.create.return_value = translate_gemini_response(
        {"candidates": [{"content": {"parts": []}, "finishReason": raw_reason}]},
        model="test/model",
    )
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("please call a tool")
    assert result.get("completed") is False
    assert result.get("failed") is True
    assert result.get("turn_exit_reason") == "malformed_function_call"
    from agent.error_surface import build_error_surface_from_result

    surface = build_error_surface_from_result(result)
    assert surface["code"] == "format_error"
    assert surface["retryable"] is False
    assert agent.client.chat.completions.create.call_count == 1

@pytest.mark.parametrize("raw_reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_tool", [False, True])
def test_native_malformed_turn_persists_failed_trajectory(agent, tmp_path, monkeypatch, raw_reason, stream, with_tool):
    import httpx
    from agent.gemini_native_adapter import GeminiNativeClient

    _RetrySetup()._setup_agent(agent)
    monkeypatch.chdir(tmp_path)
    agent.save_trajectories = True
    agent.model = "gemini-test"
    calls = []
    payload = {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10}, "candidates": [{
        "content": {"parts": [{"functionCall": {"name": "web_search", "args": {}}}] if with_tool else []},
        "finishReason": raw_reason,
    }]}

    def respond(request):
        calls.append(request)
        if stream:
            return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    client = GeminiNativeClient(api_key="fixture-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    agent.client = client
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
    monkeypatch.setattr("agent.turn_api_call._should_stream", lambda agent: stream)
    def forbid_network(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    if stream:
        agent.stream_delta_callback = lambda text: None
    try:
        with patch.object(agent, "_execute_tool_calls") as execute, patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation("please call a tool")
        assert result.get("failed") is True
        assert result.get("completed") is False
        assert result["failure_reason"] == "format_error"
        assert result["failure_retryable"] is False
        assert all(row.get("content") != "(empty)" for row in result["messages"])
        assert len(calls) == 1
        execute.assert_not_called()
        assert not (tmp_path / "trajectory_samples.jsonl").exists()
        rows = [json.loads(line) for line in (tmp_path / "failed_trajectories.jsonl").read_text(encoding="utf8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["completed"] is False
        assert agent.session_prompt_tokens == 7
        assert agent.session_completion_tokens == 3
    finally:
        client.close()


@pytest.mark.parametrize(
    "raw_reason",
    ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"),
)
def test_translate_malformed_function_call_is_not_stop(raw_reason):
    """Gemini tool-call-shape rejections must not fall through to stop.

    An empty MALFORMED_FUNCTION_CALL candidate used to map to finish_reason
    ``stop``, which the conversation loop treated as a successful empty
    reply (#83869). These reasons are also not safety refusals.
    """
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {"parts": []},
                "finishReason": raw_reason,
            }
        ],
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    choice = response.choices[0]
    assert choice.finish_reason == "malformed_function_call"
    assert choice.finish_reason != "stop"
    assert choice.finish_reason != "content_filter"
    assert not choice.message.tool_calls


@pytest.mark.parametrize(
    "raw_reason",
    ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"),
)
def test_stream_malformed_function_call_is_not_stop(raw_reason):
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": []},
                "finishReason": raw_reason,
            }
        ],
    }

    chunks = translate_stream_event(
        event, model="gemini-2.5-flash", tool_call_indices={}
    )
    assert chunks
    assert chunks[-1].choices[0].finish_reason == "malformed_function_call"
    assert chunks[-1].choices[0].finish_reason != "stop"
    assert chunks[-1].choices[0].finish_reason != "content_filter"
