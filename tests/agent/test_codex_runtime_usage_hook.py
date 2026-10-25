"""The codex_app_server runtime never reaches ``normalize_model_response``, so its one provider call
must fire ``post_api_request`` itself — with the same ``cost`` it records in the turn's
``api_call_records`` — or a usage/cost observer sees nothing for Codex turns."""

from types import SimpleNamespace
from unittest.mock import patch


def _agent():
    return SimpleNamespace(
        _usage_anchor=None, session_api_calls=0, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
        context_compressor=None, event_callback=None, _session_db=None, model="codex-test-model",
        provider="openai", base_url=None, session_id="s1", platform="cli", api_mode="codex_app_server",
        _current_turn_id="turn-1", _turn_api_call_records=[],
    )


def test_codex_turn_fires_post_api_request_with_its_recorded_cost():
    from agent.codex_runtime import _record_codex_app_server_usage

    agent = _agent()
    turn = SimpleNamespace(
        token_usage_last={"inputTokens": 1000, "cachedInputTokens": 200, "outputTokens": 50,
                          "reasoningOutputTokens": 0, "totalTokens": 1050},
        model_context_window=None, final_text="done", interrupted=False, error=None,
    )
    calls = []
    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "post_api_request"),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=lambda name, **kw: calls.append((name, kw)) or []),
    ):
        result = _record_codex_app_server_usage(agent, turn, messages=[], task_id="task-1")

    [(name, payload)] = calls
    [record] = agent._turn_api_call_records
    assert name == "post_api_request"
    assert (payload["task_id"], payload["turn_id"], payload["session_id"]) == ("task-1", "turn-1", "s1")
    assert payload["cost"] == {k: record[k] for k in ("estimated_cost_usd", "cost_status", "cost_source")}
    assert payload["cost"]["cost_status"] == result["cost_status"]
    assert payload["usage"]["total_tokens"] == record["total_tokens"] == 1050
    assert record["cache_read_tokens"] == 200
