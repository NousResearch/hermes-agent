"""Exercise response capture, adaptive tail planning and the SDK without provider traffic."""
from __future__ import annotations

import copy
import inspect
import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from agent.context_compressor import ContextCompressor
from agent.turn_response_check import check_api_response
from agent.turn_retry_state import TurnRetryState


def _history():
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row {i} " + "detail " * 250}
            for i in range(40)]


def test_successful_response_captures_the_actual_cached_request():
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://api.anthropic.com", model="claude-opus-4-7",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.anthropic_native_compaction = True
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._use_prompt_caching = True
    agent._cache_ttl = "5m"
    history = _history()
    request = {"model": agent.model, "max_tokens": 1024, "messages": [{"role": "user", "content": "wire form"}],
               "system": [{"type": "text", "text": "cached prompt", "cache_control": {"type": "ephemeral"}}]}
    response = SimpleNamespace(model=agent.model, stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="answer")], usage=None)
    with patch("agent.turn_response_check.record_response_usage",
               return_value=SimpleNamespace(compression_attempts=0, rearmed=False)):
        verdict = check_api_response(
            agent, response=response, _retry=TurnRetryState(), thinking_spinner=None, messages=history,
            api_messages=request["messages"], api_kwargs=request, active_system_prompt="cached prompt",
            conversation_history=[], finish_reason=None, retry_count=0, max_retries=1, compression_attempts=0,
            max_compression_attempts=1, length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, current_turn_user_idx=38, api_call_count=1,
            api_request_id="native-seed-test", api_start_time=time.time(), effective_task_id=None, turn_id=None,
            _preflight_compression_blocked=False, _last_preflight_pressure=None,
        )
    assert verdict.action == "break"
    seed = getattr(agent, "_anthropic_compaction_seed", None)
    assert seed is not None, "the successful-response stage must capture the cached wire request"
    assert seed.kwargs is request
    assert len(seed.fingerprints) == len(history)


@pytest.mark.parametrize("native_ok", [True, False], ids=["native-sdk", "safe-aux-fallback"])
def test_sdk_native_summary_preserves_adaptive_unseen_tail(native_ok):
    history = _history()
    wire_request = {"model": "claude-opus-4-7", "max_tokens": 1024,
                    "system": [{"type": "text", "text": "cache prompt", "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": "exact prepared wire request"}]}
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        blocks = ([{"type": "compaction", "content": "native SDK handoff"}] if native_ok
                  else [{"type": "text", "text": "ordinary assistant text is not a summary"}])
        return httpx.Response(200, json={"id": "msg_test", "type": "message", "role": "assistant",
                                       "model": wire_request["model"], "content": blocks,
                                       "stop_reason": "compaction" if native_ok else "end_turn",
                                       "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0,
                                           "iterations": [{"input_tokens": 25, "output_tokens": 5,
                                                           "cache_read_input_tokens": 100}]}})

    with anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        agent = SimpleNamespace(
            anthropic_native_compaction=True, compression_enabled=True, api_mode="anthropic_messages",
            base_url="https://api.anthropic.com", model=wire_request["model"], session_id="sdk-compaction-test",
            _anthropic_client=sdk, _use_prompt_caching=True, _cache_ttl="5m",
        )
        # Run the same probe on main without importing missing feature modules: the red
        # baseline must fail an assertion after exercising the existing compressor.
        source = None
        if "native_summary" in inspect.signature(ContextCompressor.compress).parameters:
            from agent.anthropic_native_compaction import native_summary_for, record_compaction_seed

            record_compaction_seed(agent, wire_request, history)
            source = native_summary_for(agent)
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            compressor = ContextCompressor(model=agent.model, quiet_mode=True, config_context_length=100_000)
        compressor.tail_token_budget = 1000
        unseen = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"UNSEEN-{i}"} for i in range(8)]
        messages = history + unseen
        original = copy.deepcopy(messages)
        kwargs: dict[str, Any] = {"force": True}
        if source is not None:
            kwargs["native_summary"] = source
        auxiliary_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="safe auxiliary handoff"), finish_reason="stop")])
        with patch("agent.context_compressor.call_llm", return_value=auxiliary_response) as auxiliary:
            result = compressor.compress(messages, **kwargs)
        assert len(sent) == 1, "a fresh matching cache seed must reach the real SDK exactly once"
        assert sent[0]["messages"] == wire_request["messages"]
        assert sent[0]["system"] == wire_request["system"]
        assert sent[0]["compaction"]["type"] == "summarize"
        assert messages == original
        assert all(any(kept.get("role") == row["role"] and kept.get("content") == row["content"]
                       for kept in result) for row in unseen)
        expected = "native SDK handoff" if native_ok else "safe auxiliary handoff"
        assert any(expected in row.get("content", "") for row in result)
        assert auxiliary.call_count == (0 if native_ok else 1)
        if native_ok:
            assert compressor._last_compression_telemetry["aux_prompt_tokens"] == 125
