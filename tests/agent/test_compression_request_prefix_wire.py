"""Compression reuses the main Responses prefix through the real HTTP boundary."""

import copy
from unittest.mock import patch

import pytest

from agent.conversation_compression_manual import compress_now, parse_compress_args
from agent.auxiliary_client import call_llm
from agent.context_compressor import ContextCompressor
from agent.context_compressor_cache import SummaryPrefix, summary_prefix_scope, summary_prompt_estimate
from agent.model_metadata import estimate_request_tokens_rough
from hermes_cli.config import save_config
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.fakes.providers.openai_responses import FakeResponsesServer, HttpError, Message, Turn


_TOOLS = [{"type": "function", "function": {
    "name": "inspect_file", "description": "Inspect a source file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "pattern": "^.+$"}}, "required": ["path"]},
}}]
_SUMMARY = """## Active Task
Continue the implementation.
## Completed Actions
1. Inspected the source files.
## Active State
Tests are pending.
## Pending Questions
None.
"""


def _make_agent(tmp_path, base_url):
    save_config({
        "model": {"default": "fake-responses-model", "context_length": 128000},
        "compression": {"enabled": True, "threshold": 0.9, "tail_mode": "legacy",
                        "protect_first_n": 2, "protect_last_n": 2},
        "auxiliary": {"compression": {"reuse_prompt_cache": True}, "title_generation": {"enabled": False}},
    })
    db = SessionDB(tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            model="fake-responses-model", provider="custom", base_url=base_url, api_key="test-key",
            api_mode="codex_responses", enabled_toolsets=[], quiet_mode=True,
            skip_memory=True, skip_context_files=True, session_db=db, session_id="cache-prefix-session",
        )
    agent.tools = copy.deepcopy(_TOOLS)
    agent._cached_system_prompt = "You help with a coding project. Preserve the conversation context."
    agent._disable_streaming = True
    agent.reasoning_config = {"enabled": True, "effort": "low"}
    agent.save_trajectories = False
    return agent


@pytest.mark.parametrize("manual", [False, True], ids=["automatic-dispatch", "manual-compress"])
def test_compression_keeps_main_request_prefix(tmp_path, manual):
    with FakeResponsesServer([
        Turn([Message("Source files inspected.")], input_tokens=50000),
        Turn([Message(_SUMMARY)], input_tokens=40000, cached_tokens=38000),
    ], default_text=_SUMMARY) as server:
        agent = _make_agent(tmp_path, server.base_url)
        history = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"project step {i}: " + "source context " * 220} for i in range(60)]
        history[3] = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "inspect-1", "type": "function",
            "function": {"name": "inspect_file", "arguments": '{"path":"main.py"}'},
        }]}
        history[4] = {"role": "tool", "tool_call_id": "inspect-1", "content": "observed source context " * 400}
        agent.compression_enabled = False
        result = agent.run_conversation("Continue the implementation.", conversation_history=history)
        assert result["completed"] is True
        main_request = server.main_requests()[0]
        agent.compression_enabled = True
        before = copy.deepcopy(result["messages"])

        if manual:
            outcome = compress_now(agent, result["messages"], parse_compress_args(""))
            after = outcome.after_messages
        else:
            after, _ = agent._compress_context(result["messages"], None, approx_tokens=110000)

        assert len(after) < len(before), agent.context_compressor._last_compression_telemetry
        assert result["messages"] == before
        summary_request, = [record["body"] for record in server.requests
                            if "TURNS TO SUMMARIZE:" in str(record["body"].get("input"))]
        assert summary_request["instructions"] == main_request["instructions"]
        assert summary_request["tools"] == main_request["tools"]
        assert summary_request["prompt_cache_key"] == main_request["prompt_cache_key"]
        for option in ("reasoning", "include", "text", "tool_choice", "parallel_tool_calls", "prompt_cache_retention"):
            assert summary_request.get(option) == main_request.get(option)
        prefix = summary_request["input"][:-1]
        # The last cached checkpoint may be near the end of the main request.
        # Replaying only the middle window loses that checkpoint on some routes.
        assert len(prefix) >= len(main_request["input"])
        assert prefix[:len(main_request["input"])] == main_request["input"]
        assert "Source files inspected." in str(prefix[-1])
        assert summary_request["input"][-1]["role"] == "developer"
        assert "structured checkpoint summary" in str(summary_request["input"][-1]).lower()
        assert "protected context" in str(summary_request["input"][-1])
        assert server.invalid_requests() == []


def test_provider_fallback_keeps_bounded_input_and_records_its_size():
    with FakeResponsesServer([HttpError(status=402)]) as primary, FakeResponsesServer(default_text="Recovered summary") as fallback:
        save_config({"auxiliary": {"transient_retries": 0, "compression": {"fallback_chain": [{
            "provider": "custom", "model": "gpt-4.1", "base_url": fallback.base_url,
            "api_key": "test-key", "api_mode": "codex_responses",
        }]}}})
        compressor = ContextCompressor(model="fake-responses-model", config_context_length=128000, quiet_mode=True)
        prefix = SummaryPrefix(
            system_prompt="Keep the coding task context.", tools=_TOOLS, reasoning_config=None, context_length=128000,
            runtime={"provider": "custom", "model": "fake-responses-model", "base_url": primary.base_url,
                     "api_mode": "codex_responses", "api_key": "test-key", "session_id": "fallback-prefix"},
            messages=[{"role": "user", "content": "Implement the task."}, {"role": "assistant", "content": "Files inspected."}],
        )
        bounded = [{"role": "user", "content": "Bounded, redacted summary input."}]
        route = {}
        with summary_prefix_scope(prefix, compressor, 2000, None, "", True):
            response = call_llm(
                task="compression", provider="custom", model="fake-responses-model", base_url=primary.base_url,
                api_key="test-key", api_mode="codex_responses", main_runtime=prefix.runtime,
                messages=bounded, route_info=route,
            )
            final_estimate = summary_prompt_estimate()
        assert response.choices[0].message.content
        assert primary.requests[0]["body"]["instructions"] == prefix.system_prompt
        recovered = fallback.requests[-1]["body"]
        assert not recovered.get("tools")
        recovered_messages = recovered.get("input", recovered.get("messages"))
        assert [{key: item[key] for key in ("role", "content")} for item in recovered_messages] == bounded
        assert final_estimate == estimate_request_tokens_rough(bounded)
        assert summary_prompt_estimate() is None
