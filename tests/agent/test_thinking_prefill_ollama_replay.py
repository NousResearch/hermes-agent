"""The thinking-only prefill actually reaches Ollama, and still never reaches other routes.

``turn_empty_response`` appends a ``_thinking_prefill`` stub "so the model sees its own reasoning",
but both request assemblers dropped it unconditionally, so every "prefill" was a plain re-request of
the same prompt (2026-10-15: four multi-minute gpt-oss calls on an identical 20k-token prompt).
Ollama continues a trailing reasoning-only assistant turn (OpenAI ``reasoning`` -> Thinking, the
renderer reopens the analysis channel), so on Ollama the stub is now kept when it is the last row.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.message_sanitization import trailing_prefill_replay_supported

OLLAMA = "http://10.69.20.2:11434/v1"


def _response(content, finish_reason, reasoning=None):
    """OpenAI ChatCompletion stand-in (same shape as test_run_agent._mock_response, which this file
    does not import: that module pulls in tui_gateway.server)."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        msg.reasoning = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           model="gpt-oss:120b", usage=None)


def _stub(text="Let me think about the request."):
    return {"role": "assistant", "content": "", "reasoning": text, "_thinking_prefill": True}


class TestRouteGate:
    @pytest.mark.parametrize("is_ollama, api_mode, provider, url, expected", [
        (True, "chat_completions", "custom", OLLAMA, True),
        (False, "chat_completions", "custom", "http://127.0.0.1:8080/v1", False),   # llama.cpp et al.
        (True, "codex_responses", "custom", OLLAMA, False),
        (True, "chat_completions", "mistral", "https://api.mistral.ai/v1", False),  # strict route
        (True, "chat_completions", "custom", "", False),
    ])
    def test_gate(self, is_ollama, api_mode, provider, url, expected):
        with patch("agent.model_metadata._is_ollama_server", return_value=is_ollama):
            assert trailing_prefill_replay_supported(api_mode, provider, url) is expected

    def test_gate_does_not_probe_strict_routes(self):
        with patch("agent.model_metadata._is_ollama_server") as probe:
            trailing_prefill_replay_supported("chat_completions", "mistral", "https://api.mistral.ai/v1")
        probe.assert_not_called()


@pytest.fixture()
def ollama_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="ollama", base_url=OLLAMA, provider="custom", model="gpt-oss:120b",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _second_request(agent, is_ollama, reasoning):
    agent.client.chat.completions.create.side_effect = [
        _response("", "stop", reasoning=reasoning),  # Ollama ends a reasoning-only reply with stop
        _response("pong", "stop"),
    ]
    with (
        patch("agent.model_metadata._is_ollama_server", return_value=is_ollama),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("ping")
    call = agent.client.chat.completions.create.call_args_list[1]
    return call.kwargs.get("messages") or call.args[0].get("messages"), result


def test_ollama_wire_payload_ends_on_the_reasoning_prefill(ollama_agent):
    reasoning = "The user says ping; a short acknowledgement is right."
    sent, result = _second_request(ollama_agent, True, reasoning)
    last = sent[-1]
    assert last["role"] == "assistant" and not (last.get("content") or "").strip()
    assert last.get("reasoning") == reasoning, f"reasoning did not reach the wire: {last}"
    assert not any(str(k).startswith("_") for m in sent for k in m), "internal keys reached the wire"
    assert result["final_response"] == "pong"


def test_other_servers_still_get_no_prefill_stub(ollama_agent):
    sent, result = _second_request(ollama_agent, False, "Thinking about ping.")
    assert sent[-1]["role"] == "user", f"stub reached a route that does not continue it: {sent[-1]}"
    assert result["final_response"] == "pong"
