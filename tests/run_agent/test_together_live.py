"""Paid end-to-end Together AI compatibility matrix.

Opt in explicitly:

    HERMES_LIVE_TESTS=1 HERMES_TOGETHER_MODEL_MATRIX=1 \
    TOGETHER_API_KEY=... \
        python -m pytest -o addopts='' tests/run_agent/test_together_live.py -q

Every request runs through ``AIAgent``. The tool tests use a test-only echo
tool whose dispatcher has no external side effects.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from run_agent import AIAgent


LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"
MATRIX = os.environ.get("HERMES_TOGETHER_MODEL_MATRIX") == "1"
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
TOGETHER_ENDPOINT = "https://api.together.ai/v1"

ALL_MODELS = (
    "thinkingmachines/Inkling",
    "MiniMaxAI/MiniMax-M3",
    "MiniMaxAI/MiniMax-M2.7",
    "Qwen/Qwen3.7-Max",
    "Qwen/Qwen3.6-Plus",
    "Qwen/Qwen3.5-9B",
    "moonshotai/Kimi-K2.7-Code",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.2",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-V4-Pro",
    "nvidia/nemotron-3-ultra-550b-a55b",
)

TOOL_MODELS = tuple(
    model
    for model in ALL_MODELS
    if model not in {"Qwen/Qwen3.7-Max", "Qwen/Qwen3.6-Plus"}
)

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="paid live test: set HERMES_LIVE_TESTS=1"),
    pytest.mark.skipif(
        not MATRIX,
        reason="paid model matrix: set HERMES_TOGETHER_MODEL_MATRIX=1",
    ),
    pytest.mark.skipif(
        not TOGETHER_KEY,
        reason="TOGETHER_API_KEY is not configured",
    ),
    pytest.mark.integration,
]


def _agent(model: str, *, max_tokens: int, max_iterations: int) -> AIAgent:
    agent = AIAgent(
        provider="together",
        api_key=TOGETHER_KEY,
        base_url=TOGETHER_ENDPOINT,
        api_mode="chat_completions",
        model=model,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    return agent


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_together_model_basic_chat_through_hermes(model):
    agent = _agent(model, max_tokens=512, max_iterations=2)
    agent.tools = []
    agent.valid_tool_names = set()

    result = agent.run_conversation(
        "Reply with exactly the lowercase word pong. Do not explain or use tools."
    )

    content = str(result.get("final_response") or "").strip().lower()
    assert "pong" in content, f"{model} returned no usable chat response: {result}"


@pytest.mark.parametrize("model", TOOL_MODELS, ids=TOOL_MODELS)
def test_together_model_tool_round_trip_through_hermes(model):
    agent = _agent(model, max_tokens=1024, max_iterations=4)
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "hermes_test_echo",
                "description": "Echo a short value. You must call this tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "enum": ["ping"]},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    agent.valid_tool_names = {"hermes_test_echo"}
    calls: list[tuple[str, dict]] = []

    def dispatch(function_name, function_args, *args, **kwargs):
        calls.append((function_name, dict(function_args or {})))
        return json.dumps({"success": True, "echo": function_args.get("value")})

    with patch("run_agent.handle_function_call", side_effect=dispatch):
        result = agent.run_conversation(
            "Call hermes_test_echo once with value ping. "
            "After receiving the tool result, reply with exactly TOOL_OK."
        )

    assert calls, f"{model} did not call the Hermes tool"
    assert calls[0] == ("hermes_test_echo", {"value": "ping"})
    if (
        model == "openai/gpt-oss-120b"
        and result.get("failure_reason") == "server_error"
        and "HTTP 500" in str(result.get("error") or "")
    ):
        pytest.xfail(
            "Together currently returns HTTP 500 on the GPT-OSS 120B "
            "tool-result follow-up after accepting and executing the tool call"
        )
    content = str(result.get("final_response") or "").strip().upper()
    assert "TOOL_OK" in content, f"{model} did not finish after the tool result: {result}"
