"""Integration tests for HermesAgentLoop with a local vLLM server.

Tests the full Phase 2 flow: ManagedServer + tool calling with a real
vLLM backend, producing actual token IDs and logprobs for RL training.

Requires a running vLLM server. Start one from the atropos directory:

    python -m example_trainer.vllm_api_server \
        --model Qwen/Qwen3-4B-Thinking-2507 \
        --port 9001 \
        --gpu-memory-utilization 0.8 \
        --max-model-len=32000

Tests are automatically skipped if the server is not reachable.

Run:
    pytest tests/test_agent_loop_vllm.py -v
    pytest tests/test_agent_loop_vllm.py -v -k "single"
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import requests

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from environments.agent_loop import AgentResult, HermesAgentLoop
except ImportError:
    pytest.skip("atroposlib not installed", allow_module_level=True)


# =========================================================================
# Configuration
# =========================================================================

VLLM_HOST = "localhost"
VLLM_PORT = 9001
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_MODEL = "Qwen/Qwen3-4B-Thinking-2507"


def _vllm_is_running() -> bool:
    """Check if the vLLM server is reachable."""
    try:
        r = requests.get(f"{VLLM_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# Skip all tests in this module if vLLM is not running
pytestmark = pytest.mark.skipif(
    not _vllm_is_running(),
    reason=(
        f"vLLM server not reachable at {VLLM_BASE_URL}. "
        "Start it with: python -m example_trainer.vllm_api_server "
        f"--model {VLLM_MODEL} --port {VLLM_PORT} "
        "--gpu-memory-utilization 0.8 --max-model-len=32000"
    ),
)


# =========================================================================
# Server setup
# =========================================================================

def _make_server_manager():
    """Create a ServerManager pointing to the local vLLM server."""
    from atroposlib.envs.server_handling.server_manager import (
        ServerManager,
        APIServerConfig,
    )

    config = APIServerConfig(
        base_url=VLLM_BASE_URL,
        model_name=VLLM_MODEL,
        server_type="vllm",
        health_check=False,
    )
    sm = ServerManager([config], tool_parser="hermes")
    sm.servers[0].server_healthy = True
    return sm


def _get_tokenizer():
    """Load the tokenizer for the model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(VLLM_MODEL)


# =========================================================================
# Fake tools
# =========================================================================

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature and conditions.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'",
                }
            },
            "required": ["city"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculate a math expression. Returns the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 3'",
                }
            },
            "required": ["expression"],
        },
    },
}


def _fake_tool_handler(tool_name: str, args: Dict[str, Any], **kwargs) -> str:
    """Handle fake tool calls for testing."""
    if tool_name == "get_weather":
        city = args.get("city", "Unknown")
        return json.dumps({
            "city": city,
            "temperature": 22,
            "conditions": "sunny",
            "humidity": 45,
        })
    elif tool_name == "calculate":
        expr = args.get("expression", "0")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# =========================================================================
# Tests
# =========================================================================

@pytest.mark.asyncio
async def test_vllm_single_tool_call():
    """vLLM model calls a tool, gets result, responds — full Phase 2 flow."""
    sm = _make_server_manager()
    tokenizer = _get_tokenizer()

    async with sm.managed_server(tokenizer=tokenizer) as managed:
        agent = HermesAgentLoop(
            server=managed,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.6,
            max_tokens=1000,
        )

        messages = [
            {"role": "user", "content": "What's the weather in Tokyo? Use the get_weather tool."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

    assert isinstance(result, AgentResult)
    assert result.turns_used >= 2, f"Expected at least 2 turns, got {result.turns_used}"

    # Verify tool call happened
    tool_calls_found = False
    for msg in result.messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "get_weather":
                    tool_calls_found = True
                    args = json.loads(tc["function"]["arguments"])
                    assert "city" in args
    assert tool_calls_found, "Model should have called get_weather"

    # Verify tool results in conversation
    tool_results = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1


@pytest.mark.asyncio
async def test_vllm_multi_tool_calls():
    """vLLM model calls multiple tools across turns."""
    sm = _make_server_manager()
    tokenizer = _get_tokenizer()

    async with sm.managed_server(tokenizer=tokenizer) as managed:
        agent = HermesAgentLoop(
            server=managed,
            tool_schemas=[WEATHER_TOOL, CALC_TOOL],
            valid_tool_names={"get_weather", "calculate"},
            max_turns=10,
            temperature=0.6,
            max_tokens=1000,
        )

        messages = [
            {"role": "user", "content": (
                "I need two things: "
                "1) What's the weather in Paris? Use get_weather. "
                "2) What is 15 * 7? Use calculate."
            )},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

    # Both tools should be called
    tools_called = set()
    for msg in result.messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tools_called.add(tc["function"]["name"])

    assert "get_weather" in tools_called, f"get_weather not called. Called: {tools_called}"
    assert "calculate" in tools_called, f"calculate not called. Called: {tools_called}"


@pytest.mark.asyncio
async def test_vllm_managed_server_produces_nodes():
    """ManagedServer should produce SequenceNodes with tokens and logprobs."""
    sm = _make_server_manager()
    tokenizer = _get_tokenizer()

    async with sm.managed_server(tokenizer=tokenizer) as managed:
        agent = HermesAgentLoop(
            server=managed,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.6,
            max_tokens=1000,
        )

        messages = [
            {"role": "user", "content": "What's the weather in Berlin? Use get_weather."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # Get the managed state — should have SequenceNodes
        state = managed.get_state()

    assert state is not None, "ManagedServer should return state"
    nodes = state.get("nodes", [])
    assert len(nodes) >= 1, f"Should have at least 1 node, got {len(nodes)}"

    node = nodes[0]
    assert hasattr(node, "tokens"), "Node should have tokens"
    assert hasattr(node, "logprobs"), "Node should have logprobs"
    assert len(node.tokens) > 0, "Tokens should not be empty"
    assert len(node.logprobs) > 0, "Logprobs should not be empty"
    assert len(node.tokens) == len(node.logprobs), (
        f"Tokens ({len(node.tokens)}) and logprobs ({len(node.logprobs)}) should have same length"
    )


@pytest.mark.asyncio
async def test_vllm_no_tools_direct_response():
    """vLLM model should respond directly when no tools are needed."""
    sm = _make_server_manager()
    tokenizer = _get_tokenizer()

    async with sm.managed_server(tokenizer=tokenizer) as managed:
        agent = HermesAgentLoop(
            server=managed,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.6,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer directly, no tools."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

    assert result.finished_naturally, "Should finish naturally"
    assert result.turns_used == 1, f"Should take 1 turn, took {result.turns_used}"

    final = result.messages[-1]
    assert final["role"] == "assistant"
    assert final["content"], "Should have content"


@pytest.mark.asyncio
async def test_vllm_thinking_content_extracted():
    """Qwen3-Thinking model should produce reasoning content."""
    sm = _make_server_manager()
    tokenizer = _get_tokenizer()

    async with sm.managed_server(
        tokenizer=tokenizer,
        preserve_think_blocks=True,
    ) as managed:
        agent = HermesAgentLoop(
            server=managed,
            tool_schemas=[CALC_TOOL],
            valid_tool_names={"calculate"},
            max_turns=5,
            temperature=0.6,
            max_tokens=1000,
        )

        messages = [
            {"role": "user", "content": "What is 123 * 456? Use the calculate tool."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

    # Qwen3-Thinking should generate <think> blocks
    # Check if any content contains thinking markers
    has_thinking = False
    for msg in result.messages:
        content = msg.get("content", "") or ""
        if "<think>" in content or "</think>" in content:
            has_thinking = True
            break

    # Also check reasoning_per_turn
    has_reasoning = any(r for r in result.reasoning_per_turn if r)

    # At least one of these should be true for a thinking model
    assert has_thinking or has_reasoning, (
        "Qwen3-Thinking should produce <think> blocks or reasoning content"
    )
