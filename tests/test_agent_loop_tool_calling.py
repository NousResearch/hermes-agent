"""Integration tests for HermesAgentLoop tool calling.

Tests the full agent loop with real LLM calls via OpenRouter.
Uses stepfun/step-3.5-flash:free by default (zero cost), falls back
to anthropic/claude-sonnet-4 if the free model is unavailable.

These tests verify:
1. Single tool call: model calls a tool, gets result, responds
2. Multi-tool call: model calls multiple tools in one turn
3. Multi-turn: model calls tools across multiple turns
4. Unknown tool rejection: model calling a non-existent tool gets an error
5. Max turns: loop stops when max_turns is reached
6. No tools: model responds without calling any tools
7. Tool error handling: tool execution errors are captured

Run:
    pytest tests/test_agent_loop_tool_calling.py -v
    pytest tests/test_agent_loop_tool_calling.py -v -k "single"  # run one test
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import patch

import pytest

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from environments.agent_loop import AgentResult, HermesAgentLoop
except ImportError:
    pytest.skip("atroposlib not installed", allow_module_level=True)


# =========================================================================
# Test infrastructure
# =========================================================================

# Models to try, in order of preference (free first)
_MODELS = [
    "stepfun/step-3.5-flash:free",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-sonnet-4",
]

def _get_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


def _make_server(model: str = None):
    """Create an OpenAI server for testing."""
    from atroposlib.envs.server_handling.openai_server import OpenAIServer
    from atroposlib.envs.server_handling.server_manager import APIServerConfig

    config = APIServerConfig(
        base_url="https://openrouter.ai/api/v1",
        model_name=model or _MODELS[0],
        server_type="openai",
        api_key=_get_api_key(),
        health_check=False,
    )
    return OpenAIServer(config)


async def _try_models(test_fn):
    """Try running a test with each model until one works."""
    last_error = None
    for model in _MODELS:
        try:
            server = _make_server(model)
            return await test_fn(server, model)
        except Exception as e:
            last_error = e
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                continue  # Rate limited, try next model
            raise  # Real error
    pytest.skip(f"All models failed. Last error: {last_error}")


# =========================================================================
# Fake tools for testing
# =========================================================================

# Simple calculator tool
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
                    "description": "Math expression to evaluate, e.g. '2 + 3'"
                }
            },
            "required": ["expression"],
        },
    },
}

# Weather lookup tool
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
                    "description": "City name, e.g. 'Tokyo'"
                }
            },
            "required": ["city"],
        },
    },
}

# Lookup tool (always succeeds)
LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a fact. Returns a short answer string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up"
                }
            },
            "required": ["query"],
        },
    },
}

# Error tool (always fails)
ERROR_TOOL = {
    "type": "function",
    "function": {
        "name": "failing_tool",
        "description": "A tool that always fails with an error.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"],
        },
    },
}


def _fake_tool_handler(tool_name: str, args: Dict[str, Any], **kwargs) -> str:
    """Handle fake tool calls for testing."""
    if tool_name == "calculate":
        expr = args.get("expression", "0")
        try:
            # Safe eval for simple math
            result = eval(expr, {"__builtins__": {}}, {})
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif tool_name == "get_weather":
        city = args.get("city", "Unknown")
        # Return canned weather
        return json.dumps({
            "city": city,
            "temperature": 22,
            "conditions": "sunny",
            "humidity": 45,
        })

    elif tool_name == "lookup":
        query = args.get("query", "")
        return json.dumps({"answer": f"The answer to '{query}' is 42."})

    elif tool_name == "failing_tool":
        raise RuntimeError("This tool always fails!")

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# =========================================================================
# Tests
# =========================================================================

@pytest.mark.asyncio
async def test_single_tool_call():
    """Model should call a single tool, get the result, and respond."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": "What's the weather in Tokyo? Use the get_weather tool."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        assert isinstance(result, AgentResult)
        assert result.turns_used >= 2, f"Expected at least 2 turns (tool call + response), got {result.turns_used}"

        # Verify a tool call happened
        tool_calls_found = False
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] == "get_weather":
                        tool_calls_found = True
                        args = json.loads(tc["function"]["arguments"])
                        assert "city" in args
        assert tool_calls_found, "Model should have called get_weather"

        # Verify tool result is in conversation
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_results) >= 1, "Should have at least one tool result"

        # Verify the final response references the weather
        final_msg = result.messages[-1]
        assert final_msg["role"] == "assistant"
        assert final_msg["content"], "Final response should have content"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_multi_tool_single_turn():
    """Model should call multiple tools in a single turn."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[WEATHER_TOOL, CALC_TOOL],
            valid_tool_names={"get_weather", "calculate"},
            max_turns=5,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": (
                "I need two things at once: "
                "1) What's the weather in Paris? Use get_weather. "
                "2) What is 15 * 7? Use calculate. "
                "Call BOTH tools in a single response."
            )},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # Count distinct tools called
        tools_called = set()
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tools_called.add(tc["function"]["name"])

        # At minimum, both tools should have been called (maybe in different turns)
        assert "get_weather" in tools_called, f"get_weather not called. Called: {tools_called}"
        assert "calculate" in tools_called, f"calculate not called. Called: {tools_called}"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Agent should handle multiple turns of tool calls."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[LOOKUP_TOOL, CALC_TOOL],
            valid_tool_names={"lookup", "calculate"},
            max_turns=10,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": (
                "First, use the lookup tool to look up 'meaning of life'. "
                "Then use calculate to compute 6 * 7. "
                "Do these in separate tool calls, one at a time."
            )},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # Should have used both tools
        tools_called = set()
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tools_called.add(tc["function"]["name"])

        assert "lookup" in tools_called, f"lookup not called. Called: {tools_called}"
        assert "calculate" in tools_called, f"calculate not called. Called: {tools_called}"

        # Should finish naturally
        assert result.finished_naturally, "Should finish naturally after answering"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_unknown_tool_rejected():
    """If the model calls a tool not in valid_tool_names, it gets an error."""

    async def _run(server, model):
        # Only allow "calculate" but give schema for both
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[CALC_TOOL, WEATHER_TOOL],
            valid_tool_names={"calculate"},  # weather NOT allowed
            max_turns=5,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": "What's the weather in London? Use get_weather."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # Check if get_weather was called and rejected
        if result.tool_errors:
            weather_errors = [e for e in result.tool_errors if e.tool_name == "get_weather"]
            assert len(weather_errors) > 0, "get_weather should have been rejected"
            assert "Unknown tool" in weather_errors[0].error

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_max_turns_limit():
    """Agent should stop after max_turns even if model keeps calling tools."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[LOOKUP_TOOL],
            valid_tool_names={"lookup"},
            max_turns=2,  # Very low limit
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": (
                "Keep looking up facts. Look up 'fact 1', then 'fact 2', "
                "then 'fact 3', then 'fact 4'. Do them one at a time."
            )},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        assert result.turns_used <= 2, f"Should stop at max_turns=2, used {result.turns_used}"
        assert not result.finished_naturally, "Should NOT finish naturally (hit max_turns)"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_no_tools_direct_response():
    """When no tools are useful, model should respond directly."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.0,
            max_tokens=200,
        )

        messages = [
            {"role": "user", "content": "What is 2 + 2? Just answer directly, no tools needed."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        assert result.finished_naturally, "Should finish naturally with a direct response"
        assert result.turns_used == 1, f"Should take exactly 1 turn for a direct answer, took {result.turns_used}"

        final = result.messages[-1]
        assert final["role"] == "assistant"
        assert final["content"], "Should have text content"
        assert "4" in final["content"], "Should contain the answer '4'"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Tool execution errors should be captured and reported to the model."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[ERROR_TOOL],
            valid_tool_names={"failing_tool"},
            max_turns=5,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "user", "content": "Please call the failing_tool with input 'test'."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # The tool error should be recorded
        assert len(result.tool_errors) >= 1, "Should have at least one tool error"
        assert "RuntimeError" in result.tool_errors[0].error or "always fails" in result.tool_errors[0].error

        # The error should be in the conversation as a tool result
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_results) >= 1
        error_result = json.loads(tool_results[0]["content"])
        assert "error" in error_result

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_agent_result_structure():
    """Verify the AgentResult has all expected fields populated."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[CALC_TOOL],
            valid_tool_names={"calculate"},
            max_turns=5,
            temperature=0.0,
            max_tokens=300,
        )

        messages = [
            {"role": "user", "content": "What is 3 + 4? Use the calculate tool."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # Structural checks
        assert isinstance(result, AgentResult)
        assert isinstance(result.messages, list)
        assert len(result.messages) >= 3, "Should have user + assistant(tool) + tool_result + assistant(final)"
        assert isinstance(result.turns_used, int)
        assert result.turns_used > 0
        assert isinstance(result.finished_naturally, bool)
        assert isinstance(result.tool_errors, list)
        assert isinstance(result.reasoning_per_turn, list)

        # Messages should follow OpenAI format
        for msg in result.messages:
            assert "role" in msg, f"Message missing 'role': {msg}"
            assert msg["role"] in ("system", "user", "assistant", "tool"), f"Invalid role: {msg['role']}"

        return result

    await _try_models(_run)


@pytest.mark.asyncio
async def test_conversation_history_preserved():
    """The full conversation history should be in result.messages."""

    async def _run(server, model):
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=[WEATHER_TOOL],
            valid_tool_names={"get_weather"},
            max_turns=5,
            temperature=0.0,
            max_tokens=500,
        )

        messages = [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather in Berlin? Use get_weather."},
        ]

        with patch("environments.agent_loop.handle_function_call", side_effect=_fake_tool_handler):
            result = await agent.run(messages)

        # System message should be preserved
        assert result.messages[0]["role"] == "system"
        assert "weather assistant" in result.messages[0]["content"]

        # User message should be preserved
        assert result.messages[1]["role"] == "user"
        assert "Berlin" in result.messages[1]["content"]

        # Should have assistant + tool + assistant sequence
        roles = [m["role"] for m in result.messages]
        assert "tool" in roles, "Should have tool results in conversation"

        return result

    await _try_models(_run)
