"""
Tests for environments/agent_loop.py — HermesAgentLoop.

Tests the multi-turn agent engine using mocked servers, without needing
real API keys or running servers.
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environments.agent_loop import (
    AgentResult,
    HermesAgentLoop,
    ToolError,
    _extract_reasoning_from_message,
    resize_tool_pool,
)


# ─── Mock server infrastructure ─────────────────────────────────────────


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"


@dataclass
class MockMessage:
    content: Optional[str]
    role: str = "assistant"
    tool_calls: Optional[List[MockToolCall]] = None
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[list] = None


@dataclass
class MockChoice:
    message: MockMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class MockChatCompletion:
    choices: List[MockChoice]
    id: str = "chatcmpl-mock"
    model: str = "mock-model"


class MockServer:
    """
    Mock server that returns pre-configured responses in sequence.
    Mimics the chat_completion() interface.
    """

    def __init__(self, responses: List[MockChatCompletion]):
        self.responses = responses
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []

    async def chat_completion(self, **kwargs) -> MockChatCompletion:
        self.call_history.append(kwargs)
        if self.call_count >= len(self.responses):
            # Return a simple text response if we run out
            return MockChatCompletion(
                choices=[MockChoice(message=MockMessage(content="Done."))]
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


def make_text_response(content: str) -> MockChatCompletion:
    """Create a simple text-only response (no tool calls)."""
    return MockChatCompletion(
        choices=[MockChoice(message=MockMessage(content=content))]
    )


def make_tool_response(
    tool_name: str,
    arguments: dict,
    content: str = "",
    tool_call_id: str = "call_001",
) -> MockChatCompletion:
    """Create a response with a single tool call."""
    return MockChatCompletion(
        choices=[
            MockChoice(
                message=MockMessage(
                    content=content,
                    tool_calls=[
                        MockToolCall(
                            id=tool_call_id,
                            function=MockFunction(
                                name=tool_name,
                                arguments=json.dumps(arguments),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )


# ─── Tests ───────────────────────────────────────────────────────────────


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(messages=[])
        assert result.messages == []
        assert result.managed_state is None
        assert result.turns_used == 0
        assert result.finished_naturally is False
        assert result.reasoning_per_turn == []
        assert result.tool_errors == []


class TestExtractReasoning:
    def test_reasoning_content_field(self):
        msg = MockMessage(content="hello", reasoning_content="I think...")
        assert _extract_reasoning_from_message(msg) == "I think..."

    def test_reasoning_field(self):
        msg = MockMessage(content="hello", reasoning="Let me consider...")
        assert _extract_reasoning_from_message(msg) == "Let me consider..."

    def test_reasoning_details(self):
        detail = MagicMock()
        detail.text = "Detail reasoning"
        msg = MockMessage(content="hello", reasoning_details=[detail])
        assert _extract_reasoning_from_message(msg) == "Detail reasoning"

    def test_reasoning_details_dict_format(self):
        msg = MockMessage(
            content="hello",
            reasoning_details=[{"text": "Dict reasoning"}],
        )
        assert _extract_reasoning_from_message(msg) == "Dict reasoning"

    def test_no_reasoning(self):
        msg = MockMessage(content="hello")
        assert _extract_reasoning_from_message(msg) is None

    def test_reasoning_content_takes_priority(self):
        msg = MockMessage(
            content="hello",
            reasoning_content="First",
            reasoning="Second",
        )
        assert _extract_reasoning_from_message(msg) == "First"


class TestHermesAgentLoop:
    """Test the agent loop with mock servers."""

    @pytest.fixture
    def basic_tools(self):
        """Minimal tool schema for testing."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to run",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    @pytest.fixture
    def valid_names(self):
        return {"terminal", "read_file", "todo"}

    @pytest.mark.asyncio
    async def test_simple_text_response(self, basic_tools, valid_names):
        """Model responds with text only, no tool calls."""
        server = MockServer([make_text_response("Hello! How can I help?")])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = await agent.run(messages)

        assert result.finished_naturally is True
        assert result.turns_used == 1
        assert len(result.messages) >= 2  # user + assistant
        assert result.messages[-1]["role"] == "assistant"
        assert result.messages[-1]["content"] == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self, basic_tools, valid_names):
        """Model calls a tool, then responds with text."""
        server = MockServer([
            make_tool_response("todo", {"todos": [{"id": "1", "content": "test", "status": "pending"}]}),
            make_text_response("I created a todo for you."),
        ])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Create a todo"}]
        result = await agent.run(messages)

        assert result.finished_naturally is True
        assert result.turns_used == 2
        # Should have: user, assistant (tool_call), tool (result), assistant (text)
        roles = [m["role"] for m in result.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

    @pytest.mark.asyncio
    async def test_max_turns_reached(self, basic_tools, valid_names):
        """Model keeps calling tools until max_turns is hit."""
        # Create responses that always call a tool
        responses = [
            make_tool_response("todo", {"todos": [{"id": str(i), "content": f"task {i}", "status": "pending"}]}, tool_call_id=f"call_{i}")
            for i in range(10)
        ]
        server = MockServer(responses)
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=3,
        )
        messages = [{"role": "user", "content": "Keep going"}]
        result = await agent.run(messages)

        assert result.finished_naturally is False
        assert result.turns_used == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_name(self, basic_tools, valid_names):
        """Model calls a tool not in valid_tool_names."""
        server = MockServer([
            make_tool_response("nonexistent_tool", {"arg": "val"}),
            make_text_response("OK, that didn't work."),
        ])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Call something weird"}]
        result = await agent.run(messages)

        # Should record a tool error
        assert len(result.tool_errors) >= 1
        assert result.tool_errors[0].tool_name == "nonexistent_tool"

    @pytest.mark.asyncio
    async def test_empty_response(self, basic_tools, valid_names):
        """Server returns empty response."""
        server = MockServer([MockChatCompletion(choices=[])])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = await agent.run(messages)

        assert result.finished_naturally is False
        assert result.turns_used == 1

    @pytest.mark.asyncio
    async def test_api_error_handling(self, basic_tools, valid_names):
        """Server raises an exception."""

        class FailingServer:
            async def chat_completion(self, **kwargs):
                raise ConnectionError("Server unreachable")

        agent = HermesAgentLoop(
            server=FailingServer(),
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = await agent.run(messages)

        assert result.finished_naturally is False
        assert result.turns_used == 1

    @pytest.mark.asyncio
    async def test_tools_passed_to_server(self, basic_tools, valid_names):
        """Verify tools are passed in the chat_completion kwargs."""
        server = MockServer([make_text_response("OK")])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        await agent.run(messages)

        assert len(server.call_history) == 1
        assert "tools" in server.call_history[0]
        assert server.call_history[0]["tools"] == basic_tools

    @pytest.mark.asyncio
    async def test_extra_body_forwarded(self, basic_tools, valid_names):
        """extra_body should be forwarded to server."""
        extra = {"provider": {"ignore": ["DeepInfra"]}}
        server = MockServer([make_text_response("OK")])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
            extra_body=extra,
        )
        messages = [{"role": "user", "content": "Hi"}]
        await agent.run(messages)

        assert server.call_history[0].get("extra_body") == extra

    @pytest.mark.asyncio
    async def test_managed_state_returned(self, basic_tools, valid_names):
        """If server has get_state(), result should include managed_state."""
        server = MockServer([make_text_response("OK")])
        server.get_state = lambda: {"nodes": [{"test": True}]}

        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = await agent.run(messages)

        assert result.managed_state is not None
        assert "nodes" in result.managed_state

    @pytest.mark.asyncio
    async def test_no_managed_state_without_get_state(self, basic_tools, valid_names):
        """Regular server without get_state() should return None managed_state."""
        server = MockServer([make_text_response("OK")])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Hi"}]
        result = await agent.run(messages)

        assert result.managed_state is None

    @pytest.mark.asyncio
    async def test_memory_tool_blocked(self, basic_tools):
        """Memory tool should return error in RL environments."""
        valid = {"terminal", "read_file", "todo", "memory"}
        server = MockServer([
            make_tool_response("memory", {"action": "add", "target": "user", "content": "test"}),
            make_text_response("Done"),
        ])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Remember this"}]
        result = await agent.run(messages)

        # Find the tool response
        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        tool_result = json.loads(tool_msgs[0]["content"])
        assert "error" in tool_result
        assert "not available" in tool_result["error"].lower()

    @pytest.mark.asyncio
    async def test_session_search_blocked(self, basic_tools):
        """session_search should return error in RL environments."""
        valid = {"terminal", "read_file", "todo", "session_search"}
        server = MockServer([
            make_tool_response("session_search", {"query": "test"}),
            make_text_response("Done"),
        ])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "Search sessions"}]
        result = await agent.run(messages)

        tool_msgs = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        tool_result = json.loads(tool_msgs[0]["content"])
        assert "error" in tool_result

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self, basic_tools, valid_names):
        """Reasoning content should be extracted and preserved."""
        resp = MockChatCompletion(
            choices=[
                MockChoice(
                    message=MockMessage(
                        content="The answer is 42.",
                        reasoning_content="Let me think about this step by step...",
                    )
                )
            ]
        )
        server = MockServer([resp])
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=basic_tools,
            valid_tool_names=valid_names,
            max_turns=10,
        )
        messages = [{"role": "user", "content": "What is the meaning of life?"}]
        result = await agent.run(messages)

        assert len(result.reasoning_per_turn) == 1
        assert result.reasoning_per_turn[0] == "Let me think about this step by step..."


class TestResizeToolPool:
    def test_resize_works(self):
        """resize_tool_pool should not raise."""
        resize_tool_pool(16)  # Small pool for testing
        resize_tool_pool(128)  # Restore default
