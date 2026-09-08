"""Projection and fallback bridge — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""


import pytest

from agent.turn_runtime_handoff import _sdk_result_failover_reason
from agent.transports.claude_sdk_event_projector import (
    ClaudeSdkEventProjector,
)
from tests.agent.claude_sdk_fakes import (
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ServerToolUseBlock,
    ResultMessage,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestClaudeSdkFallbackBridge:
    def test_validated_quota_outcome_maps_to_canonical_reason(self):
        result = {
            "failed": True,
            "interrupted": False,
            "failover_reason": "rate_limit",
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        }

        assert _sdk_result_failover_reason(result).value == "rate_limit"


# ---------- projector ----------


class TestProjector:
    def test_assistant_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(AssistantMessage(content=[TextBlock("hello")]))
        assert out.messages == [{"role": "assistant", "content": "hello"}]
        assert out.final_text == "hello"
        assert not out.is_tool_iteration

    def test_assistant_tool_use_and_thinking(self):
        p = ClaudeSdkEventProjector()
        # Thinking arrives first, stashes onto the next assistant entry.
        p.project(AssistantMessage(content=[ThinkingBlock("pondering")]))
        out = p.project(
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]
            )
        )
        (msg,) = out.messages
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert msg["reasoning"] == "pondering"
        (call,) = msg["tool_calls"]
        assert call["id"] == "t1"
        assert call["function"]["name"] == "Bash"
        assert '"command": "ls"' in call["function"]["arguments"]

    def test_tool_result_projection(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
        )
        assert out.is_tool_iteration
        assert out.messages == [
            {"role": "tool", "tool_call_id": "t1", "content": "ok"}
        ]

    def test_tool_result_error_and_list_content(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="t2",
                        content=[{"type": "text", "text": "boom"}],
                        is_error=True,
                    )
                ]
            )
        )
        assert out.messages[0]["content"] == "[error] boom"

    def test_tool_result_truncation(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t3", content="x" * 9000)]
            )
        )
        assert len(out.messages[0]["content"]) == 4000

    def test_result_message_sets_final_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(ResultMessage(result="the answer"))
        assert out.is_result
        assert out.final_text == "the answer"
        assert out.messages == []

    def test_server_tool_use_never_emits_dangling_tool_calls(self):
        # Validator C8: server tools (web_search, ...) execute API-side and
        # never produce a {role:'tool'} echo — emitting a tool_calls entry
        # for them leaves a dangling tool_call_id that can break replay
        # through a native provider after a /model switch.
        p = ClaudeSdkEventProjector()
        out = p.project(
            AssistantMessage(content=[
                ServerToolUseBlock(id="srv-1", name="web_search", input={"query": "x"}),
                TextBlock("found it"),
            ])
        )
        (msg,) = out.messages
        assert msg.get("tool_calls") in (None, [],) or "srv-1" not in str(msg.get("tool_calls"))
        assert msg["content"] == "found it"

    def test_lifecycle_messages_ignored(self):
        p = ClaudeSdkEventProjector()
        assert p.project(SystemMessage()).messages == []
        # A plain-text user echo must not duplicate the real user turn.
        assert p.project(UserMessage(content="hi")).messages == []
