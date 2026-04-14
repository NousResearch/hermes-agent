"""Tests for Matrix thinking/acting panes."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(thinking_enabled=True):
    from gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test123",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            "thinking_fields_enabled": thinking_enabled,
        },
    )
    adapter = MatrixAdapter(config)
    adapter._client = AsyncMock()
    return adapter


class TestThinkingManagerTitles:
    def test_field_titles_match_branch_c_requirement(self):
        from gateway.platforms.matrix_thinking import ThinkingManager

        assert ThinkingManager._field_title("thinking", "gpt-5.4 via openai-codex") == "Agent Thinking: Hermes via gpt-5.4 via openai-codex"
        assert ThinkingManager._field_title("tools", "gpt-5.4 via openai-codex") == "Agent Acting:"

    def test_plaintext_summary_includes_required_heading(self):
        from gateway.platforms.matrix_thinking import ThinkingManager

        text = ThinkingManager._plaintext_summary(
            "Agent Thinking: Hermes via gpt-5.4 via openai-codex",
            "Processing request...",
        )
        assert "Agent Thinking: Hermes via gpt-5.4 via openai-codex" in text
        assert "Processing request..." in text


class TestThinkingManagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_thinking_uses_branch_c_heading(self):
        from gateway.platforms.matrix_thinking import ThinkingManager

        adapter = _make_adapter()
        send_message_event = AsyncMock(return_value="$evt_123")
        adapter._client.send_message_event = send_message_event

        mgr = ThinkingManager(adapter)
        event_id = await mgr.start(
            "!room:example.org",
            "task-1",
            "Processing request...",
            field_kind="thinking",
            model_label="gpt-5.4 via openai-codex",
            initial_content_md="first delta",
        )

        assert event_id == "$evt_123"
        content = send_message_event.call_args.args[2]
        assert "Agent Thinking: Hermes via gpt-5.4 via openai-codex" in content["formatted_body"]
        assert "first delta" in content["formatted_body"]

    @pytest.mark.asyncio
    async def test_start_tool_activity_uses_agent_acting_heading(self):
        from gateway.platforms.matrix_thinking import ThinkingManager

        adapter = _make_adapter()
        send_message_event = AsyncMock(return_value="$evt_456")
        adapter._client.send_message_event = send_message_event

        mgr = ThinkingManager(adapter)
        event_id = await mgr.start(
            "!room:example.org",
            "task-2",
            "Tool activity",
            field_kind="tools",
            model_label="gpt-5.4 via openai-codex",
            initial_content_md='💻 terminal: "pwd"',
        )

        assert event_id == "$evt_456"
        content = send_message_event.call_args.args[2]
        assert "Agent Acting:" in content["formatted_body"]
        assert '💻 terminal: &quot;pwd&quot;' in content["formatted_body"]

    def test_elapsed_str_still_formats_duration(self):
        from gateway.platforms.matrix_thinking import ThinkingManager

        now = time.time()
        assert ThinkingManager._elapsed_str(now - 45) == "45s"
        assert ThinkingManager._elapsed_str(now - 125) == "2m5s"
