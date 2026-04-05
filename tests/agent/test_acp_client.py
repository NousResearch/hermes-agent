"""Tests for the generalised ACP client."""

from __future__ import annotations

import json
import subprocess
import threading
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from agent.acp_client import (
    ACPClient,
    _format_messages_as_prompt,
    _render_message_content,
    extract_agent_from_url,
)


# ---------------------------------------------------------------------------
# extract_agent_from_url
# ---------------------------------------------------------------------------

class TestExtractAgentFromUrl:
    def test_claude(self):
        assert extract_agent_from_url("acp://claude") == "claude"

    def test_codex(self):
        assert extract_agent_from_url("acp://codex") == "codex"

    def test_copilot(self):
        assert extract_agent_from_url("acp://copilot") == "copilot"

    def test_non_acp_url(self):
        assert extract_agent_from_url("https://api.openai.com") is None

    def test_empty(self):
        assert extract_agent_from_url("") is None

    def test_bare_prefix(self):
        assert extract_agent_from_url("acp://") is None


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

class TestFormatMessages:
    def test_basic_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _format_messages_as_prompt(msgs, agent_name="claude")
        assert "User:\nHello" in result
        assert "agent: claude" in result

    def test_system_and_user(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _format_messages_as_prompt(msgs)
        assert "System:\nBe helpful" in result
        assert "User:\nHi" in result

    def test_model_hint(self):
        result = _format_messages_as_prompt([], model="gpt-5")
        assert "gpt-5" in result

    def test_empty_messages(self):
        result = _format_messages_as_prompt([])
        assert "Continue the conversation" in result


class TestRenderMessageContent:
    def test_string(self):
        assert _render_message_content("hello") == "hello"

    def test_none(self):
        assert _render_message_content(None) == ""

    def test_dict_with_text(self):
        assert _render_message_content({"text": "hi"}) == "hi"

    def test_list_of_parts(self):
        parts = [{"text": "a"}, {"text": "b"}]
        assert _render_message_content(parts) == "a\nb"


# ---------------------------------------------------------------------------
# ACPClient init
# ---------------------------------------------------------------------------

class TestACPClientInit:
    def test_known_agent(self):
        client = ACPClient(agent_name="gemini")
        assert client.agent_name == "gemini"
        assert client.base_url == "acp://gemini"

    def test_agent_from_base_url(self):
        client = ACPClient(base_url="acp://codex")
        assert client.agent_name == "codex"

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown ACP agent"):
            ACPClient(agent_name="nonexistent-xyz")

    def test_custom_command(self):
        client = ACPClient(agent_name="custom", acp_command="/bin/my-agent --acp")
        assert client.agent_name == "custom"

    def test_close_idempotent(self):
        client = ACPClient(agent_name="gemini")
        client.close()
        client.close()  # should not raise


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestCopilotBackwardsCompat:
    def test_import(self):
        from agent.copilot_acp_client import CopilotACPClient, ACP_MARKER_BASE_URL
        assert ACP_MARKER_BASE_URL == "acp://copilot"

    def test_creates_copilot_agent(self):
        from agent.copilot_acp_client import CopilotACPClient
        client = CopilotACPClient()
        assert client.agent_name == "copilot"
        assert client.base_url == "acp://copilot"


# ---------------------------------------------------------------------------
# ACP protocol flow (mocked subprocess)
# ---------------------------------------------------------------------------

class TestACPProtocolFlow:
    """Test the full initialize → session/new → session/prompt flow with a mock process."""

    def _make_mock_process(self, responses: list[dict[str, Any]]) -> mock.MagicMock:
        """Create a mock Popen that feeds back canned ACP responses."""
        proc = mock.MagicMock()
        proc.poll.return_value = None

        # Track what's written to stdin
        written_lines: list[str] = []

        class FakeStdin:
            def write(self, data: str) -> None:
                written_lines.append(data)

            def flush(self) -> None:
                pass

        proc.stdin = FakeStdin()
        proc._written = written_lines

        # Feed responses through stdout as NDJSON lines
        response_iter = iter(responses)

        def line_generator():
            for resp in response_iter:
                yield json.dumps(resp) + "\n"

        proc.stdout = line_generator()
        proc.stderr = iter([])  # empty stderr
        return proc

    def test_successful_prompt(self):
        """Full ACP flow returns assembled text."""
        responses = [
            # initialize response
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}},
            # session/new response
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-123"}},
            # streaming text chunks
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "Hello "},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "world!"},
                    }
                },
            },
            # session/prompt response
            {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "completed"}},
        ]

        with mock.patch("subprocess.Popen") as mock_popen:
            proc = self._make_mock_process(responses)
            mock_popen.return_value = proc

            client = ACPClient(agent_name="gemini")
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gemini-acp",
            )

            assert result.choices[0].message.content == "Hello world!"
            assert result.model == "gemini-acp"

    def test_reasoning_chunks(self):
        """Thought chunks are captured as reasoning."""
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-456"}},
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"text": "thinking..."},
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "answer"},
                    }
                },
            },
            {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "completed"}},
        ]

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self._make_mock_process(responses)
            client = ACPClient(agent_name="gemini")
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Think"}],
            )

            assert result.choices[0].message.content == "answer"
            assert result.choices[0].message.reasoning == "thinking..."
