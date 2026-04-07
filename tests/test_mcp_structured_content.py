"""Tests for MCP tool structured_content preservation."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


class _FakeContentBlock:
    """Minimal content block with .text and .type attributes."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in."""

    def __init__(self, content, is_error=False, structured_content=None):
        self.content = content
        self.isError = is_error
        self.structured_content = structured_content


@pytest.fixture
def _patch_mcp_server():
    """Patch _servers and the MCP event loop so _make_tool_handler can run."""
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}):
        yield fake_session


class TestStructuredContentPreservation:
    """Ensure structured_content from CallToolResult is forwarded."""

    def test_text_only_result(self, _patch_mcp_server):
        """When no structured_content, result is text-only (existing behaviour)."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("hello")],
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data == {"result": "hello"}
        assert "structuredContent" not in data

    def test_structured_content_included(self, _patch_mcp_server):
        """When structured_content is present, it must appear in the response."""
        session = _patch_mcp_server
        payload = {"value": "secret-123", "revealed": True}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("OK")],
                structured_content=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data["result"] == "OK"
        assert data["structuredContent"] == payload

    def test_structured_content_none_omitted(self, _patch_mcp_server):
        """When structured_content is explicitly None, key is omitted."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                structured_content=None,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data == {"result": "done"}
        assert "structuredContent" not in data

    def test_empty_text_with_structured_content(self, _patch_mcp_server):
        """When content blocks are empty but structured_content exists."""
        session = _patch_mcp_server
        payload = {"status": "ok", "data": [1, 2, 3]}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[],
                structured_content=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data["result"] == ""
        assert data["structuredContent"] == payload
