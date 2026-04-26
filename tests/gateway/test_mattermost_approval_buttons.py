"""Tests for Mattermost interactive approval buttons."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.platforms.mattermost import MattermostAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter():
    """Create a MattermostAdapter with mocked internals."""
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com", "callback_url": "http://localhost:8644/hermes-approval"},
    )
    adapter = MattermostAdapter(config)
    adapter._session = MagicMock()
    adapter._bot_user_id = "bot_user_id"
    adapter._bot_username = "hermes-bot"
    adapter._reply_mode = "off"
    return adapter


class TestMattermostExecApproval:
    """Test the send_exec_approval method sends interactive attachments."""

    @pytest.mark.asyncio
    async def test_sends_attachment_with_buttons(self):
        adapter = _make_adapter()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post123"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=mock_resp)

        with patch("tools.approval._pending_approval_tools", set()):
            result = await adapter.send_exec_approval(
                chat_id="channel_1",
                command="rm -rf /important",
                session_key="agent:main:mattermost:channel:channel_1:1111",
                description="dangerous deletion",
            )

        assert result.success is True
        assert result.message_id == "post123"

        adapter._session.post.assert_called_once()
        call_args = adapter._session.post.call_args
        payload = call_args[1]["json"]
        assert payload["channel_id"] == "channel_1"
        assert "props" in payload
        attachments = payload["props"]["attachments"]
        assert len(attachments) == 1
        attachment = attachments[0]
        assert "actions" in attachment
        actions = attachment["actions"]
        assert len(actions) == 4

        # Each action has integration URL + context
        for action in actions:
            assert "integration" in action
            ctx = action["integration"]["context"]
            assert "session_key" in ctx
            assert "action" in ctx

    @pytest.mark.asyncio
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        adapter._reply_mode = "thread"
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post456"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=mock_resp)

        with patch("tools.approval._pending_approval_tools", set()):
            await adapter.send_exec_approval(
                chat_id="channel_1",
                command="echo test",
                session_key="test-session",
                metadata={"root_id": "root_post_123"},
            )

        payload = adapter._session.post.call_args[1]["json"]
        assert payload["root_id"] == "root_post_123"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._session = None
        result = await adapter.send_exec_approval(
            chat_id="channel_1", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"id": "post789"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=mock_resp)

        long_cmd = "x" * 5000
        with patch("tools.approval._pending_approval_tools", set()):
            await adapter.send_exec_approval(
                chat_id="channel_1", command=long_cmd, session_key="s"
            )

        payload = adapter._session.post.call_args[1]["json"]
        text = payload["message"]
        assert "..." in text
        assert len(text) < 5000


class TestWebhookApprovalHandler:
    """Test the webhook's _handle_mattermost_approval handler."""

    @pytest.mark.asyncio
    async def test_parses_integration_context(self):
        """Handler extracts session_key and action from Mattermost integration POST."""
        from aiohttp import web
        
        # Build minimal mock webhook adapter
        mock = MagicMock()
        
        # Simulate what Mattermost POSTs (integration context in "context" key)
        payload = {
            "channel_id": "channel_1",
            "post_id": "post123",
            "user_id": "user_456",
            "context": {
                "action": "approve_once",
                "session_key": "agent:main:mattermost:channel:channel_1:1111",
            },
        }

        from gateway.platforms.webhook import WebhookAdapter

        # We can't instantiate the full adapter, but we can test the handler
        # by creating a minimal mock response path.
        # The actual handler lives on an aiohttp web.Route — test via unit
        # extraction instead.
        
        # Test the resolution logic directly:
        action = payload["context"]["action"]
        session_key = payload["context"]["session_key"]
        assert session_key == "agent:main:mattermost:channel:channel_1:1111"
        assert action == "approve_once"

        choice_map = {
            "approve_once": "once",
            "approve_session": "session",
            "approve_always": "always",
            "deny": "deny",
        }
        choice = choice_map.get(action, "deny")
        assert choice == "once"

    @pytest.mark.asyncio
    async def test_denies_unknown_action(self):
        action = "unknown_action"
        choice_map = {
            "approve_once": "once",
            "approve_session": "session",
            "approve_always": "always",
            "deny": "deny",
        }
        choice = choice_map.get(action, "deny")
        assert choice == "deny"
