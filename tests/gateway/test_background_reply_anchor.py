"""Tests verifying that background task results preserve and forward reply anchors (msg_id).

Regression tests for QQBot and other platforms where omitting reply_to in background
task delivery results in messages being treated as unpermitted proactive messages.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.qqbot.adapter import QQAdapter
from gateway.session import SessionSource


def _make_runner():
    from gateway.run_turn import GatewayTurnMixin
    runner = object.__new__(GatewayTurnMixin)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    mock_store = MagicMock()
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store
    return runner


class TestBackgroundReplyAnchorPassing:
    """Verifies that _run_background_task forwards event_message_id as reply_to."""

    @pytest.mark.asyncio
    async def test_text_result_forwards_event_message_id_as_reply_to(self):
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.name = "mock_platform"
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent_1"))
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello background result"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello background result"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="user",
        )

        mock_result = {"final_response": "Hello background result", "messages": []}

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent

            await runner._run_background_task(
                "do work", source, "task_1", event_message_id="ANCHOR_MSG_123"
            )

        mock_adapter.send.assert_called_once()
        _, kwargs = mock_adapter.send.call_args
        assert kwargs.get("reply_to") == "ANCHOR_MSG_123", (
            f"Expected reply_to='ANCHOR_MSG_123', got {kwargs.get('reply_to')}"
        )

    @pytest.mark.asyncio
    async def test_fallback_empty_result_forwards_reply_to(self):
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.name = "mock_platform"
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent_empty"))
        mock_adapter.extract_media = MagicMock(return_value=([], ""))
        mock_adapter.extract_images = MagicMock(return_value=([], ""))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="user",
        )

        mock_result = {"final_response": "", "messages": []}

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent

            await runner._run_background_task(
                "empty prompt", source, "task_2", event_message_id="ANCHOR_EMPTY_456"
            )

        mock_adapter.send.assert_called_once()
        _, kwargs = mock_adapter.send.call_args
        assert kwargs.get("reply_to") == "ANCHOR_EMPTY_456"

    @pytest.mark.asyncio
    async def test_media_delivery_forwards_reply_to(self, tmp_path):
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.name = "mock_platform"
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent_text"))
        mock_adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="sent_doc"))
        mock_adapter.extract_images = MagicMock(return_value=([], "text"))

        test_file = tmp_path / "result.pdf"
        test_file.write_bytes(b"dummy")

        mock_adapter.extract_media = MagicMock(return_value=([(str(test_file), False)], "text"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="user",
        )

        mock_result = {"final_response": f"text MEDIA:{test_file}", "messages": []}

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent, \
             patch("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path.resolve(),)):
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent

            await runner._run_background_task(
                "doc task", source, "task_3", event_message_id="ANCHOR_MEDIA_789"
            )

        mock_adapter.send_document.assert_called_once()
        _, doc_kwargs = mock_adapter.send_document.call_args
        assert doc_kwargs.get("reply_to") == "ANCHOR_MEDIA_789"


class TestQQBotReplyAnchorIntegration:
    """Verifies QQAdapter properly converts reply_to into msg_id in the HTTP payload."""

    @pytest.mark.asyncio
    async def test_qqbot_send_populates_msg_id_when_reply_to_passed(self):
        adapter = QQAdapter(PlatformConfig(enabled=True, token="dummy_token"))
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["user_target"] = "c2c"

        captured = []
        async def mock_api(method, path, body=None, timeout=None):
            captured.append({"method": method, "path": path, "body": body})
            return {"id": "qq_resp_1"}

        adapter._api_request = mock_api

        result = await adapter.send(
            chat_id="user_target",
            content="Hello from QQ test",
            reply_to="QQ_INBOUND_MSG_OID_1001",
        )

        assert result.success is True
        assert len(captured) == 1
        body = captured[0]["body"]
        assert body.get("msg_id") == "QQ_INBOUND_MSG_OID_1001", (
            f"Expected msg_id in QQ API body, got {body}"
        )

    @pytest.mark.asyncio
    async def test_qqbot_send_metadata_fallback_populates_msg_id(self):
        adapter = QQAdapter(PlatformConfig(enabled=True, token="dummy_token"))
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["user_target"] = "c2c"

        captured = []
        async def mock_api(method, path, body=None, timeout=None):
            captured.append({"method": method, "path": path, "body": body})
            return {"id": "qq_resp_2"}

        adapter._api_request = mock_api

        # When caller passes metadata containing reply_to_message_id but reply_to=None
        result = await adapter.send(
            chat_id="user_target",
            content="Hello metadata fallback",
            reply_to=None,
            metadata={"reply_to_message_id": "QQ_FALLBACK_OID_2002"},
        )

        assert result.success is True
        assert len(captured) == 1
        body = captured[0]["body"]
        assert body.get("msg_id") == "QQ_FALLBACK_OID_2002"

    @pytest.mark.asyncio
    async def test_end_to_end_background_command_to_qqbot_api(self):
        """End-to-end sandbox verification: _run_background_task -> QQAdapter -> QQ OpenAPI."""
        runner = _make_runner()
        adapter = QQAdapter(PlatformConfig(enabled=True, token="dummy_token"))
        adapter._running = True
        adapter._ws = SimpleNamespace(closed=False)
        adapter._chat_type_map["user_openid_999"] = "c2c"

        captured = []
        async def mock_api(method, path, body=None, timeout=None):
            captured.append({"method": method, "path": path, "body": body})
            return {"id": "qq_delivered_msg_id"}

        adapter._api_request = mock_api
        runner.adapters[Platform.QQBOT] = adapter

        source = SessionSource(
            platform=Platform.QQBOT,
            user_id="user_openid_999",
            chat_id="user_openid_999",
            user_name="testuser",
        )

        mock_result = {
            "final_response": "这是后台运行出来的最终分析报告",
            "messages": [],
        }

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent

            await runner._run_background_task(
                "分析任务",
                source,
                "bg_task_verified",
                event_message_id="ROBOT1.0_QQ_INBOUND_ANCHOR_555",
            )

        assert len(captured) == 1
        req = captured[0]
        assert req["path"] == "/v2/users/user_openid_999/messages"
        assert req["body"]["msg_id"] == "ROBOT1.0_QQ_INBOUND_ANCHOR_555", (
            f"Expected msg_id='ROBOT1.0_QQ_INBOUND_ANCHOR_555', got: {req['body']}"
        )
        assert "这是后台运行出来的最终分析报告" in req["body"]["markdown"]["content"]
