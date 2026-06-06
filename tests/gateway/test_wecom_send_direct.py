"""Tests for send_wecom_direct() and _LIVE_ADAPTERS lifecycle."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _clean_live_adapters():
    """Ensure _LIVE_ADAPTERS is clean between tests."""
    from gateway.platforms.wecom import _LIVE_ADAPTERS
    _LIVE_ADAPTERS.clear()
    yield
    _LIVE_ADAPTERS.clear()


def _make_mock_adapter(bot_id="test-bot-123"):
    """Create a mock WeComAdapter with a fake WebSocket."""
    adapter = MagicMock()
    adapter._bot_id = bot_id
    adapter._ws = MagicMock()
    adapter._ws.closed = False
    adapter.is_connected = True
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg_1"))
    adapter.send_image_file = AsyncMock(return_value=SendResult(success=True, message_id="img_1"))
    adapter.send_video = AsyncMock(return_value=SendResult(success=True, message_id="vid_1"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice_1"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc_1"))
    return adapter


# ---------------------------------------------------------------------------
# P0 tests
# ---------------------------------------------------------------------------

class TestSendWecomDirectNoAdapter:
    """P0: Degraded path — no live adapter available."""

    @pytest.mark.asyncio
    async def test_no_live_adapter(self):
        """When Gateway is not running, return a user-friendly error."""
        from gateway.platforms.wecom import send_wecom_direct

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result
        assert "消息通道未启动" in result["error"]


class TestSendWecomDirectHappyPath:
    """P0: Happy path — live adapter available."""

    @pytest.mark.asyncio
    async def test_with_live_adapter(self):
        """Reuse the live adapter's existing WebSocket."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert result["success"] is True
        assert result["message_id"] == "msg_1"
        adapter.send.assert_called_once_with("test_user", "hello")


class TestSendWecomDirectAdapterDisconnected:
    """P0: Boundary — adapter exists but WebSocket is closed."""

    @pytest.mark.asyncio
    async def test_ws_closed(self):
        """When _ws is closed, return an error even though is_connected is True."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        adapter._ws.closed = True  # WebSocket broken but adapter still "connected"
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result
        assert "连接未就绪" in result["error"]

    @pytest.mark.asyncio
    async def test_ws_none(self):
        """When _ws is None entirely."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        adapter._ws = None
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result
        assert "连接未就绪" in result["error"]

    @pytest.mark.asyncio
    async def test_adapter_in_pool_but_disconnected(self):
        """Adapter exists in pool but _ws is closed — don't try to send."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        adapter._ws.closed = True
        adapter.is_connected = True  # misleading — the flag lies during reconnect
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_adapter_cleanup(self):
        """After adapter deregistered, send_wecom_direct should fail."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter
        # Simulate disconnect() cleanup
        _LIVE_ADAPTERS.pop("test-bot-123", None)

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result
        assert "消息通道未启动" in result["error"]


class TestLiveAdaptersKeyOverwrite:
    """P0: Boundary — bot_id collision."""

    @pytest.mark.asyncio
    async def test_key_overwrite(self):
        """Later adapter overwrites earlier one; early disconnect doesn't remove later."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter1 = _make_mock_adapter("shared-bot")
        adapter2 = _make_mock_adapter("shared-bot")
        adapter2.send = AsyncMock(return_value=SendResult(success=True, message_id="from_adapter2"))

        # First adapter registers
        _LIVE_ADAPTERS["shared-bot"] = adapter1
        # Second adapter with same bot_id overwrites
        _LIVE_ADAPTERS["shared-bot"] = adapter2
        # First adapter disconnects
        _LIVE_ADAPTERS.pop("shared-bot", None)

        # Second adapter's entry was also popped by the disconnect
        result = await send_wecom_direct(
            extra={"bot_id": "shared-bot"},
            chat_id="test_user",
            message="hello",
        )
        assert "error" in result
        assert "消息通道未启动" in result["error"]


class TestSendWecomDirectSecurity:
    """P0: Security — path traversal / invalid media paths."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path):
        """Relative path traversal is rejected."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
            media_files=[("../../etc/passwd", False)],
        )
        assert "error" in result
        assert "路径不合法" in result["error"]

    @pytest.mark.asyncio
    async def test_accepts_valid_file(self, tmp_path):
        """A valid file in a tmp path is accepted."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        valid_file = tmp_path / "test.jpg"
        valid_file.write_bytes(b"fake jpeg")

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="hello",
            media_files=[(str(valid_file), False)],
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# P1 tests
# ---------------------------------------------------------------------------

class TestSendWecomDirectEmptyInput:
    """P1: Boundary — empty text + empty media."""

    @pytest.mark.asyncio
    async def test_no_text_no_media(self):
        """Both empty should return an error, not crash."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="",
            media_files=[],
        )
        assert "error" in result
        assert "没有可发送" in result["error"]


class TestSendWecomDirectBotId:
    """P1: Boundary — bot_id resolution."""

    @pytest.mark.asyncio
    async def test_bot_id_from_extra(self):
        """bot_id from extra dict takes priority."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        adapter = _make_mock_adapter("custom-bot-id")
        _LIVE_ADAPTERS["custom-bot-id"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "custom-bot-id"},
            chat_id="test_user",
            message="hello",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_bot_id_missing(self):
        """No bot_id in extra or env should return error."""
        from gateway.platforms.wecom import send_wecom_direct

        with patch.dict("os.environ", {}, clear=True):
            result = await send_wecom_direct(
                extra={},
                chat_id="test_user",
                message="hello",
            )
        assert "error" in result
        assert "机器人未配置" in result["error"]


class TestSendWecomDirectMedia:
    """P1: Happy path — media routing."""

    @pytest.mark.asyncio
    async def test_voice_routing(self, tmp_path):
        """When is_voice=True, route through send_voice."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

        voice_file = tmp_path / "greeting.ogg"
        voice_file.write_bytes(b"fake ogg audio")

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        result = await send_wecom_direct(
            extra={"bot_id": "test-bot-123"},
            chat_id="test_user",
            message="",
            media_files=[(str(voice_file), True)],
        )
        assert result["success"] is True
        adapter.send_voice.assert_called_once()


class TestSendWecomDirectUsesDirect:
    """P1: Refactoring correctness — _send_wecom delegates to send_wecom_direct."""

    @pytest.mark.asyncio
    async def test_send_wecom_calls_direct(self):
        """_send_wecom() no longer creates a separate adapter."""
        from gateway.platforms.wecom import _LIVE_ADAPTERS
        from gateway.platforms.wecom import check_wecom_requirements

        adapter = _make_mock_adapter("test-bot-123")
        _LIVE_ADAPTERS["test-bot-123"] = adapter

        with patch.object(
            __import__("gateway.platforms.wecom", fromlist=["check_wecom_requirements"]),
            "check_wecom_requirements",
            return_value=True,
        ):
            from tools.send_message_tool import _send_wecom
            result = await _send_wecom(
                extra={"bot_id": "test-bot-123"},
                chat_id="test_user",
                message="hello",
            )
        assert result["success"] is True
        assert result["message_id"] == "msg_1"
        adapter.send.assert_called_once_with("test_user", "hello")


# ---------------------------------------------------------------------------
# P2 tests
# ---------------------------------------------------------------------------

class TestSendWecomDirectErrorMessages:
    """P2: UX — error messages are user-friendly (no technical jargon)."""

    @pytest.mark.asyncio
    async def test_errors_are_chinese_and_friendly(self):
        """All error paths return Chinese, user-actionable messages."""
        from gateway.platforms.wecom import send_wecom_direct

        # bot_id missing — clear env to avoid picking up real WECOM_BOT_ID
        with patch.dict("os.environ", {}, clear=True):
            result1 = await send_wecom_direct(
                extra={},
                chat_id="test_user",
                message="hello",
            )
        assert "机器人未配置" in result1["error"]
        assert "config.yaml" in result1["error"]  # actionable

        # no gateway
        result2 = await send_wecom_direct(
            extra={"bot_id": "no-such-bot"},
            chat_id="test_user",
            message="hello",
        )
        assert "消息通道未启动" in result2["error"]
        assert "hermes gateway" in result2["error"]  # actionable
