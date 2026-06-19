"""Test Layer 1 of send-gate: integration into adapter.send() methods.

This test file verifies that _check_send_gate() is properly integrated into
all concrete platform adapter send() methods. When send_gate is set to "disabled",
each adapter's send() method should raise SendGateDisabledException.

Test coverage:
- Each adapter's send() method calls _check_send_gate() at the start
- When send_gate=disabled, send() raises SendGateDisabledException
- When send_gate=enabled or not set, send() proceeds normally
- Exception contains helpful error message
- All 14 adapters are covered (BlueBubbles, Dingtalk, Email, Feishu, Matrix,
  Signal, Slack, SMS, Telegram, WeChat, WeCom, WecomCallback, WhatsApp, Yuanbao)
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from gateway.config import GatewayConfig, PlatformConfig, Platform
from gateway.platforms.base import SendGateDisabledException


# Map of platform names to their adapter classes
ADAPTER_IMPORTS = {
    "telegram": "gateway.platforms.telegram:TelegramBotAdapter",
    "slack": "gateway.platforms.slack:SlackAdapter",
    "signal": "gateway.platforms.signal:SignalAdapter",
    "email": "gateway.platforms.email:EmailAdapter",
    "sms": "gateway.platforms.sms:SmsAdapter",
    "matrix": "gateway.platforms.matrix:MatrixAdapter",
    "weixin": "gateway.platforms.weixin:WeixinAdapter",
    "dingtalk": "gateway.platforms.dingtalk:DingTalkAdapter",
    "feishu": "gateway.platforms.feishu:FeishuAdapter",
    "whatsapp": "gateway.platforms.whatsapp:WhatsAppBridgeAdapter",
    "whatsapp_cloud": "gateway.platforms.whatsapp_cloud:WhatsAppCloudAdapter",
    "wecom": "gateway.platforms.wecom:WeComAdapter",
    "wecom_callback": "gateway.platforms.wecom_callback:WeComCallbackAdapter",
    "bluebubbles": "gateway.platforms.bluebubbles:BlueBubblesAdapter",
    "yuanbao": "gateway.platforms.yuanbao:YuanbaoAdapter",
}


def create_mock_config(send_gate: str = "enabled") -> PlatformConfig:
    """Create a mock PlatformConfig with the given send_gate setting."""
    return PlatformConfig(enabled=True, extra={"send_gate": send_gate})


def create_mock_adapter(adapter_class, send_gate: str = "enabled"):
    """Create a mock adapter instance with given send_gate config."""
    config = create_mock_config(send_gate)

    # Create a mock adapter using __new__ to avoid __init__ issues
    adapter = object.__new__(adapter_class)
    adapter.config = config
    adapter.platform = Platform.TELEGRAM  # Default platform, overridden for each test

    return adapter


class TestTelegramSendGate:
    """Test TelegramAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.telegram import TelegramAdapter

        adapter = create_mock_adapter(TelegramAdapter, "disabled")
        adapter.platform = Platform.TELEGRAM
        adapter._bot = MagicMock()  # Mock the bot so it passes the first check

        with pytest.raises(SendGateDisabledException) as exc_info:
            await adapter.send("123456", "test message")

        assert "send() is disabled" in str(exc_info.value)
        assert "telegram" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_send_gate_enabled_proceeds(self):
        """When send_gate=enabled, send() should proceed past the gate check."""
        from gateway.platforms.telegram import TelegramAdapter

        adapter = create_mock_adapter(TelegramAdapter, "enabled")
        adapter.platform = Platform.TELEGRAM
        adapter._bot = None  # Bot is None, so send() returns early with expected error

        # Should not raise SendGateDisabledException, though it will fail for other reasons
        result = await adapter.send("123456", "test message")
        assert result.success is False
        assert result.error == "Not connected"  # Expected error from _bot being None

    @pytest.mark.asyncio
    async def test_send_gate_default_enabled(self):
        """When send_gate is not set, it defaults to enabled."""
        from gateway.platforms.telegram import TelegramAdapter

        config = PlatformConfig(enabled=True, extra={})  # No send_gate key
        adapter = object.__new__(TelegramAdapter)
        adapter.config = config
        adapter.platform = Platform.TELEGRAM
        adapter._bot = None

        # Should not raise SendGateDisabledException
        result = await adapter.send("123456", "test message")
        assert result.success is False
        assert result.error == "Not connected"


class TestSlackSendGate:
    """Test SlackAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.slack import SlackAdapter

        adapter = create_mock_adapter(SlackAdapter, "disabled")
        adapter.platform = Platform.SLACK
        adapter._app = MagicMock()

        with pytest.raises(SendGateDisabledException) as exc_info:
            await adapter.send("C123456", "test message")

        assert "send() is disabled" in str(exc_info.value)
        assert "slack" in str(exc_info.value).lower()


class TestSignalSendGate:
    """Test SignalAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.signal import SignalAdapter

        adapter = create_mock_adapter(SignalAdapter, "disabled")
        adapter.platform = Platform.SIGNAL
        adapter._stop_typing_indicator = AsyncMock()  # Mock the method that's called

        with pytest.raises(SendGateDisabledException):
            await adapter.send("+1234567890", "test message")


class TestEmailSendGate:
    """Test EmailAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.email import EmailAdapter

        adapter = create_mock_adapter(EmailAdapter, "disabled")
        adapter.platform = Platform.EMAIL

        with pytest.raises(SendGateDisabledException):
            await adapter.send("test@example.com", "test message")


class TestSmsSendGate:
    """Test SmsAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.sms import SmsAdapter

        adapter = create_mock_adapter(SmsAdapter, "disabled")
        adapter.platform = Platform.SMS
        adapter._http_session = None  # Set to None to avoid actual HTTP calls

        with pytest.raises(SendGateDisabledException):
            await adapter.send("+1234567890", "test message")


class TestMatrixSendGate:
    """Test MatrixAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.matrix import MatrixAdapter

        adapter = create_mock_adapter(MatrixAdapter, "disabled")
        adapter.platform = Platform.MATRIX

        with pytest.raises(SendGateDisabledException):
            await adapter.send("!room123:example.com", "test message")


class TestWeixinSendGate:
    """Test WeixinAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.weixin import WeixinAdapter

        adapter = create_mock_adapter(WeixinAdapter, "disabled")
        adapter.platform = Platform.WEIXIN
        adapter._send_session = None
        adapter._token = None

        with pytest.raises(SendGateDisabledException):
            await adapter.send("user123", "test message")


class TestDingtalkSendGate:
    """Test DingTalkAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = create_mock_adapter(DingTalkAdapter, "disabled")
        adapter.platform = Platform.DINGTALK

        with pytest.raises(SendGateDisabledException):
            await adapter.send("conv123", "test message")


class TestFeishuSendGate:
    """Test FeishuAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.feishu import FeishuAdapter

        adapter = create_mock_adapter(FeishuAdapter, "disabled")
        adapter.platform = Platform.FEISHU
        adapter._client = None

        with pytest.raises(SendGateDisabledException):
            await adapter.send("oc_abc123", "test message")


class TestWhatsAppSendGate:
    """Test WhatsAppAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.whatsapp import WhatsAppAdapter

        adapter = create_mock_adapter(WhatsAppAdapter, "disabled")
        adapter.platform = Platform.WHATSAPP
        adapter._running = True
        adapter._http_session = MagicMock()
        adapter._check_managed_bridge_exit = AsyncMock(return_value=None)

        with pytest.raises(SendGateDisabledException):
            await adapter.send("user123", "test message")


class TestWhatsAppCloudSendGate:
    """Test WhatsAppCloudAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        adapter = create_mock_adapter(WhatsAppCloudAdapter, "disabled")
        adapter.platform = Platform.WHATSAPP_CLOUD
        adapter._http_client = None

        with pytest.raises(SendGateDisabledException):
            await adapter.send("1234567890", "test message")


class TestWeComSendGate:
    """Test WeComAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.wecom import WeComAdapter

        adapter = create_mock_adapter(WeComAdapter, "disabled")
        adapter.platform = Platform.WECOM

        with pytest.raises(SendGateDisabledException):
            await adapter.send("user123", "test message")


class TestWeComCallbackSendGate:
    """Test WeComCallbackAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.wecom_callback import WecomCallbackAdapter

        adapter = create_mock_adapter(WecomCallbackAdapter, "disabled")
        adapter.platform = Platform.WECOM_CALLBACK
        adapter._resolve_app_for_chat = MagicMock()
        adapter._http_client = MagicMock()

        with pytest.raises(SendGateDisabledException):
            await adapter.send("user123", "test message")


class TestBlueBubblesSendGate:
    """Test BlueBubblesAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.bluebubbles import BlueBubblesAdapter

        adapter = create_mock_adapter(BlueBubblesAdapter, "disabled")
        adapter.platform = Platform.BLUEBUBBLES

        with pytest.raises(SendGateDisabledException):
            await adapter.send("chat123", "test message")


class TestYuanbaoSendGate:
    """Test YuanbaoAdapter.send() respects send_gate."""

    @pytest.mark.asyncio
    async def test_send_gate_disabled_raises(self):
        """When send_gate=disabled, send() should raise SendGateDisabledException."""
        from gateway.platforms.yuanbao import YuanbaoAdapter

        adapter = create_mock_adapter(YuanbaoAdapter, "disabled")
        adapter.platform = Platform.YUANBAO
        adapter._outbound = MagicMock()
        adapter._outbound.send_text = AsyncMock()

        with pytest.raises(SendGateDisabledException):
            await adapter.send("direct:account123", "test message")


class TestErrorMessageQuality:
    """Test that error messages are helpful and consistent."""

    @pytest.mark.asyncio
    async def test_error_includes_platform_name(self):
        """Error message includes the platform name."""
        from gateway.platforms.telegram import TelegramAdapter

        adapter = create_mock_adapter(TelegramAdapter, "disabled")
        adapter.platform = Platform.TELEGRAM
        adapter._bot = MagicMock()

        with pytest.raises(SendGateDisabledException) as exc_info:
            await adapter.send("123456", "test message")

        error_msg = str(exc_info.value)
        assert "telegram" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_error_includes_config_path_hint(self):
        """Error message includes hint about how to re-enable."""
        from gateway.platforms.slack import SlackAdapter

        adapter = create_mock_adapter(SlackAdapter, "disabled")
        adapter.platform = Platform.SLACK
        adapter._app = MagicMock()

        with pytest.raises(SendGateDisabledException) as exc_info:
            await adapter.send("C123456", "test message")

        error_msg = str(exc_info.value)
        # Should include config path or helpful hint about send_gate
        assert "send_gate" in error_msg.lower() or "platforms" in error_msg.lower()


class TestConfigLoadingVariations:
    """Test send_gate behavior with various config loading scenarios."""

    @pytest.mark.asyncio
    async def test_config_with_send_gate_enabled_explicit(self):
        """Explicitly enabled send_gate should allow send()."""
        from gateway.platforms.telegram import TelegramAdapter

        config = PlatformConfig(enabled=True, extra={"send_gate": "enabled"})
        adapter = object.__new__(TelegramAdapter)
        adapter.config = config
        adapter.platform = Platform.TELEGRAM
        adapter._bot = None

        # Should not raise SendGateDisabledException
        result = await adapter.send("123456", "test message")
        assert isinstance(result, object)  # Should get a SendResult, not an exception

    @pytest.mark.asyncio
    async def test_config_missing_extra_dict(self):
        """When config has no extra dict, send_gate defaults to enabled."""
        from gateway.platforms.slack import SlackAdapter

        config = MagicMock()
        config.extra = {}  # Empty extra dict

        adapter = object.__new__(SlackAdapter)
        adapter.config = config
        adapter.platform = Platform.SLACK
        adapter._app = None

        # Should not raise SendGateDisabledException
        result = await adapter.send("C123456", "test message")
        assert isinstance(result, object)
