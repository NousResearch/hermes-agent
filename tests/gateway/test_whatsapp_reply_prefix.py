"""Tests for WhatsApp reply_prefix config.yaml support.

Covers:
- config.yaml whatsapp.reply_prefix bridging into PlatformConfig.extra
- WhatsAppAdapter reading reply_prefix from config.extra
- Bridge subprocess receiving WHATSAPP_REPLY_PREFIX env var
- config.yaml whatsapp.send_read_receipts bridging into PlatformConfig.extra
- WhatsAppAdapter parsing send_read_receipts as a boolean
- Config version covers all ENV_VARS_BY_VERSION keys (regression guard)
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType


class _AsyncResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Config bridging from config.yaml
# ---------------------------------------------------------------------------


class TestConfigYamlBridging:
    """Test that whatsapp.reply_prefix in config.yaml flows into PlatformConfig."""

    def test_reply_prefix_bridged_from_yaml(self, tmp_path):
        """whatsapp.reply_prefix in config.yaml sets PlatformConfig.extra."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text('whatsapp:\n  reply_prefix: "Custom Bot"\n')

        with patch("gateway.config.get_hermes_home", return_value=tmp_path):
            from gateway.config import load_gateway_config
            # Need to also patch WHATSAPP_ENABLED so the platform exists
            with patch.dict("os.environ", {"WHATSAPP_ENABLED": "true"}, clear=False):
                config = load_gateway_config()

        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert wa_config.extra.get("reply_prefix") == "Custom Bot"

    def test_empty_reply_prefix_bridged(self, tmp_path):
        """Empty string reply_prefix disables the header."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text('whatsapp:\n  reply_prefix: ""\n')

        with patch("gateway.config.get_hermes_home", return_value=tmp_path):
            from gateway.config import load_gateway_config
            with patch.dict("os.environ", {"WHATSAPP_ENABLED": "true"}, clear=False):
                config = load_gateway_config()

        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert wa_config.extra.get("reply_prefix") == ""


# ---------------------------------------------------------------------------
# WhatsAppAdapter __init__
# ---------------------------------------------------------------------------


class TestAdapterInit:
    """Test that WhatsAppAdapter reads reply_prefix from config.extra."""

    def test_reply_prefix_from_extra(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
        config = PlatformConfig(enabled=True, extra={"reply_prefix": "Bot\\n"})
        adapter = WhatsAppAdapter(config)
        assert adapter._reply_prefix == "Bot\\n"


class TestReadReceiptPolicyOrdering:
    @pytest.mark.asyncio
    async def test_accepted_receipt_key_is_sent_to_bridge(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(
            PlatformConfig(enabled=True, extra={"send_read_receipts": True})
        )
        response = SimpleNamespace(status=200)
        session = MagicMock()
        session.post.return_value = _AsyncResponseContext(response)
        adapter._http_session = session
        key = {
            "id": "incoming-1",
            "remoteJid": "120363001234567890@g.us",
            "participant": "15550001111@s.whatsapp.net",
            "fromMe": False,
        }

        await adapter._send_read_receipt({"readReceiptKey": key})

        assert session.post.call_args.kwargs["json"] == {"key": key}
        assert session.post.call_args.args[0].endswith("/read")


# ---------------------------------------------------------------------------
# Config version regression guard
# ---------------------------------------------------------------------------


class TestConfigVersionCoverage:
    """Ensure _config_version covers all ENV_VARS_BY_VERSION keys."""

    def test_default_config_version_covers_env_var_versions(self):
        """_config_version must be >= the highest ENV_VARS_BY_VERSION key."""
        from hermes_cli.config import DEFAULT_CONFIG, ENV_VARS_BY_VERSION
        assert DEFAULT_CONFIG["_config_version"] >= max(ENV_VARS_BY_VERSION)
