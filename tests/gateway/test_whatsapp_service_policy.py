"""Tests for WhatsApp service-conversation policy defaults and normalization."""

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.whatsapp_service_policy import (
    WhatsAppServiceConversationConfig,
    WhatsAppServiceConversationPolicy,
    normalize_whatsapp_chat_id,
)


def test_normalize_whatsapp_chat_id_for_phone_number():
    assert normalize_whatsapp_chat_id("+34 600-111-222") == "34600111222@s.whatsapp.net"


def test_normalize_whatsapp_chat_id_preserves_jid_forms():
    assert normalize_whatsapp_chat_id("34600111222@s.whatsapp.net") == "34600111222@s.whatsapp.net"
    assert normalize_whatsapp_chat_id("12345678901234@lid") == "12345678901234@lid"
    assert normalize_whatsapp_chat_id("12345@g.us") == "12345@g.us"


def test_normalize_whatsapp_chat_id_rejects_human_label():
    assert normalize_whatsapp_chat_id("Movistar Support (dm)") is None


def test_service_conversation_config_defaults_are_conservative():
    cfg = WhatsAppServiceConversationConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.default_mode == "draft_first"
    assert cfg.require_explicit_approval_for_first_contact is True
    assert cfg.allow_inbound_media is False
    assert cfg.allow_outbound_media is False
    assert cfg.allow_agent_initiated_service_chats is False
    assert cfg.max_new_service_chats_per_day == 3


def test_service_conversation_config_normalizes_approved_chats():
    cfg = WhatsAppServiceConversationConfig.from_dict(
        {
            "enabled": True,
            "approved_chats": [
                "+34 600-111-222",
                "34600111222@s.whatsapp.net",
                "Movistar Support (dm)",
            ],
        }
    )
    assert cfg.enabled is True
    assert cfg.approved_chats == ["34600111222@s.whatsapp.net"]


def test_policy_accepts_only_approved_chats_when_enabled():
    platform_cfg = PlatformConfig(enabled=True, extra={
        "service_conversations": {
            "enabled": True,
            "approved_chats": ["+34 600-111-222"],
        }
    })
    gateway_cfg = GatewayConfig(platforms={Platform.WHATSAPP: platform_cfg})
    policy = WhatsAppServiceConversationPolicy.from_gateway_config(gateway_cfg)

    assert policy.can_accept_inbound("34600111222@s.whatsapp.net") is True
    assert policy.can_accept_inbound("+34 600-111-222") is True
    assert policy.can_accept_inbound("34999999999@s.whatsapp.net") is False


def test_policy_disallows_operator_commands_from_provider_chat():
    policy = WhatsAppServiceConversationPolicy(
        WhatsAppServiceConversationConfig.from_dict({"enabled": True, "approved_chats": ["+34 600-111-222"]})
    )
    assert policy.is_operator_command_allowed(from_approved_service_chat=True) is False
    assert policy.is_operator_command_allowed(from_approved_service_chat=False) is True
