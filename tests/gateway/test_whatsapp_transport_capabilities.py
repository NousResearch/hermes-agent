from gateway.config import PlatformConfig


def test_whatsapp_bundled_bridge_keeps_edit_support_by_default():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter.SUPPORTS_MESSAGE_EDITING is True


def test_whatsapp_custom_bridge_can_declare_final_only_delivery():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={"supports_message_editing": False},
        )
    )
    assert adapter.SUPPORTS_MESSAGE_EDITING is False
