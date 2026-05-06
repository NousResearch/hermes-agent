from tools.send_message_tool import _parse_target_ref


def test_parse_target_ref_accepts_whatsapp_lid_jid():
    assert _parse_target_ref("whatsapp", "15551234567@lid") == (
        "15551234567@lid",
        None,
        True,
    )


def test_parse_target_ref_accepts_whatsapp_phone_jid():
    assert _parse_target_ref("whatsapp", "15551234567@s.whatsapp.net") == (
        "15551234567@s.whatsapp.net",
        None,
        True,
    )


def test_parse_target_ref_accepts_whatsapp_group_jid():
    assert _parse_target_ref("whatsapp", "120363001234567890@g.us") == (
        "120363001234567890@g.us",
        None,
        True,
    )


def test_parse_target_ref_accepts_legacy_e164_whatsapp_target():
    assert _parse_target_ref("whatsapp", "+15551234567") == (
        "+15551234567",
        None,
        True,
    )
