from plugins.platforms.rubika.inbound import parse_update, parse_inline_message, ParsedMessage


def test_parse_update_text_message():
    raw = {
        "type": "NewMessage",
        "chat_id": "c123",
        "new_message": {
            "message_id": "m1",
            "text": "hello",
            "sender_id": "u1",
            "reply_to_message_id": None,
            "aux_data": None,
        },
        "chat_type": "User",
    }
    parsed = parse_update(raw)
    assert parsed == ParsedMessage(
        chat_id="c123", sender_id="u1", text="hello", message_id="m1",
        is_group=False, reply_to_message_id=None, aux_data=None)


def test_parse_update_group_chat_type():
    raw = {
        "type": "NewMessage", "chat_id": "g1",
        "new_message": {"message_id": "m2", "text": "hi group", "sender_id": "u2",
                        "reply_to_message_id": "m0", "aux_data": None},
        "chat_type": "Group",
    }
    parsed = parse_update(raw)
    assert parsed.is_group is True
    assert parsed.reply_to_message_id == "m0"


def test_parse_inline_message_button_press():
    raw = {
        "chat_id": "c123", "message_id": "m5", "sender_id": "u1",
        "aux_data": {"button_id": "confirm_yes"},
    }
    parsed = parse_inline_message(raw)
    assert parsed.chat_id == "c123"
    assert parsed.sender_id == "u1"
    assert parsed.message_id == "m5"
    assert parsed.text == "confirm_yes"
    assert parsed.aux_data == {"button_id": "confirm_yes"}
    assert parsed.is_group is False
