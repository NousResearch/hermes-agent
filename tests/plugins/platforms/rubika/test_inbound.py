from plugins.platforms.rubika.inbound import parse_update, ParsedMessage


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
