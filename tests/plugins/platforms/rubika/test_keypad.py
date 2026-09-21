from plugins.platforms.rubika.keypad import build_chat_keypad, build_inline_keypad


def test_build_chat_keypad_simple_buttons():
    result = build_chat_keypad([("yes", "✅ Yes"), ("no", "❌ No")])
    assert result == {
        "rows": [
            {"buttons": [{"id": "yes", "type": "Simple", "button_text": "✅ Yes"}]},
            {"buttons": [{"id": "no", "type": "Simple", "button_text": "❌ No"}]},
        ]
    }


def test_build_inline_keypad_simple_buttons():
    result = build_inline_keypad([("confirm_yes", "Confirm")])
    assert result == {
        "rows": [
            {"buttons": [{"id": "confirm_yes", "type": "Simple", "button_text": "Confirm"}]},
        ]
    }
