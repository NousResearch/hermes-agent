from gateway.platforms.telegram import TelegramAdapter


def test_telegram_clarify_choice_split_label_and_description():
    adapter = TelegramAdapter.__new__(TelegramAdapter)

    choice = adapter._normalize_clarify_choice("Fast — Shortest path", 0)

    assert choice == {
        "label": "Fast",
        "description": "Shortest path",
        "value": "Fast — Shortest path",
    }


def test_telegram_clarify_choice_object_uses_value():
    adapter = TelegramAdapter.__new__(TelegramAdapter)

    choice = adapter._normalize_clarify_choice(
        {"label": "Careful", "description": "More verification", "value": "careful"},
        1,
    )

    assert choice == {
        "label": "Careful",
        "description": "More verification",
        "value": "careful",
    }
