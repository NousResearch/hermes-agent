from unittest.mock import patch

from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class FakeButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _adapter():
    return object.__new__(TelegramAdapter)


def test_jaimes_final_normal_bubble_gets_corresponding_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**TLDR:**
- Work completed

**Challenges/Blockers:**
None

**Next steps for approval:**
1. Run live canary
2. Keep current configuration
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is not None
    assert [[button.text for button in row] for row in markup.inline_keyboard] == [["1", "2"]]
    assert not content.lstrip().startswith("```")


def test_jaimes_final_no_action_has_no_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**Next steps for approval:**
No action needed
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is None


def test_jaimes_final_no_action_with_explanation_has_no_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**Next steps for approval:**
- No action needed; future alerts use this layout
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is None
