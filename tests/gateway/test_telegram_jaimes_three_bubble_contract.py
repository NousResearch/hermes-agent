import inspect
import json
from unittest.mock import patch

from gateway.platforms.base import BasePlatformAdapter
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


def test_jaimes_final_prose_never_infers_buttons():
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
    assert markup is None


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


def test_jaimes_final_builds_completion_edit_for_current_active_card(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    script = tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts" / "jaimes_work_card.py"
    state_path.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("# test\n")
    state_path.write_text(json.dumps({"active_cards": {
        "current": {
            "status": "active",
            "key": "turn-123",
            "ack_message_id": "44",
            "objective": "Verify the KALEIDO dip against tactical entry gates",
            "telegram_chat_id": "-1003589561528",
            "telegram_thread_id": "17",
            "started_at": "2026-07-12T23:22:00Z",
        }
    }}))
    content = "Model: openai-codex/gpt-5.6-sol | Route: live market check | Why: verified\nComplete: Yes"
    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        command = _adapter()._jaimes_pre_final_card_command(
            "-1003589561528", content, "17", {"notify": True}
        )
    assert command is not None
    assert command[2] == "done"
    assert command[command.index("--key") + 1] == "turn-123"
    assert command[command.index("--now") + 1] == "summary sent"
    assert command[command.index("--model") + 1] == "openai-codex/gpt-5.6-sol"


def test_jaimes_final_send_does_not_rearm_typing_or_autofinalize_card():
    source = inspect.getsource(TelegramAdapter.send)
    assert "await self._jaimes_finalize_card_before_final" not in source
    assert "Final/direct sends may not have an active gateway refresh loop" not in source
    assert "jaimes_reply_markup = None" in source


def test_gateway_stops_typing_before_final_delivery():
    source = inspect.getsource(BasePlatformAdapter._process_message_background)
    response_branch = source.index("if response:")
    stop_at = source.index("await _stop_typing_task()", response_branch)
    send_at = source.index("result = await self._send_with_retry", response_branch)
    assert stop_at < send_at
