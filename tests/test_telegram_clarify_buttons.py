import asyncio
import types

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def test_clarify_choices_render_only_as_text_buttons_not_numbered_body():
    async def run_test():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = object()

    captured = {}

    async def fake_send_message_with_thread_fallback(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(message_id=123)

    adapter._send_message_with_thread_fallback = fake_send_message_with_thread_fallback

    result = await adapter.send_clarify(
        chat_id="12345",
        question="מה מפריע שם בעיקר?",
        choices=["עצה מוקדמת", "לא מכיר אותי", "לא הבין קושי", "הכול נכון"],
        clarify_id="clarify-1",
        session_key="telegram:12345",
    )

    assert result.success is True
    assert captured["text"] == "❓ מה מפריע שם בעיקר?"
    assert "1." not in captured["text"]
    assert "2." not in captured["text"]
    assert "עצה מוקדמת" not in captured["text"]

    keyboard = captured["reply_markup"].inline_keyboard
    labels = [button.text for row in keyboard for button in row]
    assert labels[:4] == ["עצה מוקדמת", "לא מכיר אותי", "לא הבין קושי", "הכול נכון"]
    assert all(label not in {"1", "2", "3", "4"} for label in labels)
