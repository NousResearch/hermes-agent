"""Tests for Telegram per-channel prompt resolution."""

import sys
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _adapter(channel_prompts: dict[str, str]) -> TelegramAdapter:
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"channel_prompts": channel_prompts},
    )
    return TelegramAdapter(config)


def test_forum_topic_prompt_uses_chat_scoped_key():
    adapter = _adapter(
        {
            "-1001111111111": "Group A prompt",
            "-1001111111111:42": "Group A topic 42 prompt",
        }
    )

    assert adapter._resolve_channel_prompt("-1001111111111", "42") == "Group A topic 42 prompt"


def test_forum_topic_falls_back_to_own_parent_chat_prompt():
    adapter = _adapter(
        {
            "-1002222222222": "Group B prompt",
            "-1001111111111:42": "Group A topic 42 prompt",
        }
    )

    assert adapter._resolve_channel_prompt("-1002222222222", "42") == "Group B prompt"


def test_bare_forum_topic_id_does_not_cross_chat_boundaries():
    adapter = _adapter(
        {
            "42": "Legacy unscoped topic prompt",
            "-1002222222222": "Group B prompt",
        }
    )

    assert adapter._resolve_channel_prompt("-1002222222222", "42") == "Group B prompt"
