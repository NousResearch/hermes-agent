"""Tests for Telegram document-size cap.

The public Telegram Bot API caps `getFile` at 20MB. A locally-hosted
`telegram-bot-api` server raises that ceiling to 2GB. We treat the presence
of `extra.base_url` as the explicit opt-in to the higher cap.
"""

import sys
import enum
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    class _ParseMode(str, enum.Enum):
        MARKDOWN = "Markdown"
        MARKDOWN_V2 = "MarkdownV2"
        HTML = "HTML"

    class _ChatType(str, enum.Enum):
        PRIVATE = "private"
        GROUP = "group"
        SUPERGROUP = "supergroup"
        CHANNEL = "channel"

    telegram_mod.constants.ParseMode = _ParseMode
    telegram_mod.constants.ChatType = _ChatType
    # Also set directly for ``from telegram.constants import ...``
    telegram_mod.ParseMode = _ParseMode
    telegram_mod.ChatType = _ChatType

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def test_max_doc_bytes_defaults_to_20mb_without_base_url():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    assert adapter._max_doc_bytes == 20 * 1024 * 1024


def test_max_doc_bytes_raised_to_2gb_when_base_url_set():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"base_url": "http://localhost:8081/bot"},
        )
    )
    assert adapter._max_doc_bytes == 2 * 1024 * 1024 * 1024


def test_max_doc_bytes_empty_base_url_keeps_default():
    """An empty/falsy `base_url` should not flip the cap — only a real URL does."""
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="***", extra={"base_url": ""}),
    )
    assert adapter._max_doc_bytes == 20 * 1024 * 1024
