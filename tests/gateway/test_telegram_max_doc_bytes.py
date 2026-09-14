"""Tests for Telegram document-size cap.

The public Telegram Bot API caps `getFile` at 20MB. A locally-hosted
`telegram-bot-api` server raises that ceiling to 2GB. We treat the presence
of `extra.base_url` as the explicit opt-in to the higher cap.
"""


from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def test_max_doc_bytes_raised_to_2gb_when_base_url_set():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"base_url": "http://localhost:8081/bot"},
        )
    )
    assert adapter._max_doc_bytes == 2 * 1024 * 1024 * 1024


