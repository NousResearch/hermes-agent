"""Tests for ``_resolve_telegram_file_url`` in the Telegram platform adapter.

python-telegram-bot builds media download URLs as
``<base_file_url><token>/<file_path>``. When a custom ``base_url`` is set
(reverse proxy in front of api.telegram.org, local telegram-bot-api server)
but ``base_file_url`` is missing — or was explicitly copied from ``base_url``
— the resolved URL ends in ``/bot`` and is the *API method* endpoint, not the
*file* endpoint. Every media download (voice note, audio, photo, document)
then hits a bogus API method, gets HTTP 404, and PTB raises
``telegram.error.InvalidToken`` ("Failed to cache voice: Not Found").

These tests pin the resolution + warning behavior for both broken
situations, the correct /file/ configuration, and the PTB-defaults path.
"""

import pytest

from plugins.platforms.telegram.adapter import _resolve_telegram_file_url


@pytest.mark.parametrize(
    ("extra", "expected_url", "expect_warning"),
    [
        # No custom base_url → PTB defaults apply, no warning.
        ({}, "", False),
        # Only base_url set → falls back to base_url (/bot), must warn.
        ({"base_url": "https://tg.example.com/bot"}, "https://tg.example.com/bot", True),
        # base_file_url explicitly copied from base_url → same broken URL, warn.
        (
            {
                "base_url": "https://tg.example.com/bot",
                "base_file_url": "https://tg.example.com/bot",
            },
            "https://tg.example.com/bot",
            True,
        ),
        # Correct file endpoint → no warning.
        (
            {
                "base_url": "https://tg.example.com/bot",
                "base_file_url": "https://tg.example.com/file/bot",
            },
            "https://tg.example.com/file/bot",
            False,
        ),
        # Local telegram-bot-api server (docs example) → no warning.
        (
            {
                "base_url": "http://127.0.0.1:8081/bot",
                "base_file_url": "http://127.0.0.1:8081/file/bot",
            },
            "http://127.0.0.1:8081/file/bot",
            False,
        ),
        # Suspicious suffix without a /file/ segment → soft warning.
        (
            {
                "base_url": "https://tg.example.com/bot",
                "base_file_url": "https://cdn.example.com/downloads",
            },
            "https://cdn.example.com/downloads",
            True,
        ),
    ],
)
def test_resolve_telegram_file_url(extra, expected_url, expect_warning):
    url, warning = _resolve_telegram_file_url(extra)
    assert url == expected_url
    assert (warning is not None) is expect_warning


def test_warning_suggests_file_endpoint():
    """The warning must point at the correct /file/bot URL for a /bot base_url."""
    _, warning = _resolve_telegram_file_url({"base_url": "https://tg2.rmg7.com/bot"})
    assert warning is not None
    assert "https://tg2.rmg7.com/file/bot" in warning
