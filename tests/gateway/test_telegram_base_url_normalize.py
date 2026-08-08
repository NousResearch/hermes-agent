"""Regression tests for plugins.platforms.telegram.adapter._normalize_telegram_base_url.

Background
----------
Issue #81788: on Linux, a Telegram gateway with a custom ``base_url`` configured
to ``https://api.telegram.org`` (missing the trailing ``/bot`` segment) fails to
connect with::

    WARNING hermes_plugins.telegram_platform.adapter: [Telegram] Connect
    attempt 1/8 failed: Unknown error in HTTP implementation: InvalidURL(
    "Invalid port: '<token-after-colon>'") -- retrying in 1s

python-telegram-bot's ``Bot._parse_base_url`` appends the bot token to whatever
URL is supplied as ``base_url``. With ``https://api.telegram.org`` as base,
PTB builds ``https://api.telegram.org<id>:<secret>/<endpoint>``. The
authority regex parses the post-colon half (``<secret>``) as the port number,
httpx rejects it, and the retry loop surfaces a "Unknown error" wrapper.

These tests pin down the helper that auto-normalizes a misconfigured base URL
to the ``/bot`` suffix PTB requires, and that it fails fast on clearly
malformed URLs (the same ``Invalid port`` class as #6360).
"""

from __future__ import annotations

import pytest

from plugins.platforms.telegram.adapter import _normalize_telegram_base_url


class TestNormalizeTelegramBaseUrl:
    def test_appends_bot_when_missing_suffix(self):
        """The exact symptom from #81788: base_url without /bot.

        Without the trailing ``/bot`` segment, PTB appends the token to the
        authority and httpx parses the post-colon half as a port. Auto-fix
        by appending ``/bot`` so the token lands in the path.
        """
        assert (
            _normalize_telegram_base_url("https://api.telegram.org")
            == "https://api.telegram.org/bot"
        )

    def test_appends_bot_for_trailing_slash(self):
        assert (
            _normalize_telegram_base_url("https://api.telegram.org/")
            == "https://api.telegram.org/bot"
        )

    def test_passes_through_already_correct_url(self):
        assert (
            _normalize_telegram_base_url("https://api.telegram.org/bot")
            == "https://api.telegram.org/bot"
        )

    def test_preserves_trailing_slash_when_suffix_already_present(self):
        # Don't strip a trailing slash that's already attached to /bot — the
        # builder treats both forms identically, but matching exactly what
        # the user wrote avoids surprise in log output.
        assert (
            _normalize_telegram_base_url("https://api.telegram.org/bot/")
            == "https://api.telegram.org/bot/"
        )

    def test_local_bot_api_server_without_suffix(self):
        """The local telegram-bot-api convention requires /bot too."""
        assert (
            _normalize_telegram_base_url("http://127.0.0.1:8081")
            == "http://127.0.0.1:8081/bot"
        )

    def test_local_bot_api_server_with_suffix(self):
        assert (
            _normalize_telegram_base_url("http://localhost:8081/bot")
            == "http://localhost:8081/bot"
        )

    def test_empty_string_passes_through(self):
        """Empty / unset base_url is the caller's signal to use the default.

        The helper must not invent a default on its own — ``builder.base_url``
        is only invoked when ``custom_base_url`` is truthy at the call site.
        """
        assert _normalize_telegram_base_url("") == ""

    def test_raises_clear_error_on_malformed_port(self):
        """Same root cause as #6360 (proxy env malformed by stray 'export').

        Fail fast with a message that names the offending config key, instead
        of letting the cryptic httpx ``InvalidURL("Invalid port: ...")`` leak
        out through PTB's ``NetworkError`` wrapper.
        """
        with pytest.raises(RuntimeError) as excinfo:
            _normalize_telegram_base_url("http://127.0.0.1:6153export")
        msg = str(excinfo.value)
        assert "Malformed Telegram base_url" in msg
        assert "http://127.0.0.1:6153export" in msg
        # The error must steer the operator at the right config key so they
        # don't have to grep through the retry loop traceback.
        assert "gateway.platforms.telegram.extra.base_url" in msg

    def test_raises_clear_error_on_scheme_less_url(self):
        """Team-review finding (#81788 follow-up): ``urlparse("//host")``
        does NOT raise — a scheme-less base_url slipped through the port
        validation and surfaced later as the same cryptic httpx
        ``InvalidURL`` at request time. Fail fast instead."""
        with pytest.raises(RuntimeError) as excinfo:
            _normalize_telegram_base_url("127.0.0.1:8081")
        msg = str(excinfo.value)
        assert "missing URL scheme" in msg
        assert "gateway.platforms.telegram.extra.base_url" in msg

    def test_passes_through_token_placeholder_suffix(self):
        """PTB's documented ``.../bot{token}`` form must not get a second
        ``/bot`` appended (would produce ``/bot{token}/bot``)."""
        assert (
            _normalize_telegram_base_url("https://api.telegram.org/bot{token}")
            == "https://api.telegram.org/bot{token}"
        )

    def test_appends_bot_when_token_placeholder_has_bare_path(self):
        """A bare path without the /bot suffix still gets it appended."""
        assert (
            _normalize_telegram_base_url("https://example.com")
            == "https://example.com/bot"
        )


class TestNormalizeProducesParseableUrl:
    """The end-to-end invariant: token append + httpx parse must succeed."""

    @pytest.mark.parametrize(
        "raw_base",
        [
            "https://api.telegram.org",        # the bug case
            "https://api.telegram.org/",
            "http://127.0.0.1:8081",
            "http://localhost:8081",
        ],
    )
    def test_normalized_url_parses_cleanly_with_real_token(self, raw_base):
        """The fix must produce a URL that httpx parses without InvalidURL.

        Before the fix, ``urlparse(base + token + '/getMe')`` raised
        ``InvalidURL("Invalid port: '<secret>'")``. After normalization to
        the ``/bot`` suffix, the same token-appended URL parses cleanly.
        """
        import httpx

        token = "1234567890:AAGURFJSzXoINq_Fj_Srvuf7mpZ4XRXO6rQ"
        normalized = _normalize_telegram_base_url(raw_base)
        # Simulate exactly what PTB does: base + token + /<endpoint>.
        url = f"{normalized}{token}/getMe"
        # Must NOT raise InvalidURL. host must be the configured api host,
        # port must be None (default scheme port).
        parsed = httpx.URL(url)
        assert parsed.host == "api.telegram.org" or parsed.host == "127.0.0.1" or parsed.host == "localhost"
        assert parsed.port is None or parsed.port in (80, 8081)
        assert token in parsed.path  # token lives in the path, not the authority