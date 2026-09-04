"""Credential-safety regression test for _resolve_telegram_profiles.

httpx's own internal request logger logs the full request URL at INFO level
-- including the bot token embedded in the URL path, since that's how
Telegram's Bot API requires auth (https://api.telegram.org/bot<TOKEN>/...).
hermes_logging.py's _NOISY_LOGGERS list happens to suppress "httpx" to
WARNING app-wide today (for noise reasons, not security), which prevents
this in practice -- but _resolve_telegram_profiles floors the "httpx"
logger's level itself, reversibly, around its own request so this doesn't
depend on that unrelated list staying the way it happens to be today.

This test proves the safety property directly: even with the "httpx" logger
explicitly set to DEBUG beforehand (simulating _NOISY_LOGGERS being
weakened, or a developer raising it for unrelated troubleshooting), a real
token run through _resolve_telegram_profiles never reaches captured log
output -- and the logger's level is restored afterward, not permanently
clamped.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from hermes_cli import web_server


FAKE_TOKEN = "123456:AAFakeNotARealTokenXYZ789secretvalue"


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    web_server._TELEGRAM_PROFILE_CACHE.clear()
    yield
    web_server._TELEGRAM_PROFILE_CACHE.clear()


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert FAKE_TOKEN in str(request.url)  # sanity: this IS the credential-bearing call
        return httpx.Response(
            200, json={"ok": True, "result": {"first_name": "Test", "username": "testuser"}}
        )

    return httpx.MockTransport(handler)


def test_token_never_reaches_log_output_even_with_httpx_logger_at_debug(monkeypatch, caplog):
    httpx_logger = logging.getLogger("httpx")
    original_level = httpx_logger.level
    # Simulate the exact scenario the defense is FOR: something (a future
    # change to _NOISY_LOGGERS, a developer debugging an unrelated network
    # issue) has made the "httpx" logger verbose again.
    httpx_logger.setLevel(logging.DEBUG)

    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = _mock_transport()
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setattr(
        "hermes_cli.config.load_env", lambda: {"TELEGRAM_BOT_TOKEN": FAKE_TOKEN}
    )

    try:
        with caplog.at_level(logging.DEBUG, logger="httpx"):
            result = asyncio.run(web_server._resolve_telegram_profiles(["555"]))
    finally:
        httpx_logger.setLevel(original_level)

    assert result["555"]["username"] == "testuser"

    all_log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_TOKEN not in all_log_text, (
        "bot token leaked into log output during _resolve_telegram_profiles"
    )


def test_httpx_logger_level_restored_after_call_not_permanently_clamped(monkeypatch):
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)

    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = _mock_transport()
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    monkeypatch.setattr(
        "hermes_cli.config.load_env", lambda: {"TELEGRAM_BOT_TOKEN": FAKE_TOKEN}
    )

    asyncio.run(web_server._resolve_telegram_profiles(["555"]))

    # Reversible, not a permanent side effect: a developer's intentional
    # DEBUG level (for unrelated troubleshooting) survives past this call.
    assert httpx_logger.level == logging.DEBUG
    httpx_logger.setLevel(logging.WARNING)  # restore test-suite-wide default


def test_no_token_configured_returns_empty_without_error(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {})
    result = asyncio.run(web_server._resolve_telegram_profiles(["555"]))
    assert result == {}
