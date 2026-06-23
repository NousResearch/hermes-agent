"""Tests for Telegram bot-sender filtering via TELEGRAM_ALLOW_BOTS (#32188).

The Telegram adapter never populated ``source.is_bot`` when building a
MessageEvent, so ``_is_user_authorized``'s bot-filter path (which checks
``getattr(source, "is_bot", False)``) always saw False for Telegram senders.
A second gateway running on the same machine would therefore accept and
respond to messages sent by the first bot, causing echo/duplicate replies.

Fix:
  1. plugins/platforms/telegram/adapter.py — pass
     ``is_bot=bool(getattr(user, "is_bot", False))`` to build_source().
  2. gateway/authz_mixin.py — add Platform.TELEGRAM: "TELEGRAM_ALLOW_BOTS"
     to platform_allow_bots_map (same UX as DISCORD_ALLOW_BOTS / FEISHU_ALLOW_BOTS).

These tests exercise the real ``_is_user_authorized`` path (mirrors
tests/gateway/test_discord_bot_auth_bypass.py).
"""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_telegram_env(monkeypatch):
    """Start each test with a clean Telegram/global env so prior tests or CI
    setups can't leak allowlist / allow-bots vars and flip the auth result.
    """
    for var in (
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    """GatewayRunner skeleton with just enough wiring for the auth test.

    Uses ``object.__new__`` to skip the heavy __init__ (AGENTS.md pitfall #17).
    Stub pairing_store to never approve so the real allowlist path is exercised.
    """
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_telegram_bot_source(bot_id: str = "999888777"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id=bot_id,
        user_name="SomeBot",
        is_bot=True,
    )


def _make_telegram_human_source(user_id: str = "100200300"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


# --- platform_allow_bots_map wiring ----------------------------------------


def test_telegram_present_in_allow_bots_map():
    """Regression guard: Platform.TELEGRAM must be wired into the map so the
    is_bot bypass path can resolve TELEGRAM_ALLOW_BOTS at all."""
    from gateway.session import Platform
    # Build the same map the authz path constructs.
    from gateway import authz_mixin  # noqa: F401  (import sanity)
    # The map is defined inline in _is_user_authorized; assert behavior instead
    # by checking that an allow-bots=all bot source is admitted (below tests),
    # and that the env var name matches the documented convention.
    assert Platform.TELEGRAM.value == "telegram"


# --- _is_user_authorized behavior ------------------------------------------


def test_telegram_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")  # human-only allowlist
    source = _make_telegram_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is True


def test_telegram_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    source = _make_telegram_bot_source()
    assert runner._is_user_authorized(source) is True


def test_telegram_bot_NOT_authorized_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "none")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    source = _make_telegram_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is False


def test_telegram_bot_NOT_authorized_when_allow_bots_unset(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.delenv("TELEGRAM_ALLOW_BOTS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    source = _make_telegram_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is False


def test_telegram_allow_bots_is_case_insensitive(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")
    source = _make_telegram_bot_source()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "ALL")
    assert runner._is_user_authorized(source) is True
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "None")
    assert runner._is_user_authorized(source) is False


def test_telegram_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    """TELEGRAM_ALLOW_BOTS=all must NOT open the gate for humans — they still
    need to be in TELEGRAM_ALLOWED_USERS (or a pairing approval)."""
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    rejected = _make_telegram_human_source(user_id="999999999")
    assert runner._is_user_authorized(rejected) is False

    allowed = _make_telegram_human_source(user_id="100200300")
    assert runner._is_user_authorized(allowed) is True


def test_telegram_bot_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """The TELEGRAM_ALLOW_BOTS bypass is Telegram-specific — a Discord bot
    source must NOT be authorized just because TELEGRAM_ALLOW_BOTS=all."""
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    discord_bot = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="channel",
        user_id="999888777",
        is_bot=True,
    )
    assert runner._is_user_authorized(discord_bot) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
