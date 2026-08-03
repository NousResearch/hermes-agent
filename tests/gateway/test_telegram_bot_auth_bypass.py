"""Regression guard for Telegram bot-origin authorization (#32188)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_telegram_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_telegram_bot_source(bot_id: str = "999888777"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id=bot_id,
        user_name="OtherProfileBot",
        is_bot=True,
    )


def _make_telegram_human_source(user_id: str = "100200300"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


def test_telegram_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_telegram_bot_source("999888777")) is True


def test_telegram_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_telegram_bot_source()) is True


def test_telegram_bot_not_authorized_when_allow_bots_unset(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_telegram_bot_source("999888777")) is False


def test_telegram_bot_not_authorized_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "none")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_telegram_bot_source("999888777")) is False


def test_telegram_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_telegram_human_source("999999999")) is False
    assert runner._is_user_authorized(_make_telegram_human_source("100200300")) is True


def _build_telegram_message(*, is_bot: bool):
    user = SimpleNamespace(
        id=999888777 if is_bot else 100200300,
        full_name="OtherProfileBot" if is_bot else "Alice",
        is_bot=is_bot,
    )
    chat = SimpleNamespace(
        id=123,
        type="private",
        title=None,
        full_name="Alice",
        is_forum=False,
    )
    message = MagicMock()
    message.from_user = user
    message.chat = chat
    message.message_id = 4242
    message.message_thread_id = None
    message.is_topic_message = False
    message.forum_topic_created = None
    message.reply_to_message = None
    message.quote = None
    message.text = "hello"
    message.caption = None
    return message


def _capture_build_source_is_bot(is_bot: bool):
    from gateway.platforms.base import MessageType
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = SimpleNamespace(extra={})
    message = _build_telegram_message(is_bot=is_bot)
    captured: dict = {}

    def fake_build_source(**kwargs):
        captured.update(kwargs)
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(kwargs.get("chat_id") or ""),
            chat_type=kwargs.get("chat_type") or "dm",
            user_id=kwargs.get("user_id"),
            is_bot=kwargs.get("is_bot", False),
        )

    with patch.object(adapter, "build_source", side_effect=fake_build_source):
        try:
            adapter._build_message_event(message, MessageType.TEXT, update_id=1)
        except Exception:
            # The method may continue into PTB-specific optional fields after
            # source construction; this test only pins the source kwarg.
            pass

    return captured.get("is_bot")


def test_telegram_adapter_propagates_is_bot_true():
    assert _capture_build_source_is_bot(True) is True


def test_telegram_adapter_propagates_is_bot_false():
    assert _capture_build_source_is_bot(False) is False
