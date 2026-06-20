"""Telegram Operator Multiview gate-cache regressions."""

import json
from datetime import datetime, timezone

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource


def _event(*, chat_id="-100123", thread_id="42", text="Need approval"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_name="Ops",
            chat_type="group",
            user_id="7",
            user_name="Ada",
            thread_id=thread_id,
            message_id="10",
        ),
        message_id="10",
        platform_update_id=123,
        media_types=["text/plain"],
        timestamp=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
    )


def test_telegram_gate_cache_writes_matching_channel_events_only(tmp_path, monkeypatch):
    cache_path = tmp_path / "telegram-events.jsonl"
    monkeypatch.setenv("TELEGRAM_GATE_CHAT_ID", "-100123")
    monkeypatch.setenv("TELEGRAM_GATE_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_GATE_EVENTS_JSONL", str(cache_path))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    assert adapter._append_telegram_gate_event(_event(chat_id="-100999")) is False
    assert adapter._append_telegram_gate_event(_event(thread_id="99")) is False
    assert adapter._append_telegram_gate_event(_event()) is True

    lines = cache_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["platform"] == "telegram"
    assert record["direction"] == "inbound"
    assert record["chat_id"] == "-100123"
    assert record["thread_id"] == "42"
    assert record["message_id"] == "10"
    assert record["update_id"] == 123
    assert record["sender_name"] == "Ada"
    assert record["text"] == "Need approval"
    assert record["timestamp"] == "2026-06-17T12:00:00+00:00"
    assert record["media_types"] == ["text/plain"]
    assert "secret-token" not in lines[0]


def test_telegram_gate_cache_uses_home_channel_fallback(tmp_path, monkeypatch):
    cache_path = tmp_path / "home-events.jsonl"
    monkeypatch.delenv("TELEGRAM_GATE_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_GATE_THREAD_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_HOME_CHAT_ID", "-100123")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_GATE_EVENTS_JSONL", str(cache_path))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    assert adapter._append_telegram_gate_event(_event()) is True
    record = json.loads(cache_path.read_text())
    assert record["chat_id"] == "-100123"
    assert record["thread_id"] == "42"


def test_telegram_gate_cache_disabled_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_GATE_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_GATE_EVENTS_JSONL", str(tmp_path / "events.jsonl"))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    assert adapter._append_telegram_gate_event(_event()) is False
    assert not (tmp_path / "events.jsonl").exists()
