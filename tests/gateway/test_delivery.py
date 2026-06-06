"""Tests for the delivery routing module."""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import SendResult
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_platform_only_no_chat_id(self):
        target = DeliveryTarget.parse("discord")
        assert target.platform == Platform.DISCORD
        assert target.chat_id is None
        assert target.is_explicit is False

    def test_local_target(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_origin_with_source(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"

    def test_local_roundtrip(self):
        target = DeliveryTarget.parse("local")
        assert target.to_string() == "local"

    def test_platform_only_roundtrip(self):
        target = DeliveryTarget.parse("discord")
        assert target.to_string() == "discord"

    def test_explicit_chat_roundtrip(self):
        target = DeliveryTarget.parse("telegram:999")
        s = target.to_string()
        assert s == "telegram:999"

        reparsed = DeliveryTarget.parse(s)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "999"


class TestCaseSensitiveChatIdParsing:
    """Test that chat IDs preserve their original case (issue #11768)."""
    
    def test_slack_uppercase_chat_id_preserved(self):
        """Slack channel IDs like C123ABC should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"  # Should NOT be lowercased to c123abc
        assert target.is_explicit is True
    
    def test_slack_chat_id_with_thread_preserved(self):
        """Slack channel:thread IDs should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC:thread123")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"
        assert target.thread_id == "thread123"
    
    def test_matrix_room_id_preserved(self):
        """Matrix room IDs like !RoomABC:example.org should preserve case.
        
        Note: Matrix room IDs contain colons (e.g., !RoomABC:example.org).
        Due to the platform:chat_id:thread_id format, these are parsed as
        chat_id=!RoomABC and thread_id=example.org. This is a known limitation
        of the current format. The fix preserves case but doesn't change the
        parsing structure.
        """
        target = DeliveryTarget.parse("matrix:!RoomABC:example.org")
        assert target.platform == Platform.MATRIX
        # The room ID is split at the first colon after the platform prefix
        # This is a format limitation - the case is preserved but the structure is split
        assert target.chat_id == "!RoomABC"
        assert target.thread_id == "example.org"
    
    def test_mixed_case_chat_id_roundtrip(self):
        """Mixed-case chat IDs should survive parse-to_string roundtrip."""
        original = "telegram:ChatId123ABC"
        target = DeliveryTarget.parse(original)
        s = target.to_string()
        reparsed = DeliveryTarget.parse(s)
        assert reparsed.chat_id == "ChatId123ABC"


class TestPlatformNameCaseInsensitivity:
    """Test that platform names are case-insensitive."""
    
    def test_uppercase_platform_name(self):
        """Platform names should be case-insensitive."""
        target = DeliveryTarget.parse("TELEGRAM:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
    
    def test_mixed_case_platform_name(self):
        """Mixed-case platform names should work."""
        target = DeliveryTarget.parse("TeleGram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"

class RecordingAdapter:
    def __init__(self):
        self.calls = []
        self.ensure_dm_topic_calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}

    async def ensure_dm_topic(self, chat_id, topic_name, force_create=False):
        self.ensure_dm_topic_calls.append(
            {"chat_id": chat_id, "topic_name": topic_name, "force_create": force_create}
        )
        return "38049"


class StaleTopicAdapter:
    def __init__(self):
        self.calls = []
        self.ensure_dm_topic_calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": dict(metadata or {})})
        if len(self.calls) == 1:
            return SendResult(success=False, error="Bad Request: message thread not found")
        return SendResult(success=True, message_id="fresh-message")

    async def ensure_dm_topic(self, chat_id, topic_name, force_create=False):
        self.ensure_dm_topic_calls.append(
            {"chat_id": chat_id, "topic_name": topic_name, "force_create": force_create}
        )
        return "38064" if force_create else "32343"


@pytest.mark.asyncio
async def test_explicit_telegram_private_thread_requires_reply_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    with pytest.raises(RuntimeError, match="requires telegram_reply_to_message_id"):
        await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_named_telegram_private_topic_is_created_before_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:Hermes API Test")

    await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.ensure_dm_topic_calls == [
        {"chat_id": "722341991", "topic_name": "Hermes API Test", "force_create": False}
    ]
    assert adapter.calls == [
        {
            "chat_id": "722341991",
            "content": "hello",
            "metadata": {
                "thread_id": "38049",
                "telegram_dm_topic_created_for_send": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_named_telegram_private_topic_refreshes_stale_thread_id(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = StaleTopicAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:Personal")

    result = await router._deliver_to_platform(target, "hello", metadata=None)

    assert getattr(result, "message_id", None) == "fresh-message"
    assert adapter.ensure_dm_topic_calls == [
        {"chat_id": "722341991", "topic_name": "Personal", "force_create": False},
        {"chat_id": "722341991", "topic_name": "Personal", "force_create": True},
    ]
    assert [call["metadata"]["thread_id"] for call in adapter.calls] == ["32343", "38064"]
    assert all(call["metadata"]["telegram_dm_topic_created_for_send"] is True for call in adapter.calls)


@pytest.mark.asyncio
async def test_explicit_telegram_private_thread_uses_reply_fallback_with_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    await router._deliver_to_platform(
        target,
        "hello",
        metadata={"telegram_reply_to_message_id": "9001"},
    )

    assert adapter.calls == [
        {
            "chat_id": "722341991",
            "content": "hello",
            "metadata": {
                "telegram_reply_to_message_id": "9001",
                "thread_id": "32344",
                "telegram_dm_topic_reply_fallback": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_explicit_telegram_direct_messages_topic_metadata_is_respected(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    await router._deliver_to_platform(
        target,
        "hello",
        metadata={"telegram_direct_messages_topic_id": "32344"},
    )

    assert adapter.calls[0]["metadata"] == {"telegram_direct_messages_topic_id": "32344"}


@pytest.mark.asyncio
async def test_explicit_telegram_group_thread_does_not_mark_dm_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:-100123:42")

    await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.calls[0]["metadata"] == {"thread_id": "42"}


class FailingAdapter:
    async def send(self, chat_id, content, metadata=None):
        return SendResult(success=False, error="route failed", retryable=False)


@pytest.mark.asyncio
async def test_platform_send_failure_raises_for_delivery_result(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: FailingAdapter()})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    with pytest.raises(RuntimeError, match="route failed"):
        await router._deliver_to_platform(target, "hello", metadata={"telegram_reply_to_message_id": "9001"})


def _simulate_windows_codepage_write(monkeypatch):
    """Make ``Path.write_text`` behave like a non-UTF-8 Windows console.

    On Windows ``Path.write_text(data)`` with no ``encoding=`` encodes through
    the platform code page (cp1252), which raises ``UnicodeEncodeError`` for
    emoji/CJK/accented text. POSIX CI runs default to UTF-8 and would hide the
    regression, so we reproduce the Windows default deterministically: encode
    with cp1252 when the caller omits ``encoding=``, otherwise honor it.
    """
    import pathlib

    real_write_text = pathlib.Path.write_text

    def fake_write_text(self, data, encoding=None, *args, **kwargs):
        effective = encoding or "cp1252"
        data.encode(effective)  # mirrors the encode open() performs on write
        return real_write_text(self, data, encoding=effective, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", fake_write_text)


# Non-ASCII content larger than MAX_PLATFORM_OUTPUT (4000) to force the
# truncate-and-save branch in _deliver_to_platform.
_NON_ASCII_OVERSIZED = ("数据备份完成 🎉 résumé — " * 250)


@pytest.mark.asyncio
async def test_oversized_non_ascii_output_is_delivered_on_windows_codepage(tmp_path, monkeypatch):
    """Oversized cron output containing emoji/CJK must still be delivered.

    Without an explicit utf-8 encoding the full-output save raises
    UnicodeEncodeError on a Windows code page, aborting the whole
    truncate-and-send path so the user receives nothing.
    """
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    _simulate_windows_codepage_write(monkeypatch)

    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:12345")

    result = await router._deliver_to_platform(
        target, _NON_ASCII_OVERSIZED, metadata={"job_id": "nightly"}
    )

    # The truncated message reached the adapter unharmed.
    assert len(adapter.calls) == 1
    assert "🎉" in adapter.calls[0]["content"]
    assert "truncated, full output saved to" in adapter.calls[0]["content"]

    # The full-output backup was written and round-trips as UTF-8.
    saved = list((tmp_path / "cron" / "output").glob("nightly_*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == _NON_ASCII_OVERSIZED
    assert result["success"] is True


def test_local_delivery_writes_non_ascii_on_windows_codepage(tmp_path, monkeypatch):
    """Local file delivery must persist emoji/CJK content as UTF-8."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    _simulate_windows_codepage_write(monkeypatch)

    router = DeliveryRouter(GatewayConfig())

    result = router._deliver_local(
        "完了 ✅ café", job_id="job1", job_name="日次レポート", metadata=None
    )

    written = Path(result["path"]).read_text(encoding="utf-8")
    assert "完了 ✅ café" in written
    assert "日次レポート" in written
