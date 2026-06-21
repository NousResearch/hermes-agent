"""Tests for the delivery routing module."""

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


# ---------------------------------------------------------------------------
# Cron output truncation / adapter-aware chunking (issue #50126)
# ---------------------------------------------------------------------------

class ChunkingAdapter:
    """Adapter that declares splits_long_messages=True (like Discord/Telegram)."""
    splits_long_messages = True

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


class NonChunkingAdapter:
    """Adapter without splits_long_messages (default False — legacy behavior)."""

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


@pytest.mark.asyncio
async def test_long_output_truncated_for_non_chunking_adapter(tmp_path, monkeypatch):
    """Non-chunking adapters receive truncated content with a footer + file save."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job1"})

    delivered = adapter.calls[0]["content"]
    assert len(delivered) < 5000  # was truncated
    assert "truncated" in delivered.lower()
    assert "full output saved to" in delivered
    # Full output was saved to disk
    saved_files = list(tmp_path.glob("cron/output/job1_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == long_content


@pytest.mark.asyncio
async def test_long_output_preserved_for_chunking_adapter(tmp_path, monkeypatch):
    """Chunking adapters (splits_long_messages=True) receive the FULL content."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = ChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job2"})

    delivered = adapter.calls[0]["content"]
    assert delivered == long_content  # NOT truncated — adapter handles chunking
    assert "truncated" not in delivered.lower()
    # Full output still saved to disk as audit trail
    saved_files = list(tmp_path.glob("cron/output/job2_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == long_content


@pytest.mark.asyncio
async def test_short_output_never_truncated(tmp_path, monkeypatch):
    """Output under the limit passes through untouched for any adapter."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    short_content = "x" * 100
    await router._deliver_to_platform(target, short_content, metadata={"job_id": "job3"})

    assert adapter.calls[0]["content"] == short_content
    # Nothing saved to disk
    assert not list(tmp_path.glob("cron/output/*.txt"))


@pytest.mark.asyncio
async def test_env_override_changes_truncation_threshold(tmp_path, monkeypatch):
    """HERMES_DELIVERY_MAX_PLATFORM_OUTPUT env var overrides the default 4000."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_DELIVERY_MAX_PLATFORM_OUTPUT", "200")
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    content = "x" * 300  # over the env-override threshold of 200
    await router._deliver_to_platform(target, content, metadata={"job_id": "job4"})

    delivered = adapter.calls[0]["content"]
    assert len(delivered) < 300  # truncated because env lowered the bar
    assert "truncated" in delivered.lower()
    # Audit file saved (truncation path always saves when it truncates)
    saved_files = list(tmp_path.glob("cron/output/job4_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == content


@pytest.mark.asyncio
async def test_env_override_disable_truncation(tmp_path, monkeypatch):
    """Setting HERMES_DELIVERY_MAX_PLATFORM_OUTPUT=0 disables truncation entirely."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_DELIVERY_MAX_PLATFORM_OUTPUT", "0")
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    content = "x" * 10000
    await router._deliver_to_platform(target, content, metadata={"job_id": "job5"})

    # With max_output=0, truncation is disabled — even non-chunking adapters
    # receive the full content (they may error at the platform API level, but
    # that's the user's explicit choice).
    assert adapter.calls[0]["content"] == content
    # Audit file STILL saved — the audit threshold (4000) is independent of
    # the truncation setting.  Content (10000) exceeds it.
    saved_files = list(tmp_path.glob("cron/output/job5_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == content


@pytest.mark.asyncio
async def test_audit_save_failure_does_not_break_chunking_delivery(tmp_path, monkeypatch):
    """If the audit save fails (disk full, permissions), chunking adapters
    still receive the full content — the save is best-effort."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)

    adapter = ChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000

    call_count = {"n": 0}

    def failing_save(content, job_id):
        call_count["n"] += 1
        raise OSError("No space left on device")

    monkeypatch.setattr(router, "_save_full_output", failing_save)

    # Should NOT raise — audit failure is caught
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job6"})

    # Adapter still got the full content
    assert adapter.calls[0]["content"] == long_content
    # Save was attempted
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_audit_save_failure_does_not_break_non_chunking_delivery(tmp_path, monkeypatch):
    """If the audit save fails AND truncation is needed, the fallback save
    in Step 2 is NOT caught — the footer needs a valid path, so this is a
    real failure. But if content exceeds the audit threshold AND truncation
    is disabled (max_output=0), the caught Step 1 failure lets delivery
    proceed."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_DELIVERY_MAX_PLATFORM_OUTPUT", "0")

    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000

    def failing_save(content, job_id):
        raise OSError("No space left on device")

    monkeypatch.setattr(router, "_save_full_output", failing_save)

    # max_output=0 → no truncation → Step 1 failure is caught → delivery proceeds
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job7"})

    # Non-chunking adapter still got the full content (truncation disabled)
    assert adapter.calls[0]["content"] == long_content
