"""Tests for the delivery routing module."""

from pathlib import Path

import pytest
from typing import Any, cast

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True


class TestTransportReceiptContract:
    def test_transport_target_rejects_string_subclasses_without_magic_methods(self):
        from gateway.platforms.base import TransportTarget

        class HostileText(str):
            def __bool__(self):
                raise AssertionError("hostile text truthiness was evaluated")

            def __len__(self):
                raise AssertionError("hostile text length was evaluated")

            def __eq__(self, _other):
                raise AssertionError("hostile text was compared")

            def __hash__(self):
                raise AssertionError("hostile text was hashed")

        with pytest.raises(ValueError, match="platform must be"):
            TransportTarget(platform=HostileText("telegram"), chat_id="123")

    def test_transport_receipt_rejects_subclasses_before_magic_methods(self):
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget(platform="telegram", chat_id="123")

        class HostileText(str):
            def __bool__(self):
                raise AssertionError("hostile text truthiness was evaluated")

            def __hash__(self):
                raise AssertionError("hostile text was hashed")

            def __eq__(self, _other):
                raise AssertionError("hostile text was compared")

        with pytest.raises(ValueError, match="outcome must be"):
            TransportReceipt(outcome=HostileText("unknown"), requested_target=target)

        class HostileTarget(TransportTarget):
            pass

        with pytest.raises(TypeError, match="requested_target"):
            TransportReceipt(
                outcome="unknown",
                requested_target=HostileTarget("telegram", "123"),
            )

    def test_transport_receipt_rejects_datetime_subclasses_and_custom_tzinfo(self):
        from datetime import datetime, timedelta, timezone, tzinfo
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget(platform="telegram", chat_id="123")

        class HostileDateTime(datetime):
            def astimezone(self, *_args, **_kwargs):
                raise AssertionError("hostile datetime conversion was called")

        hostile_datetime = HostileDateTime.now(timezone.utc)
        with pytest.raises(ValueError, match="timezone-aware"):
            TransportReceipt(
                outcome="unknown", requested_target=target,
                observed_at=hostile_datetime,
            )

        class HostileTimezone(tzinfo):
            def utcoffset(self, _dt):
                raise AssertionError("hostile timezone offset was called")

            def dst(self, _dt):
                return timedelta(0)

        custom_tz_datetime = datetime(2026, 8, 23, tzinfo=HostileTimezone())
        with pytest.raises(ValueError, match="timezone-aware"):
            TransportReceipt(
                outcome="unknown", requested_target=target,
                observed_at=custom_tz_datetime,
            )

    def test_matrix_receipt_target_metadata_rejects_subclasses_without_methods(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        class HostileDict(dict):
            def __bool__(self):
                raise AssertionError("hostile metadata truthiness was evaluated")

            def get(self, *_args, **_kwargs):
                raise AssertionError("hostile metadata get was called")

        with pytest.raises(TypeError, match="metadata"):
            MatrixAdapter._transport_receipt_targets(
                "!room:example.org", HostileDict({})
            )
        with pytest.raises(TypeError, match="requested target"):
            MatrixAdapter._transport_receipt_targets(
                "!room:example.org",
                {"_transport_receipt_requested_target": HostileDict({})},
            )

    @pytest.mark.asyncio
    async def test_matrix_send_rejects_receipt_metadata_before_provider_dispatch(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = object.__new__(MatrixAdapter)
        adapter.plan_transport_text = lambda _content: ["bounded"]
        adapter._client = SimpleNamespace(send_message_event=AsyncMock())

        class HostileDict(dict):
            def __bool__(self):
                raise AssertionError("hostile metadata truthiness was evaluated")

            def get(self, *_args, **_kwargs):
                raise AssertionError("hostile metadata get was called")

        class HostileText(str):
            def __bool__(self):
                raise AssertionError("hostile content truthiness was evaluated")

            def __str__(self):
                raise AssertionError("hostile content was stringified")

        metadata_result = await adapter.send(
            "!room:example.org", "bounded", metadata=HostileDict({})
        )
        content_result = await adapter.send(
            "!room:example.org", HostileText("bounded"), metadata={}
        )

        for result in (metadata_result, content_result):
            assert result.success is False
            assert result.error_kind == "invalid_transport_receipt"
        adapter._client.send_message_event.assert_not_awaited()

        media_adapter = object.__new__(MatrixAdapter)
        media_adapter._client = SimpleNamespace(
            upload_media=AsyncMock(), send_message_event=AsyncMock(),
        )
        media_adapter._max_media_bytes = 1024
        media_adapter._encryption = False
        media_result = await media_adapter._upload_and_send(
            "!room:example.org", b"bounded", "report.pdf",
            "application/pdf", "m.file", metadata=HostileDict({}),
        )
        assert media_result.success is False
        assert media_result.error_kind == "invalid_transport_receipt"
        media_adapter._client.upload_media.assert_not_awaited()
        media_adapter._client.send_message_event.assert_not_awaited()

    def test_provider_message_id_normalization_never_calls_hostile_stringification(self):
        from gateway.platforms.base import normalize_transport_provider_message_id

        class HostileText(str):
            def __str__(self):
                raise AssertionError("hostile provider id was stringified")

            def __len__(self):
                raise AssertionError("hostile provider id length was evaluated")

        class HostileTextMeta(type):
            def __getattribute__(self, _name):
                raise AssertionError("hostile provider id type metadata was read")

        class HostileMetaText(str, metaclass=HostileTextMeta):
            pass

        class HostileObject:
            def __str__(self):
                raise AssertionError("provider object was stringified")

        class_spoof_calls = []

        class HostileClassSpoof:
            @property
            def __class__(self):
                class_spoof_calls.append("called")
                return str

        assert normalize_transport_provider_message_id(HostileText("provider-1")) == "provider-1"
        assert normalize_transport_provider_message_id(HostileMetaText("provider-2")) == "provider-2"
        assert normalize_transport_provider_message_id(42) == "42"
        assert normalize_transport_provider_message_id(HostileObject()) is None
        assert normalize_transport_provider_message_id(HostileClassSpoof()) is None
        assert class_spoof_calls == []
        assert normalize_transport_provider_message_id(True) is None

    def test_receipt_targets_are_typed_and_observation_is_always_aware(self):
        from datetime import datetime
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget(platform="telegram", chat_id="123")
        receipt = TransportReceipt(
            outcome="delivered", provider_message_id="1",
            requested_target=target, actual_target=target,
        )
        assert receipt.observed_at is not None
        assert receipt.observed_at.utcoffset() is not None
        for requested, actual in (({}, target), (target, object())):
            with pytest.raises(TypeError, match="TransportTarget"):
                TransportReceipt(
                    outcome="delivered", provider_message_id="1",
                    requested_target=requested, actual_target=actual,
                )
        with pytest.raises(ValueError, match="timezone-aware"):
            TransportReceipt(
                outcome="delivered", provider_message_id="1",
                requested_target=target, actual_target=target,
                observed_at=datetime.now(),
            )
        from datetime import timedelta, timezone

        for unbounded in (
            datetime(1969, 12, 31, tzinfo=timezone.utc),
            datetime.now(timezone.utc) + timedelta(minutes=6),
        ):
            with pytest.raises(ValueError, match="bounded"):
                TransportReceipt(
                    outcome="delivered", provider_message_id="1",
                    requested_target=target, actual_target=target,
                    observed_at=unbounded,
                )
        with pytest.raises(ValueError, match="ordinal"):
            TransportReceipt(
                outcome="delivered", provider_message_id="1",
                requested_target=target, actual_target=target, ordinal=True,
            )

    def test_receipts_are_ordered_component_evidence_not_a_final_id_alias(self):
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget(platform="telegram", chat_id="123")
        first = TransportReceipt(
            outcome="delivered", provider_message_id="1", requested_target=target,
            actual_target=target, component="text", ordinal=0,
        )
        second = TransportReceipt(
            outcome="delivered", provider_message_id="2", requested_target=target,
            actual_target=target, component="text", ordinal=1,
        )
        result = SendResult(success=True, message_id="2", receipts=(first, second))

        assert result.receipt == first
        assert result.receipts == (first, second)
        with pytest.raises(ValueError, match="ordered"):
            SendResult(success=True, receipts=(second, first))

    def test_send_result_rejects_receipt_subclasses_and_mutations_before_methods(self):
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget("telegram", "123")
        receipt = TransportReceipt(outcome="unknown", requested_target=target)

        class HostileTuple(tuple):
            def __iter__(self):
                raise AssertionError("hostile receipt tuple was iterated")

            def __len__(self):
                raise AssertionError("hostile receipt tuple length was evaluated")

        with pytest.raises(ValueError, match="immutable tuple"):
            SendResult(success=False, receipts=HostileTuple((receipt,)))

        class ReceiptSubclass(TransportReceipt):
            pass

        subclass_receipt = ReceiptSubclass(
            outcome="unknown", requested_target=target,
        )
        with pytest.raises(ValueError, match="TransportReceipt"):
            SendResult(success=False, receipts=(subclass_receipt,))

        class HostileText(str):
            def __hash__(self):
                raise AssertionError("hostile receipt outcome was hashed")

            def __eq__(self, _other):
                raise AssertionError("hostile receipt outcome was compared")

        object.__setattr__(receipt, "outcome", HostileText("unknown"))
        with pytest.raises(ValueError, match="outcome"):
            SendResult(success=False, receipts=(receipt,))

    @pytest.mark.parametrize("bad", [" id", "id ", "id\nnext", "id\u00a0next"])
    def test_transport_ids_reject_controls_and_whitespace_confusables(self, bad):
        from gateway.platforms.base import TransportTarget

        with pytest.raises(ValueError):
            TransportTarget(platform="telegram", chat_id=bad)

    def test_delivered_receipt_requires_provider_evidence_and_actual_target(self):
        from gateway.platforms.base import TransportReceipt, TransportTarget

        requested = TransportTarget(platform="matrix", chat_id="!requested:example.org")
        actual = TransportTarget(platform="matrix", chat_id="!actual:example.org")
        receipt = TransportReceipt(
            outcome="delivered",
            provider_message_id="$event",
            requested_target=requested,
            actual_target=actual,
        )

        result = SendResult(success=True, message_id="legacy-id", receipt=receipt)
        assert result.success is True  # legacy semantics stay independent
        assert result.message_id == "legacy-id"
        assert result.receipt == receipt

        with pytest.raises(ValueError, match="provider_message_id"):
            TransportReceipt(
                outcome="delivered",
                requested_target=requested,
                actual_target=actual,
            )
        with pytest.raises(ValueError, match="actual_target"):
            TransportReceipt(
                outcome="delivered",
                provider_message_id="$event",
                requested_target=requested,
            )

    def test_failed_receipt_requires_bounded_category_and_unknown_is_default(self):
        from gateway.platforms.base import TransportReceipt, TransportTarget

        target = TransportTarget(platform="telegram", chat_id="123", thread_id="7")
        assert SendResult(success=True).receipt is None
        with pytest.raises(ValueError, match="failure_kind"):
            TransportReceipt(outcome="failed", requested_target=target)
        failed = TransportReceipt(
            outcome="failed", requested_target=target, failure_kind="not_configured"
        )
        assert failed.failure_kind == "not_configured"


    def test_origin_with_source(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.is_origin is True


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"


class TestCaseSensitiveChatIdParsing:
    """Test that chat IDs preserve their original case (issue #11768)."""
    
    def test_slack_uppercase_chat_id_preserved(self):
        """Slack channel IDs like C123ABC should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"  # Should NOT be lowercased to c123abc
        assert target.is_explicit is True
    
    
    


class TestPlatformNameCaseInsensitivity:
    """Test that platform names are case-insensitive."""
    
    def test_uppercase_platform_name(self):
        """Platform names should be case-insensitive."""
        target = DeliveryTarget.parse("TELEGRAM:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
    

class _RelayDeliveryTransport:
    """Relay transport that advertises Slack and records outbound wire frames."""

    def __init__(self):
        self._identities = [("slack", "bot-1")]
        self.sent = []

    async def send_outbound(self, action, *, platform=None):
        self.sent.append((action, platform))
        if not action.get("metadata", {}).get("user_id"):
            return {"success": False, "error": "target not routed to an onboarded tenant"}
        return {"success": True, "message_id": "relay-message-1"}


def _make_relay(transport):
    return RelayAdapter(
        PlatformConfig(enabled=True),
        CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform="slack",
            label="Slack",
            max_message_length=4000,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=True,
            markdown_dialect="slack",
            len_unit="chars",
        ),
        transport=cast(Any, transport),
    )


@pytest.mark.asyncio
async def test_relay_fronted_target_delivers_without_prior_inbound_chat_state(tmp_path, monkeypatch):
    """A persisted Slack home must work immediately after a gateway restart."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    transport = _RelayDeliveryTransport()
    relay = _make_relay(transport)
    config = GatewayConfig(
        platforms={
            Platform.RELAY: PlatformConfig(enabled=True),
            Platform.SLACK: PlatformConfig(
                enabled=False,
                home_channel=HomeChannel(
                    platform=Platform.SLACK,
                    chat_id="D123",
                    name="Owner DM",
                    user_id="U123",
                ),
            ),
        },
    )
    router = DeliveryRouter(config, adapters={Platform.RELAY: relay})

    result = await router._deliver_to_platform(
        DeliveryTarget(platform=Platform.SLACK, chat_id="D123"),
        "scheduled result",
        metadata={"job_id": "cron-1", "user_id": "stale-user"},
    )

    assert getattr(result, "success", False) is True
    assert len(transport.sent) == 1
    action, wire_platform = transport.sent[0]
    assert wire_platform == "slack"
    assert action["chat_id"] == "D123"
    assert action["metadata"] == {"job_id": "cron-1", "user_id": "U123"}


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


@pytest.mark.asyncio
async def test_native_adapter_wins_when_relay_also_fronts_platform(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    native = RecordingAdapter()
    transport = _RelayDeliveryTransport()
    relay = _make_relay(transport)
    config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True),
            Platform.RELAY: PlatformConfig(enabled=True),
        },
    )
    router = DeliveryRouter(
        config,
        adapters={Platform.SLACK: native, Platform.RELAY: relay},
    )

    await router._deliver_to_platform(
        DeliveryTarget(platform=Platform.SLACK, chat_id="D123"),
        "native result",
        metadata=None,
    )

    assert native.calls == [
        {"chat_id": "D123", "content": "native result", "metadata": None}
    ]
    assert transport.sent == []


@pytest.mark.asyncio
async def test_disabled_native_adapter_does_not_shadow_relay(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    native = RecordingAdapter()
    transport = _RelayDeliveryTransport()
    relay = _make_relay(transport)
    config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=False,
                home_channel=HomeChannel(
                    platform=Platform.SLACK,
                    chat_id="D123",
                    name="Owner DM",
                    user_id="U123",
                ),
            ),
            Platform.RELAY: PlatformConfig(enabled=True),
        },
    )
    router = DeliveryRouter(
        config,
        adapters={Platform.SLACK: native, Platform.RELAY: relay},
    )

    await router._deliver_to_platform(
        DeliveryTarget(platform=Platform.SLACK, chat_id="D123"),
        "relay result",
        metadata=None,
    )

    assert native.calls == []
    assert len(transport.sent) == 1
    assert transport.sent[0][1] == "slack"


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


class FailingAdapter:
    async def send(self, chat_id, content, metadata=None):
        return SendResult(success=False, error="route failed", retryable=False)


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
