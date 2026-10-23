"""Signal native quotes through the registered plugin and gateway boundaries."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from gateway.config import Platform, PlatformConfig
from tests.gateway.test_signal import _make_signal_adapter, _stub_rpc, _reset_signal_scheduler

@pytest.fixture(autouse=True)
def _discover_signal_plugin():
    from hermes_cli.plugins import discover_plugins
    discover_plugins()

class TestSignalOutboundQuotes:
    @pytest.mark.asyncio
    async def test_send_dm_quotes_actual_author(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(
            return_value="2d31b40c-5412-4a2f-830a-477f0180cb11"
        )
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="+15551230000",
            content="hello",
            reply_to="1712345678000",
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
            },
        )

        assert result.success is True
        assert captured[0]["params"]["recipient"] == [
            "2d31b40c-5412-4a2f-830a-477f0180cb11"
        ]
        assert captured[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured[0]["params"]["quoteAuthor"] == (
            "68680952-6d86-45bc-85e0-1a4d186d53ee"
        )

    @pytest.mark.asyncio
    async def test_send_group_keeps_author_separate_from_destination(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="group:NVMOlw+k0ONcmektl8eNTUQ/2gYG84IW8qbVvY8086c=",
            content="hello group",
            reply_to="1712345678000",
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
            },
        )

        assert result.success is True
        params = captured[0]["params"]
        assert params["groupId"] == "NVMOlw+k0ONcmektl8eNTUQ/2gYG84IW8qbVvY8086c="
        assert params["quoteTimestamp"] == 1712345678000
        assert params["quoteAuthor"] == "68680952-6d86-45bc-85e0-1a4d186d53ee"
        assert "recipient" not in params
        assert params["quoteAuthor"] != "group:NVMOlw+k0ONcmektl8eNTUQ/2gYG84IW8qbVvY8086c="

    @pytest.mark.asyncio
    async def test_missing_author_delivers_without_partial_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="+15551230000",
            content="still deliver",
            reply_to="1712345678000",
            metadata={},
        )

        assert result.success is True
        assert "quoteTimestamp" not in captured[0]["params"]
        assert "quoteAuthor" not in captured[0]["params"]

    @pytest.mark.asyncio
    async def test_long_reply_quotes_only_first_chunk(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="+15551230000",
            content="x" * (adapter.MAX_MESSAGE_LENGTH + 500),
            reply_to="1712345678000",
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
            },
        )

        assert result.success is True
        assert len(captured) == 2
        assert captured[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured[0]["params"]["quoteAuthor"] == (
            "68680952-6d86-45bc-85e0-1a4d186d53ee"
        )
        assert "quoteTimestamp" not in captured[1]["params"]
        assert "quoteAuthor" not in captured[1]["params"]

    @pytest.mark.asyncio
    async def test_send_image_quotes_from_metadata(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc
        image_path = tmp_path / "reply.png"
        image_path.write_bytes(b"\x89PNG" + b"\x00" * 32)

        result = await adapter.send_image(
            chat_id="+15551230000",
            image_url=f"file://{image_path}",
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
            },
        )

        assert result.success is True
        assert captured[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured[0]["params"]["quoteAuthor"] == (
            "68680952-6d86-45bc-85e0-1a4d186d53ee"
        )

    @pytest.mark.asyncio
    async def test_shared_attachment_sender_quotes_from_metadata(
        self,
        monkeypatch,
        tmp_path,
    ):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc
        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 32)

        result = await adapter.send_voice(
            chat_id="+15551230000",
            audio_path=str(audio_path),
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "+15551230001",
            },
        )

        assert result.success is True
        assert captured[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured[0]["params"]["quoteAuthor"] == "+15551230001"

    @pytest.mark.asyncio
    async def test_image_batch_quotes_from_metadata(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            return {"timestamp": 1712345678999}

        adapter._rpc = mock_rpc
        image_path = tmp_path / "batch.png"
        image_path.write_bytes(b"\x89PNG" + b"\x00" * 32)

        await adapter.send_multiple_images(
            chat_id="+15551230000",
            images=[(f"file://{image_path}", "")],
            metadata={
                "signal_quote_timestamp": "1712345678000",
                "signal_quote_author": "+15551230001",
            },
        )

        assert captured[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured[0]["params"]["quoteAuthor"] == "+15551230001"


class TestSignalAutomaticQuoteRouting:
    def test_base_metadata_prefers_aci_and_falls_back_to_phone(self):
        from gateway.platforms.base import _thread_metadata_for_source
        from gateway.session import SessionSource

        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:abc",
            chat_type="group",
            user_id="+15551230000",
            message_id="1712345678000",
            user_id_alt="68680952-6d86-45bc-85e0-1a4d186d53ee",
        )
        assert _thread_metadata_for_source(source, "1712345678000") == {
            "signal_quote_timestamp": "1712345678000",
            "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
        }

        source.user_id_alt = None
        assert _thread_metadata_for_source(source, "1712345678000") == {
            "signal_quote_timestamp": "1712345678000",
            "signal_quote_author": "+15551230000",
        }
        assert _thread_metadata_for_source(source, None) is None

    def test_gateway_runner_metadata_carries_signal_author(self):
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = GatewayRunner.__new__(GatewayRunner)
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551230000",
            user_id="+15551230000",
            message_id="1712345678000",
            user_id_alt="68680952-6d86-45bc-85e0-1a4d186d53ee",
        )

        assert runner._thread_metadata_for_source(source, "1712345678000") == {
            "signal_quote_timestamp": "1712345678000",
            "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
        }

    @pytest.mark.asyncio
    async def test_group_envelope_routes_data_timestamp_and_member_author(
        self,
        monkeypatch,
    ):
        from gateway.platforms.base import (
            _reply_anchor_for_event,
            _thread_metadata_for_source,
        )

        adapter = _make_signal_adapter(monkeypatch, group_allowed="*")
        captured_event = {}

        async def fake_handle(event):
            captured_event["event"] = event

        adapter.handle_message = fake_handle
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15551230000",
                "sourceUuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                "sourceName": "Group member",
                "timestamp": 1712345677000,
                "dataMessage": {
                    "timestamp": 1712345678000,
                    "message": "question",
                    "groupInfo": {
                        "groupId": "NVMOlw+k0ONcmektl8eNTUQ/2gYG84IW8qbVvY8086c=",
                        "groupName": "Test group",
                    },
                },
            },
        })

        event = captured_event["event"]
        anchor = _reply_anchor_for_event(event)
        metadata = _thread_metadata_for_source(event.source, anchor)
        assert event.message_id == "1712345678000"
        assert anchor == "1712345678000"
        assert metadata == {
            "signal_quote_timestamp": "1712345678000",
            "signal_quote_author": "68680952-6d86-45bc-85e0-1a4d186d53ee",
        }

        adapter._stop_typing_indicator = AsyncMock()
        mock_rpc, captured_rpc = _stub_rpc({"timestamp": 1712345678999})
        adapter._rpc = mock_rpc
        result = await adapter.send(
            chat_id=event.source.chat_id,
            content="answer",
            reply_to=anchor,
            metadata=metadata,
        )

        assert result.success is True
        params = captured_rpc[0]["params"]
        assert params["groupId"] == "NVMOlw+k0ONcmektl8eNTUQ/2gYG84IW8qbVvY8086c="
        assert params["quoteTimestamp"] == 1712345678000
        assert params["quoteAuthor"] == "68680952-6d86-45bc-85e0-1a4d186d53ee"

    @pytest.mark.asyncio
    async def test_envelope_timestamp_is_compatibility_fallback(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15551230000",
                "timestamp": 1712345678000,
                "dataMessage": {"message": "legacy shape"},
            },
        })

        assert captured["event"].message_id == "1712345678000"

    @pytest.mark.asyncio
    async def test_edited_message_quotes_original_target_timestamp(
        self, monkeypatch
    ):
        from gateway.platforms.base import (
            _reply_anchor_for_event,
            _thread_metadata_for_source,
        )

        adapter = _make_signal_adapter(monkeypatch)
        captured_event = {}

        async def fake_handle(event):
            captured_event["event"] = event

        adapter.handle_message = fake_handle
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15551230000",
                "sourceUuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                "timestamp": 1712345680000,
                "editMessage": {
                    "targetSentTimestamp": 1712345678000,
                    "dataMessage": {
                        "timestamp": 1712345680000,
                        "message": "edited question",
                    },
                },
            },
        })

        event = captured_event["event"]
        anchor = _reply_anchor_for_event(event)
        metadata = _thread_metadata_for_source(event.source, anchor)
        # The edit is a distinct transport event for dedupe/persistence, while
        # native replies and reactions still address the replaced message.
        assert event.message_id == "1712345680000"
        assert event.raw_message["timestamp_ms"] == 1712345678000
        assert int(event.timestamp.timestamp() * 1000) == 1712345680000

        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value="+15551230000")
        mock_rpc, captured_rpc = _stub_rpc({"timestamp": 1712345681000})
        adapter._rpc = mock_rpc
        result = await adapter.send(
            chat_id=event.source.chat_id,
            content="answer",
            reply_to=anchor,
            metadata=metadata,
        )

        assert result.success is True
        assert captured_rpc[0]["params"]["quoteTimestamp"] == 1712345678000
        assert captured_rpc[0]["params"]["quoteAuthor"] == (
            "68680952-6d86-45bc-85e0-1a4d186d53ee"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("second_author", ["+15551230000", "+15551230001"])
    async def test_debounce_keeps_native_anchor_with_its_author(self, monkeypatch, second_author):
        from dataclasses import replace
        from gateway.platforms.reply_context import _thread_metadata_for_event
        from gateway.platforms.base import _reply_anchor_for_event
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource

        adapter = _make_signal_adapter(monkeypatch)
        adapter._busy_text_debounce_seconds = 60.0
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551230000",
            chat_type="dm",
            user_id="+15551230000",
            message_id="1712345678000",
        )
        first = MessageEvent(
            text="first edit",
            source=source,
            message_id="1712345680000",
            reply_anchor_override="1712345678000",
            raw_message={"timestamp_ms": 1712345678000},
        )
        second = MessageEvent(
            text="second edit",
            source=replace(source, user_id=second_author, message_id="1712345679000"),
            message_id="1712345690000",
            reply_anchor_override="1712345679000",
            raw_message={"timestamp_ms": 1712345679000},
        )

        await adapter._queue_text_debounce("signal:dm", first)
        await adapter._queue_text_debounce("signal:dm", second)

        merged = adapter._text_debounce["signal:dm"].event
        assert merged.message_id == "1712345690000"
        assert _reply_anchor_for_event(merged) == "1712345679000"
        metadata = _thread_metadata_for_event(merged)
        assert metadata["signal_quote_timestamp"] == "1712345679000"
        assert metadata["signal_quote_author"] == second_author
        if second_author != source.user_id:
            # The existing sender boundary flushes the first author into its own turn.
            pending_metadata = _thread_metadata_for_event(adapter._pending_messages["signal:dm"])
            assert pending_metadata["signal_quote_timestamp"] == "1712345678000"
            assert pending_metadata["signal_quote_author"] == source.user_id
        await adapter._flush_text_debounce_now("signal:dm")

    @pytest.mark.asyncio
    async def test_separate_busy_bursts_keep_latest_signal_reply_anchor(
        self, monkeypatch
    ):
        from dataclasses import replace
        from gateway.platforms.base import _reply_anchor_for_event
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource

        adapter = _make_signal_adapter(monkeypatch)
        adapter._busy_text_debounce_seconds = 60.0
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551230000",
            chat_type="dm",
            user_id="+15551230000",
            message_id="1712345678000",
        )
        first = MessageEvent(
            text="burst one",
            source=source,
            message_id="1712345680000",
            reply_anchor_override="1712345678000",
            raw_message={"timestamp_ms": 1712345678000},
        )
        second = MessageEvent(
            text="burst two",
            source=replace(source, message_id="1712345679000"),
            message_id="1712345690000",
            reply_anchor_override="1712345679000",
            raw_message={"timestamp_ms": 1712345679000},
        )

        await adapter._queue_text_debounce("signal:dm", first)
        await adapter._flush_text_debounce_now("signal:dm")
        await adapter._queue_text_debounce("signal:dm", second)
        await adapter._flush_text_debounce_now("signal:dm")

        pending = adapter._pending_messages["signal:dm"]
        assert pending.text == "burst one\nburst two"
        assert pending.message_id == "1712345690000"
        assert _reply_anchor_for_event(pending) == "1712345679000"


@pytest.mark.asyncio
async def test_edited_quote_survives_native_admission_and_rejects_stale_redirect(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.platforms.base import _reply_anchor_for_event
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv('SIGNAL_ALLOWED_USERS', '+15551230000')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = _make_signal_adapter(monkeypatch)
    adapter.gateway_runner, runner.adapters = runner, {Platform.SIGNAL: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='signal-quotes')
    runner._wire_adapter_handlers(adapter)
    received = []

    async def capture(event):
        received.append(event)

    adapter.handle_message = capture
    try:
        await adapter._handle_envelope({'envelope': {
            'sourceNumber': '+15551230000',
            'sourceUuid': '68680952-6d86-45bc-85e0-1a4d186d53ee',
            'timestamp': 1712345680000,
            'editMessage': {'targetSentTimestamp': 1712345678000,
                            'dataMessage': {'timestamp': 1712345680000, 'message': 'edited question'}},
        }})
        receipt = await authority.admit_native(received[0])
        row, = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)
        replay = restore_native(row['payload'], runner)
        assert replay.raw_message is None
        assert replay.message_id == '1712345680000'
        anchor = _reply_anchor_for_event(replay)
        assert anchor == '1712345678000'
        metadata = runner._thread_metadata_for_source(replay.source, anchor)
        adapter._stop_typing_indicator = AsyncMock()
        adapter._resolve_recipient = AsyncMock(return_value='+15551230000')
        adapter._rpc, calls = _stub_rpc({'timestamp': 1712345681000})
        assert (await adapter.send(replay.source.chat_id, 'answer', reply_to=anchor, metadata=metadata)).success
        assert calls[-1]['params']['quoteTimestamp'] == 1712345678000
        assert calls[-1]['params']['quoteAuthor'] == '68680952-6d86-45bc-85e0-1a4d186d53ee'
        # A redirect can change the target after the turn captured metadata.
        assert (await adapter.send(replay.source.chat_id, 'redirected answer',
                                   reply_to='1712345690000', metadata=metadata)).success
        assert 'quoteTimestamp' not in calls[-1]['params']
        assert 'quoteAuthor' not in calls[-1]['params']
    finally:
        authority.db.close()
