"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded (pending→attempting) BEFORE the send await,
delivered/failed by SendResult afterward; slash commands, ephemeral
replies, and empty responses are never recorded; ledger failures never
block the send.
"""

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway import shutdown_flush
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")


def _event(text="hello agent"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK, chat_id="C1", chat_type="channel"
        ),
        message_id="msg-42",
    )


def _rows():
    with dl._connect() as conn:
        return conn.execute(
            """SELECT obligation_id, state, content, adapter_profile
               FROM delivery_obligations"""
        ).fetchall()


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


async def _run(adapter, event, response="final answer"):
    adapter._message_handler = AsyncMock(return_value=response)
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
    @pytest.mark.asyncio
    async def test_normal_turn_records_and_delivers(self):
        adapter = _Adapter()
        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "delivered"
        assert rows[0][2] == "final answer"

    @pytest.mark.asyncio
    async def test_durable_inbound_is_retained_until_outbound_transfer(self, tmp_path, monkeypatch):
        """The real adapter path acks only after creating the durable reply row."""
        adapter = _Adapter()
        event = _event()
        flush_dir = tmp_path / "pending_messages"
        flush_dir.mkdir()
        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
        assert shutdown_flush.record_durable_inbound_event(
            "agent:main:slack:channel:C1", event
        )

        await _run(adapter, event)

        assert adapter.sent == ["final answer"]
        assert _rows()[0][1] == "delivered"
        assert shutdown_flush.recover_durable_inbound_events() == []

    @pytest.mark.asyncio
    async def test_empty_terminal_keeps_durable_inbound_for_visible_retry(
        self, tmp_path, monkeypatch
    ):
        """No text means no durable terminal reply transfer to acknowledge."""
        adapter = _Adapter()
        event = _event()
        flush_dir = tmp_path / "pending_messages"
        flush_dir.mkdir()
        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
        assert shutdown_flush.record_durable_inbound_event(
            "agent:main:slack:channel:C1", event
        )

        await _run(adapter, event, response=None)

        assert adapter.sent == []
        assert [restored.text for restored in shutdown_flush.recover_durable_inbound_events()] == [
            "hello agent"
        ]

    @pytest.mark.asyncio
    async def test_tts_caption_marks_the_existing_outbound_transfer_delivered(
        self, tmp_path, monkeypatch
    ):
        """Voice-caption delivery is terminal even when it skips a text send."""
        adapter = _Adapter()
        adapter.platform = Platform.TELEGRAM
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        event = _event()
        event.message_type = MessageType.VOICE
        flush_dir = tmp_path / "pending_messages"
        flush_dir.mkdir()
        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
        audio = tmp_path / "answer.ogg"
        audio.write_bytes(b"voice")
        monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
        monkeypatch.setattr(
            "tools.tts_tool.text_to_speech_tool",
            lambda **_kwargs: json.dumps({"success": True, "file_path": str(audio)}),
        )
        adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="voice-1"))
        assert shutdown_flush.record_durable_inbound_event(
            "agent:main:slack:channel:C1", event
        )

        await _run(adapter, event, response="spoken answer")

        adapter.play_tts.assert_awaited_once()
        assert adapter.sent == []
        assert _rows()[0][1] == "delivered"
        assert shutdown_flush.recover_durable_inbound_events() == []

    @pytest.mark.asyncio
    async def test_durable_inbound_survives_crash_during_background_dispatch(
        self, tmp_path, monkeypatch
    ):
        """handle_message returning after task spawn is not an acknowledgement."""
        adapter = _Adapter()
        event = _event()
        flush_dir = tmp_path / "pending_messages"
        flush_dir.mkdir()
        monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: flush_dir)
        assert shutdown_flush.record_durable_inbound_event(
            "agent:main:slack:channel:C1", event
        )

        started = asyncio.Event()
        never = asyncio.Event()

        async def blocked_handler(_event):
            started.set()
            await never.wait()

        adapter._message_handler = blocked_handler
        await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert [restored.text for restored in shutdown_flush.recover_durable_inbound_events()] == [
            "hello agent"
        ]

        for task in list(adapter._session_tasks.values()):
            task.cancel()
        await asyncio.gather(*list(adapter._session_tasks.values()), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_send_failure_leaves_failed_row(self):
        adapter = _Adapter()
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="chat_not_found")
        )
        await _run(adapter, _event())

        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "failed"

    @pytest.mark.asyncio
    async def test_late_transient_failure_signals_reconnected_runner(self):
        """A replacement installed mid-send must trigger another ledger sweep."""
        adapter = _Adapter()
        adapter._owner_profile = "reviewer"
        replacement = _Adapter()
        replacement._owner_profile = "reviewer"
        runner = MagicMock()
        runner._adapter_for_source.side_effect = [adapter, replacement]
        runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)
        adapter.gateway_runner = runner
        adapter.send = AsyncMock(
            return_value=SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            )
        )

        await _run(adapter, _event())

        assert _rows()[0][1] == "failed"
        assert _rows()[0][3] == "reviewer"
        runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(
            Platform.SLACK, profile="reviewer"
        )


    @pytest.mark.asyncio
    async def test_slow_ledger_record_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_record, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch(
            "gateway.delivery_ledger.record_obligation",
            side_effect=slow_record,
        ), patch("gateway.delivery_ledger.mark_attempting"):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_slow_ledger_update_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_delivered, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch("gateway.delivery_ledger.record_obligation"), patch(
            "gateway.delivery_ledger.mark_attempting"
        ), patch(
            "gateway.delivery_ledger.mark_delivered",
            side_effect=slow_delivered,
        ):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_crash_between_attempting_and_ack_is_recoverable(self):
        """The core scenario (#58818): process dies mid-send. The row must
        be claimable by a later process and carry the ambiguity marker."""
        adapter = _Adapter()

        async def _dies_mid_send(chat_id, content, reply_to=None, metadata=None):
            raise ConnectionError("gateway killed mid-await")

        adapter.send = _dies_mid_send
        # _send_with_retry raising propagates; the background task catches
        # broadly — drive only through the send block by tolerating the error.
        try:
            await _run(adapter, _event())
        except Exception:
            pass

        rows = _rows()
        assert len(rows) == 1
        # Row is stuck in 'attempting' (or failed if retry wrapper caught it):
        # either way it is non-delivered and recoverable.
        assert rows[0][1] in ("attempting", "failed")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True
