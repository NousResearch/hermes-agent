"""Tests for busy-queue admission outcomes and the pending-queue cap (#135679).

The pending FIFO caps at ``_BUSY_QUEUE_MAX_PENDING`` (32). Before the fix, a
follow-up the cap refused was dropped with only a WARNING log and no
user-facing receipt, while the busy-ack path still reported success — and the
explicit /queue and /steer queue fallbacks bypassed the cap entirely.

- ``_queue_or_replace_pending_event`` reports its outcome: ``accepted`` (new
  FIFO entry), ``merged`` (photo-burst head merge), ``full`` (cap refusal) or
  ``unavailable`` (no delivery adapter).
- On ``full`` the busy path sends ONE refusal reply (never a success ack)
  before the ACK debounce path can stamp a "queued" receipt, and the refused
  event stays unaccepted.
- The /queue and /steer turn-boundary fallbacks respect the same cap: the
  33rd entry is refused with a receipt instead of growing the FIFO.
- Commands remain separate FIFO entries (no merging of /queue or /steer).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, *, message_id: str = "m1") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id=message_id,
    )


def _voice_event(text: str, *, message_id: str = "v1") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=_make_source(),
        message_id=message_id,
        media_urls=[f"/tmp/{message_id}.ogg"],
        media_types=["audio/ogg"],
    )


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.send = AsyncMock()
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    return adapter


def _make_runner() -> tuple:
    """Bare GatewayRunner with a standalone (non-multiplex) primary adapter."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = _make_adapter()
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = None
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "queue"
    runner._queued_events = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._draining = False
    runner._restart_requested = False
    runner._busy_ack_ts = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    return runner, adapter


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _fill_fifo(runner, adapter=None, sk=None) -> None:
    """Fill the session's pending FIFO to the cap."""
    sk = sk or build_session_key(_make_source())
    for i in range(runner._BUSY_QUEUE_MAX_PENDING):
        outcome = runner._queue_or_replace_pending_event(sk, _make_event(f"m{i}", message_id=f"fill-{i}"))
        assert outcome == "accepted"


class TestAdmissionOutcome:
    """``_queue_or_replace_pending_event`` reports accepted/merged/full/unavailable."""

    def test_outcome_accepted_then_full_at_cap(self):
        runner, adapter = _make_runner()
        sk = build_session_key(_make_source())
        cap = runner._BUSY_QUEUE_MAX_PENDING
        for i in range(cap):
            outcome = runner._queue_or_replace_pending_event(sk, _make_event(f"m{i}", message_id=f"m-{i}"))
            assert outcome == "accepted", f"entry {i} should be accepted"

        # The 33rd is refused — and stays unaccepted.
        refused = _make_event("overflow", message_id="m-overflow")
        assert runner._queue_or_replace_pending_event(sk, refused) == "full"
        assert refused._gateway_accepted is not True
        assert runner._queue_depth(sk, adapter=adapter) == cap

    def test_outcome_unavailable_without_delivery_adapter(self):
        runner, adapter = _make_runner()
        runner.adapters = {}  # nothing resolves for the platform
        sk = build_session_key(_make_source())
        event = _make_event("nowhere")
        assert runner._queue_or_replace_pending_event(sk, event) == "unavailable"
        assert event._gateway_accepted is not True

    def test_photo_burst_merge_reports_merged_and_keeps_album_semantics(self):
        runner, adapter = _make_runner()
        sk = build_session_key(_make_source())
        photo = MessageEvent(
            text="first",
            message_type=MessageType.PHOTO,
            source=_make_source(),
            message_id="p1",
            media_urls=["/tmp/a.jpg"],
            media_types=["image/jpeg"],
        )
        assert runner._queue_or_replace_pending_event(sk, photo) == "accepted"
        caption = _make_event("second", message_id="p2")
        assert runner._queue_or_replace_pending_event(sk, caption) == "merged"

        head = adapter._pending_messages[sk]
        assert head.media_urls == ["/tmp/a.jpg"]
        assert "first" in head.text and "second" in head.text
        assert runner._queue_depth(sk, adapter=adapter) == 1


class TestBusyPathRefusalReceipt:
    """A cap-refused follow-up gets a refusal reply, never a success ack."""

    @pytest.mark.asyncio
    async def test_refused_media_event_gets_refusal_receipt_and_stays_unaccepted(self, monkeypatch):
        """A media follow-up the cap refuses gets a refusal reply, never a success ack.

        Plain queue-mode text returns to the adapter's debounce path, but media follow-ups
        admit through the runner FIFO right here — that's the dropped-at-cap lane.
        """
        import gateway.run as _gr

        monkeypatch.setattr(_gr, "_load_gateway_config", dict)
        runner, adapter = _make_runner()
        sk = build_session_key(_make_source())
        agent = MagicMock()
        agent._active_children = []
        runner._running_agents[sk] = agent
        runner._pending_event_audio_paths = lambda _e: []
        runner._transcribe_and_echo_pending_voice = AsyncMock(return_value=("", []))

        cap = runner._BUSY_QUEUE_MAX_PENDING
        for i in range(cap):
            event = _voice_event(f"voice {i}", message_id=f"m-{i}")
            assert await runner._handle_active_session_busy_message(event, sk) is True
            assert event._gateway_accepted is True
        adapter._send_with_retry.reset_mock()

        refused = _voice_event("one too many", message_id="m-overflow")
        assert await runner._handle_active_session_busy_message(refused, sk) is True
        assert refused._gateway_accepted is not True
        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "queue is full" in content
        assert "Queued for the next turn" not in content
        assert runner._queue_depth(sk, adapter=adapter) == cap

    @pytest.mark.asyncio
    async def test_refused_event_receipt_sent_even_inside_ack_debounce_window(self, monkeypatch):
        """The refusal must go out even when a recent ack would debounce a normal ack away."""
        import gateway.run as _gr
        import time as _time

        monkeypatch.setattr(_gr, "_load_gateway_config", dict)
        runner, adapter = _make_runner()
        sk = build_session_key(_make_source())
        agent = MagicMock()
        agent._active_children = []
        runner._running_agents[sk] = agent
        runner._pending_event_audio_paths = lambda _e: []
        runner._transcribe_and_echo_pending_voice = AsyncMock(return_value=("", []))

        from gateway.run import SessionState

        runner._sessions = {sk: SessionState()}
        # A RECENT ack: any normal busy-ack would be debounced away.
        runner._sessions[sk].turn.busy_ack_ts = _time.time()
        runner._peek_session_state = lambda _sk: runner._sessions.get(_sk)

        _fill_fifo(runner, adapter, sk)
        adapter._send_with_retry.reset_mock()

        refused = _voice_event("over cap", message_id="m-over")
        assert await runner._handle_active_session_busy_message(refused, sk) is True
        assert refused._gateway_accepted is not True
        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "queue is full" in content


class TestCommandFallbackCap:
    """The /queue and /steer turn-boundary fallbacks cannot bypass the cap."""

    def _runner_for_handle_message(self):
        """Runner shaped like tests/gateway/test_queue_command.py fixtures."""
        runner, adapter = _make_runner()
        runner._voice_mode = {}
        runner.session_store = MagicMock()
        runner.session_store.get_or_create_session.return_value = _session_entry()
        runner.session_store.load_transcript.return_value = []
        runner.session_store.has_any_sessions.return_value = True
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._session_db = MagicMock()
        runner._session_db.get_session_title.return_value = None
        runner._reasoning_config = None
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._show_reasoning = False
        runner._set_session_env = lambda _context: None
        runner._should_send_voice_reply = lambda *_a, **_k: False
        runner._send_voice_reply = AsyncMock()
        runner._capture_gateway_honcho_if_configured = lambda *a, **k: None
        runner._emit_gateway_run_progress = AsyncMock()
        return runner, adapter

    @pytest.mark.asyncio
    async def test_queue_command_refused_at_cap(self):
        from gateway.run import _AGENT_PENDING_SENTINEL

        runner, adapter = self._runner_for_handle_message()
        sk = build_session_key(_make_source())
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL
        _fill_fifo(runner, adapter, sk)

        event = _make_event("/queue 33rd", message_id="q-33")
        result = await runner._handle_message(event)

        # Refused: a refusal receipt (not a success receipt), FIFO unchanged at cap.
        assert result is not None
        assert "queue is full" in result.lower()
        assert "Queued for the next turn" not in result
        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING

    @pytest.mark.asyncio
    async def test_steer_fallback_refused_at_cap(self):
        from gateway.run import _AGENT_PENDING_SENTINEL

        runner, adapter = self._runner_for_handle_message()
        sk = build_session_key(_make_source())
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL
        _fill_fifo(runner, adapter, sk)

        event = _make_event("/steer wait up", message_id="s-33")
        result = await runner._handle_message(event)

        assert result is not None
        assert "queue is full" in result.lower()
        assert "Queued for the next turn" not in result
        assert runner._queue_depth(sk, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING

    @pytest.mark.asyncio
    async def test_queue_and_steer_entries_stay_separate_fifo_items(self):
        """Commands keep one FIFO entry each — the cap must not merge or split them."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        runner, adapter = self._runner_for_handle_message()
        sk = build_session_key(_make_source())
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

        await runner._handle_message(_make_event("/queue first", message_id="q-1"))
        await runner._handle_message(MessageEvent(
            text="/steer second", message_type=MessageType.TEXT,
            source=_make_source(), message_id="s-1",
        ))

        head = adapter._pending_messages[sk]
        assert head.text == "first"
        assert [e.text for e in runner._queued_events[sk]] == ["second"]
        assert runner._queue_depth(sk, adapter=adapter) == 2
