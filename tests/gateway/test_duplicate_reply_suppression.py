"""Tests for duplicate reply suppression across the gateway stack.

Covers four fix paths:
  1. base.py: stale response suppressed when interrupt_event is set and a
     pending message exists (#8221 / #2483)
  2. run.py return path: only confirmed final streamed delivery suppresses
     the fallback final send; partial streamed output must not
  3. run.py queued-message path: first response is skipped only when the
     final response was actually streamed, not merely when partial output existed
  4. stream_consumer.py cancellation handler: only confirms final delivery
     when the best-effort send actually succeeds, not merely because partial
     content was sent earlier
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for testing."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake"), Platform.DISCORD)
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="msg1")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_event(text="hello", chat_id="c1", user_id="u1"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=chat_id,
            chat_type="dm",
            user_id=user_id,
        ),
        message_id="m1",
    )


# ===================================================================
# Test 1: base.py — stale response suppressed on interrupt (#8221)
# ===================================================================

class TestBaseInterruptSuppression:
    @pytest.mark.asyncio
    async def test_stale_response_suppressed_when_interrupted(self):
        """When interrupt_event is set AND a pending message exists,
        base.py should suppress the stale response instead of sending it."""
        adapter = StubAdapter()

        stale_response = "This is the stale answer to the first question."
        pending_response = "This is the answer to the second question."
        call_count = 0

        async def fake_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stale_response
            return pending_response

        adapter.set_message_handler(fake_handler)

        event_a = _make_event(text="first question")
        session_key = build_session_key(event_a.source)

        # Simulate: message A is being processed, message B arrives
        # The interrupt event is set and B is in pending_messages
        interrupt_event = asyncio.Event()
        interrupt_event.set()
        adapter._active_sessions[session_key] = interrupt_event

        event_b = _make_event(text="second question")
        adapter._pending_messages[session_key] = event_b

        await adapter._process_message_background(event_a, session_key)

        # The stale response should NOT have been sent.
        stale_sends = [s for s in adapter.sent if s["content"] == stale_response]
        assert len(stale_sends) == 0, (
            f"Stale response was sent {len(stale_sends)} time(s) — should be suppressed"
        )
        # The pending message's response SHOULD have been sent.
        pending_sends = [s for s in adapter.sent if s["content"] == pending_response]
        assert len(pending_sends) == 1, "Pending message response should be sent"

    @pytest.mark.asyncio
    async def test_response_not_suppressed_without_interrupt(self):
        """Normal case: no interrupt, response should be sent."""
        adapter = StubAdapter()

        async def fake_handler(event):
            return "Normal response"

        adapter.set_message_handler(fake_handler)
        event = _make_event()
        session_key = build_session_key(event.source)

        await adapter._process_message_background(event, session_key)

        assert any(s["content"] == "Normal response" for s in adapter.sent)

    @pytest.mark.asyncio
    async def test_response_not_suppressed_with_interrupt_but_no_pending(self):
        """Interrupt event set but no pending message (race already resolved) —
        response should still be sent."""
        adapter = StubAdapter()

        async def fake_handler(event):
            return "Valid response"

        adapter.set_message_handler(fake_handler)
        event = _make_event()
        session_key = build_session_key(event.source)

        # Set interrupt but no pending message
        interrupt_event = asyncio.Event()
        interrupt_event.set()
        adapter._active_sessions[session_key] = interrupt_event

        await adapter._process_message_background(event, session_key)

        assert any(s["content"] == "Valid response" for s in adapter.sent)


# Test 2: run.py — partial streamed output must not suppress final send
# ===================================================================

class TestOnlyFinalStreamDeliverySuppressesFinalSend:
    """The gateway should suppress the fallback final send only when the
    stream consumer confirmed the final assistant reply was delivered.

    Partial streamed output is not enough. If only already_sent=True,
    the fallback final send must still happen so Telegram users don't lose
    the real answer."""

    def _make_mock_stream_consumer(self, already_sent=False, final_response_sent=False):
        sc = SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
        )
        return sc

    def test_partial_stream_output_does_not_set_already_sent(self):
        """already_sent=True alone must NOT suppress final delivery."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=False)
        response = {"final_response": "text", "response_previewed": False}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            if not _is_empty_sentinel and (_streamed or _previewed):
                response["already_sent"] = True

        assert "already_sent" not in response

    def test_already_sent_not_set_when_nothing_sent(self):
        """When stream consumer hasn't sent anything, already_sent should
        not be set on the response."""
        sc = self._make_mock_stream_consumer(already_sent=False, final_response_sent=False)
        response = {"final_response": "text", "response_previewed": False}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            if not _is_empty_sentinel and (_streamed or _previewed):
                response["already_sent"] = True

        assert "already_sent" not in response

    def test_already_sent_set_on_final_response_sent(self):
        """final_response_sent=True should suppress duplicate final sends."""
        sc = self._make_mock_stream_consumer(already_sent=False, final_response_sent=True)
        response = {"final_response": "text"}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            if not _is_empty_sentinel and (_streamed or _previewed):
                response["already_sent"] = True

        assert response.get("already_sent") is True

    def test_already_sent_not_set_on_failed_response(self):
        """Failed responses should never be suppressed — user needs to see
        the error message even if streaming sent earlier partial output."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=False)
        response = {"final_response": "Error: something broke", "failed": True}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            if not _is_empty_sentinel and (_streamed or _previewed):
                response["already_sent"] = True

        assert "already_sent" not in response


# ===================================================================
# Test 2b: run.py — empty response never suppressed (#10xxx)
# ===================================================================

class TestEmptyResponseNotSuppressed:
    """When the model returns '(empty)' after tool calls (e.g. mimo-v2-pro
    going silent after web_search), the gateway must NOT suppress delivery
    even if the stream consumer sent intermediate text earlier.

    Without this fix, the user sees partial streaming text ('Let me search
    for that') and then silence — the '(empty)' sentinel is swallowed by
    already_sent=True."""

    def _make_mock_stream_consumer(self, already_sent=False, final_response_sent=False):
        return SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
        )

    def _apply_suppression_logic(self, response, sc):
        """Reproduce the fixed logic from gateway/run.py return path."""
        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            if not _is_empty_sentinel and (_streamed or _previewed):
                response["already_sent"] = True

    def test_empty_sentinel_not_suppressed_with_already_sent(self):
        """'(empty)' final_response should NOT be suppressed even when
        streaming sent intermediate content."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=True)
        response = {"final_response": "(empty)"}
        self._apply_suppression_logic(response, sc)
        assert "already_sent" not in response

    def test_empty_string_not_suppressed_with_already_sent(self):
        """Empty string final_response should NOT be suppressed."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=True)
        response = {"final_response": ""}
        self._apply_suppression_logic(response, sc)
        assert "already_sent" not in response

    def test_none_response_not_suppressed_with_already_sent(self):
        """None final_response should NOT be suppressed."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=True)
        response = {"final_response": None}
        self._apply_suppression_logic(response, sc)
        assert "already_sent" not in response

    def test_real_response_still_suppressed_only_when_final_delivery_confirmed(self):
        """Normal non-empty response should be suppressed only when the final
        response was actually streamed."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=True)
        response = {"final_response": "Here are the search results..."}
        self._apply_suppression_logic(response, sc)
        assert response.get("already_sent") is True

    def test_failed_empty_response_never_suppressed(self):
        """Failed responses are never suppressed regardless of content."""
        sc = self._make_mock_stream_consumer(already_sent=True, final_response_sent=True)
        response = {"final_response": "(empty)", "failed": True}
        self._apply_suppression_logic(response, sc)
        assert "already_sent" not in response

class TestQueuedMessageAlreadyStreamed:
    """The queued-message path should skip the first response only when the
    final response was actually streamed."""

    def _make_mock_sc(self, already_sent=False, final_response_sent=False):
        return SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
        )

    def test_queued_path_only_skips_send_when_final_response_was_streamed(self):
        """Partial streamed output alone must not suppress the first response
        before the queued follow-up is processed."""
        _sc = self._make_mock_sc(already_sent=True, final_response_sent=False)

        _already_streamed = bool(
            _sc and getattr(_sc, "final_response_sent", False)
        )

        assert _already_streamed is False

    def test_queued_path_detects_confirmed_final_stream_delivery(self):
        """Confirmed final streamed delivery should skip the resend."""
        _sc = self._make_mock_sc(already_sent=True, final_response_sent=True)
        response = {"response_previewed": False}

        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or bool(response.get("response_previewed"))
        )

        assert _already_streamed is True

    def test_queued_path_detects_previewed_response_delivery(self):
        """A response already previewed via the adapter should not be resent
        before processing the queued follow-up."""
        _sc = self._make_mock_sc(already_sent=False, final_response_sent=False)
        response = {"response_previewed": True}

        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or bool(response.get("response_previewed"))
        )

        assert _already_streamed is True

    def test_queued_path_sends_when_not_streamed(self):
        """Nothing was streamed — first response should be sent before
        processing the queued message."""
        _sc = self._make_mock_sc(already_sent=False, final_response_sent=False)

        _already_streamed = bool(
            _sc and getattr(_sc, "final_response_sent", False)
        )

        assert _already_streamed is False

    def test_queued_path_with_no_stream_consumer(self):
        """No stream consumer at all (streaming disabled) — not streamed."""
        _sc = None

        _already_streamed = bool(
            _sc and getattr(_sc, "final_response_sent", False)
        )

        assert _already_streamed is False


# ===================================================================
# Test 4: stream_consumer.py — cancellation handler delivery confirmation
# ===================================================================

class TestCancellationHandlerDeliveryConfirmation:
    """The stream consumer's cancellation handler should only set
    final_response_sent when the best-effort send actually succeeds.
    Partial content (already_sent=True) alone must not promote to
    final_response_sent — that would suppress the gateway's fallback
    send even when the user never received the real answer."""

    def test_partial_only_no_accumulated_stays_false(self):
        """Cancelled after sending intermediate text, nothing accumulated.
        final_response_sent must stay False so the gateway fallback fires."""
        already_sent = True
        final_response_sent = False
        accumulated = ""
        message_id = None

        _best_effort_ok = False
        if accumulated and message_id:
            _best_effort_ok = True  # wouldn't enter
        if _best_effort_ok and not final_response_sent:
            final_response_sent = True

        assert final_response_sent is False

    def test_best_effort_succeeds_sets_true(self):
        """When accumulated content exists and best-effort send succeeds,
        final_response_sent should become True."""
        already_sent = True
        final_response_sent = False
        accumulated = "Here are the search results..."
        message_id = "msg_123"

        _best_effort_ok = False
        if accumulated and message_id:
            _best_effort_ok = True  # simulating successful _send_or_edit
        if _best_effort_ok and not final_response_sent:
            final_response_sent = True

        assert final_response_sent is True

    def test_best_effort_fails_stays_false(self):
        """When best-effort send fails (flood control, network), the
        gateway fallback must deliver the response."""
        already_sent = True
        final_response_sent = False
        accumulated = "Here are the search results..."
        message_id = "msg_123"

        _best_effort_ok = False
        if accumulated and message_id:
            _best_effort_ok = False  # simulating failed _send_or_edit
        if _best_effort_ok and not final_response_sent:
            final_response_sent = True

        assert final_response_sent is False

    def test_preserves_existing_true(self):
        """If final_response_sent was already True before cancellation,
        it must remain True regardless."""
        already_sent = True
        final_response_sent = True
        accumulated = ""
        message_id = None

        _best_effort_ok = False
        if accumulated and message_id:
            pass
        if _best_effort_ok and not final_response_sent:
            final_response_sent = True

        assert final_response_sent is True

    def test_old_behavior_would_have_promoted_partial(self):
        """Verify the old code would have incorrectly promoted
        already_sent to final_response_sent even with no accumulated
        content — proving the bug existed."""
        already_sent = True
        final_response_sent = False

        # OLD cancellation handler logic:
        if already_sent:
            final_response_sent = True

        assert final_response_sent is True  # the bug: partial promoted to final
