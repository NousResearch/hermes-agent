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

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.run import _normalize_whitespace, _stream_visible_matches_final
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for testing."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake"), Platform.DISCORD)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):
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

        # The in-band pending-drain now hands off to a fresh task instead
        # of recursing (#17758).  Wait for that task to finish before
        # checking the sent list.
        for _ in range(200):
            if any(s["content"] == pending_response for s in adapter.sent):
                break
            await asyncio.sleep(0.01)
        await adapter.cancel_background_tasks()

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

    def _make_mock_stream_consumer(self, already_sent=False, final_response_sent=False, visible_prefix="", adapter_requires_finalize=False):
        sc = SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
            _message_id="msg1",
        )
        sc._visible_prefix = lambda: visible_prefix
        sc._clean_for_display = lambda t: t
        sc._adapter_requires_finalize = adapter_requires_finalize
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
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
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
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
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
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
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
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
                response["already_sent"] = True

        assert "already_sent" not in response

    def test_visible_covers_final_suppresses(self):
        """When stream consumer visible text already contains the full final
        answer (e.g. visible ends with cursor/residue), the gateway should
        suppress the duplicate."""
        sc = self._make_mock_stream_consumer(visible_prefix="Hello world \u2589")
        response = {"final_response": "Hello world", "response_previewed": False}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
                response["already_sent"] = True

        assert response.get("already_sent") is True

    def test_visible_prefix_of_final_does_not_suppress(self):
        """When visible text is only a prefix of the final answer (tail not
        yet delivered), the gateway must send the full response."""
        sc = self._make_mock_stream_consumer(visible_prefix="Hello")
        response = {"final_response": "Hello world, here is the answer", "response_previewed": False}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
                response["already_sent"] = True

        assert "already_sent" not in response

    def test_visible_covers_final_but_adapter_requires_finalize_does_not_suppress(self):
        """When the adapter requires explicit finalize, visible-text match
        alone must not suppress — the gateway must still deliver so the
        stream consumer can fire a finalize=True edit."""
        sc = self._make_mock_stream_consumer(
            visible_prefix="Hello world \u2589",
            adapter_requires_finalize=True,
        )
        response = {"final_response": "Hello world", "response_previewed": False}

        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
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
        sc = SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
        )
        sc._visible_prefix = lambda: ""
        sc._clean_for_display = lambda t: t
        sc._adapter_requires_finalize = False
        return sc

    def _apply_suppression_logic(self, response, sc):
        """Reproduce the fixed logic from gateway/run.py return path."""
        if sc and isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            _streamed = bool(sc and getattr(sc, "final_response_sent", False))
            _previewed = bool(response.get("response_previewed"))
            _requires_finalize = bool(sc and getattr(sc, "_adapter_requires_finalize", False))
            _visible_match = _stream_visible_matches_final(sc, _final) and not _requires_finalize
            _transformed = bool(response.get("response_transformed"))
            if not _is_empty_sentinel and not _transformed and (_streamed or _previewed or _visible_match):
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

    def _make_mock_sc(self, already_sent=False, final_response_sent=False, visible_prefix="", adapter_requires_finalize=False):
        sc = SimpleNamespace(
            already_sent=already_sent,
            final_response_sent=final_response_sent,
            _message_id="msg1",
        )
        sc._visible_prefix = lambda: visible_prefix
        sc._clean_for_display = lambda t: t
        sc._adapter_requires_finalize = adapter_requires_finalize
        return sc

    def test_queued_path_only_skips_send_when_final_response_was_streamed(self):
        """Partial streamed output alone must not suppress the first response
        before the queued follow-up is processed."""
        _sc = self._make_mock_sc(already_sent=True, final_response_sent=False)
        result = {"final_response": "hello", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is False

    def test_queued_path_detects_confirmed_final_stream_delivery(self):
        """Confirmed final streamed delivery should skip the resend."""
        _sc = self._make_mock_sc(already_sent=True, final_response_sent=True)
        result = {"final_response": "hello", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is True

    def test_queued_path_detects_previewed_response_delivery(self):
        """A response already previewed via the adapter should not be resent
        before processing the queued follow-up."""
        _sc = self._make_mock_sc(already_sent=False, final_response_sent=False)
        result = {"final_response": "hello", "response_previewed": True}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is True

    def test_queued_path_previewed_response_does_not_trigger_media_redelivery(self):
        """Previewed responses already used the adapter delivery path, so the
        queued MEDIA backfill must not run for pure preview suppression."""
        _sc = self._make_mock_sc(already_sent=False, final_response_sent=False)
        result = {"final_response": "hello MEDIA:/tmp/img.png", "response_previewed": True}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _stream_confirmed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or (_sc and getattr(_sc, "final_content_delivered", False))
        )
        _visible_confirmed = bool(
            _stream_visible_matches_final(_sc, result.get("final_response", ""))
            and not _requires_finalize
        )
        _already_streamed = bool(_stream_confirmed or _visible_confirmed or _previewed)
        should_redeliver_media = _stream_confirmed or _visible_confirmed

        assert _already_streamed is True
        assert should_redeliver_media is False

    def test_queued_path_sends_when_not_streamed(self):
        """Nothing was streamed — first response should be sent before
        processing the queued message."""
        _sc = self._make_mock_sc(already_sent=False, final_response_sent=False)
        result = {"final_response": "hello", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is False

    def test_queued_path_with_no_stream_consumer(self):
        """No stream consumer at all (streaming disabled) — not streamed."""
        _sc = None
        result = {"final_response": "hello", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is False

    def test_queued_path_visible_covers_final_skips_resend(self):
        """When visible text already contains the full final answer, the
        queued-message path should skip resending."""
        _sc = self._make_mock_sc(visible_prefix="Hello world \u2589")
        result = {"final_response": "Hello world", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is True

    def test_queued_path_visible_confirmed_triggers_media_redelivery(self):
        """When visible text suppresses the first response, MEDIA tags still
        need the queued-path backfill because text streaming handled only text."""
        _sc = self._make_mock_sc(visible_prefix="Hello world")
        result = {"final_response": "Hello world", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _stream_confirmed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or (_sc and getattr(_sc, "final_content_delivered", False))
        )
        _visible_confirmed = bool(
            _stream_visible_matches_final(_sc, result.get("final_response", ""))
            and not _requires_finalize
        )
        _already_streamed = bool(_stream_confirmed or _visible_confirmed or _previewed)
        should_redeliver_media = _stream_confirmed or _visible_confirmed

        assert _already_streamed is True
        assert _visible_confirmed is True
        assert should_redeliver_media is True

    def test_queued_path_visible_prefix_of_final_still_sends(self):
        """When visible text is only a prefix, the queued-message path must
        still send the first response."""
        _sc = self._make_mock_sc(visible_prefix="Hello")
        result = {"final_response": "Hello world, here is the answer", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
        )

        assert _already_streamed is False

    def test_queued_path_visible_covers_final_but_requires_finalize_still_sends(self):
        """When the adapter requires explicit finalize, visible-text match
        alone must not skip the queued-message first-response delivery."""
        _sc = self._make_mock_sc(
            visible_prefix="Hello world \u2589",
            adapter_requires_finalize=True,
        )
        result = {"final_response": "Hello world", "response_previewed": False}

        _previewed = bool(result.get("response_previewed"))
        _requires_finalize = bool(_sc and getattr(_sc, "_adapter_requires_finalize", False))
        _already_streamed = bool(
            (_sc and getattr(_sc, "final_response_sent", False))
            or _previewed
            or (_sc and getattr(_sc, "final_content_delivered", False))
            or (
                _stream_visible_matches_final(_sc, result.get("final_response", ""))
                and not _requires_finalize
            )
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


class TestFinalContentDeliveredSuppression:
    """When stream consumer delivered the final content but the cosmetic
    final edit (cursor removal) failed, the gateway must suppress the
    fallback send to prevent duplicate messages.

    Covers the scenario not handled by final_response_sent alone:
    content reached the user via _send_or_edit, but the subsequent edit
    that clears a typing cursor or streaming marker failed, leaving
    final_response_sent=False even though the user already saw the text.
    """

    def test_content_delivered_but_final_edit_failed_suppresses(self):
        """final_content_delivered=True + final_response_sent=False
        must suppress (content already visible to user)."""
        sc = SimpleNamespace(
            already_sent=True,
            final_response_sent=False,
            final_content_delivered=True,
        )
        sc._visible_prefix = lambda: ""
        sc._clean_for_display = lambda t: t
        sc._adapter_requires_finalize = False
        response = {"final_response": "Hello!", "response_previewed": False}

        _streamed = bool(getattr(sc, "final_response_sent", False))
        _previewed = bool(response.get("response_previewed"))
        _content_delivered = bool(getattr(sc, "final_content_delivered", False))
        _requires_finalize = bool(getattr(sc, "_adapter_requires_finalize", False))
        _visible_match = _stream_visible_matches_final(sc, response.get("final_response", "")) and not _requires_finalize
        _is_empty_sentinel = (
            not response.get("final_response")
            or response.get("final_response") == "(empty)"
        )
        _transformed = bool(response.get("response_transformed"))
        if not _is_empty_sentinel and not _transformed and (
            _streamed or _previewed or _content_delivered or _visible_match
        ):
            response["already_sent"] = True

        assert response.get("already_sent") is True

    def test_intermediate_text_only_does_not_suppress(self):
        """already_sent=True from intermediate text + final_content_delivered=False
        must NOT suppress (user still needs the real final answer)."""
        sc = SimpleNamespace(
            already_sent=True,
            final_response_sent=False,
            final_content_delivered=False,
        )
        sc._visible_prefix = lambda: ""
        sc._clean_for_display = lambda t: t
        sc._adapter_requires_finalize = False
        response = {"final_response": "Real answer", "response_previewed": False}

        _streamed = bool(getattr(sc, "final_response_sent", False))
        _previewed = bool(response.get("response_previewed"))
        _content_delivered = bool(getattr(sc, "final_content_delivered", False))
        _requires_finalize = bool(getattr(sc, "_adapter_requires_finalize", False))
        _visible_match = _stream_visible_matches_final(sc, response.get("final_response", "")) and not _requires_finalize
        _is_empty_sentinel = (
            not response.get("final_response")
            or response.get("final_response") == "(empty)"
        )
        _transformed = bool(response.get("response_transformed"))
        if not _is_empty_sentinel and not _transformed and (
            _streamed or _previewed or _content_delivered or _visible_match
        ):
            response["already_sent"] = True

        assert "already_sent" not in response


# ===================================================================
# Test 6: run.py — _stream_visible_matches_final helper
# ===================================================================


class TestNormalizeWhitespace:
    """Unit tests for _normalize_whitespace — the normalization step used by
    _stream_visible_matches_final to compare visible and final text."""

    def test_collapse_multiple_spaces(self):
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_collapse_newlines(self):
        assert _normalize_whitespace("line1\n\nline2\n") == "line1 line2"

    def test_collapse_tabs(self):
        assert _normalize_whitespace("\thello\t\tworld") == "hello world"

    def test_strip_edges(self):
        assert _normalize_whitespace("  foo  ") == "foo"

    def test_mixed_whitespace(self):
        assert _normalize_whitespace("a  \n\t\n  b") == "a b"

    def test_no_whitespace(self):
        assert _normalize_whitespace("hello") == "hello"

    def test_only_whitespace(self):
        assert _normalize_whitespace("   \n\t  ") == ""

    def test_empty_string(self):
        assert _normalize_whitespace("") == ""

    def test_none_input(self):
        assert _normalize_whitespace(None) == ""


class TestStreamVisibleMatchesFinal:
    """Unit tests for _stream_visible_matches_final — the tolerant comparison
    used when stream consumer flags are lost but visible text already shows
    the final answer."""

    @staticmethod
    def _make_sc(visible_prefix="", clean_fn=None):
        sc = SimpleNamespace()
        sc._message_id = "msg1"
        sc._visible_prefix = lambda: visible_prefix
        if clean_fn is None:
            clean_fn = lambda t: t
        sc._clean_for_display = clean_fn
        return sc

    def test_returns_false_when_sc_is_none(self):
        assert _stream_visible_matches_final(None, "hello") is False

    def test_returns_false_when_final_is_empty(self):
        sc = self._make_sc(visible_prefix="hello")
        assert _stream_visible_matches_final(sc, "") is False

    def test_returns_false_when_visible_is_empty(self):
        sc = self._make_sc(visible_prefix="")
        assert _stream_visible_matches_final(sc, "hello") is False

    def test_exact_match_normalised(self):
        sc = self._make_sc(visible_prefix="hello world")
        assert _stream_visible_matches_final(sc, "hello  world") is True

    def test_visible_starts_with_final(self):
        # visible shows full answer plus trailing cursor/residue
        sc = self._make_sc(visible_prefix="hello world \u2589")
        assert _stream_visible_matches_final(sc, "hello world") is True

    def test_final_starts_with_visible_not_matched(self):
        # visible is only a prefix of final — must NOT suppress
        # (the tail still needs to be delivered)
        sc = self._make_sc(visible_prefix="hello wor")
        assert _stream_visible_matches_final(sc, "hello world") is False

    def test_final_starts_with_visible_low_coverage(self):
        # visible shows too little — should NOT match
        sc = self._make_sc(visible_prefix="he")
        assert _stream_visible_matches_final(sc, "hello world and more text") is False

    def test_clean_for_display_applied_to_final(self):
        clean_fn = lambda t: t.replace("MEDIA:/tmp/img.png", "").strip()
        sc = self._make_sc(
            visible_prefix="Here is your result",
            clean_fn=clean_fn,
        )
        # final has a MEDIA tag that the clean function removes
        assert _stream_visible_matches_final(
            sc, "Here is your result MEDIA:/tmp/img.png"
        ) is True

    def test_mismatched_text_returns_false(self):
        sc = self._make_sc(visible_prefix="hello world")
        assert _stream_visible_matches_final(sc, "completely different") is False

    def test_issue_53449_scenario(self):
        """Exact scenario from issue #53449: visible_len=476, final_len=474,
        all flags default. Visible text differs from final by trailing cursor
        artifacts. Should match."""
        sc = self._make_sc(visible_prefix="Hello! Here is the answer \u2589")
        assert _stream_visible_matches_final(sc, "Hello! Here is the answer") is True

    def test_draft_streaming_disables_visible_match(self):
        """Draft frames write to _last_sent_text but do NOT constitute real
        message delivery.  _stream_visible_matches_final must return False
        when draft transport is active so the final answer still reaches the
        user through the regular send path."""
        sc = self._make_sc(visible_prefix="Hello world \u2589")
        sc._use_draft_streaming = True
        # Even though visible text covers the full final answer, draft
        # delivery means the user hasn't received a real message yet.
        assert _stream_visible_matches_final(sc, "Hello world") is False

    def test_missing_message_id_disables_visible_match(self):
        sc = self._make_sc(visible_prefix="Hello world \u2589")
        sc._message_id = None
        assert _stream_visible_matches_final(sc, "Hello world") is False

    def test_polling_only_consumer_rejects_visible_match(self):
        """When the stream consumer's polling task never found a consumer
        (_sc is a polling-only sentinel with no real state), visible text
        must not be trusted."""
        sc = self._make_sc(visible_prefix="Hello")
        # Simulate a sentinel that only polls — no real stream delivery state.
        sc._use_draft_streaming = False
        sc._visible_prefix = lambda: "Hello"
        sc._message_id = None
        assert _stream_visible_matches_final(sc, "Hello") is False
        # But when draft transport is active, even the same content is rejected
        sc._use_draft_streaming = True
        assert _stream_visible_matches_final(sc, "Hello") is False
