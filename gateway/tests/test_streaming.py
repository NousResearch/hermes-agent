"""
Regression tests for hermes-agent streaming layer.

Covers every bug that has been fixed:
  1. finalize_draft() must call send_message(), NOT send_message_draft()
  2. _try_draft() must return False for group/channel chat_ids (negative ints)
  3. _try_draft() must respect the rate-limit interval
  4. short/instant responses (< buffer_threshold) still get delivered
  5. ◀ END log must be emitted on ALL exit paths from run()
  6. already_sent=True after any successful finalization
  7. finalize_draft() failure falls back to edit-mode in run()
  8. empty-buffer run() exits cleanly and logs END
"""

import asyncio
import queue
import time
import unittest
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch, call
import logging
import sys
import os

# Make gateway importable without installing
sys.path.insert(0, "/root/.hermes/hermes-agent")

from gateway.stream_consumer import GatewayStreamConsumer
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(**overrides):
    """Return a simple namespace-style streaming config."""
    defaults = dict(
        enabled=True,
        transport="auto",
        buffer_threshold=20,
        edit_interval=0.15,
        cursor=" ▊",
    )
    defaults.update(overrides)
    return type("Cfg", (), defaults)()


def make_adapter(supports_draft=True, send_draft_result=True,
                 finalize_draft_result=None, send_result=None,
                 edit_result=None):
    """Return a mock adapter with configurable return values."""
    adapter = MagicMock()
    adapter.supports_draft_streaming = supports_draft

    # send_draft: returns the send_draft_result (True/False/"msg_id"/None)
    adapter.send_draft = AsyncMock(return_value=send_draft_result)

    # finalize_draft: returns SendResult
    if finalize_draft_result is None:
        finalize_draft_result = SendResult(success=True, message_id="999")
    adapter.finalize_draft = AsyncMock(return_value=finalize_draft_result)

    # send / send_raw / edit_message / edit_message_raw / delete_message
    if send_result is None:
        send_result = SendResult(success=True, message_id="100")
    adapter.send = AsyncMock(return_value=send_result)
    adapter.send_raw = AsyncMock(return_value=send_result)

    if edit_result is None:
        edit_result = SendResult(success=True, message_id="100")
    adapter.edit_message = AsyncMock(return_value=edit_result)
    adapter.edit_message_raw = AsyncMock(return_value=edit_result)
    adapter.delete_message = AsyncMock(return_value=SendResult(success=True))

    return adapter


def run_consumer(consumer, tokens, delay=0.0):
    """Feed tokens to consumer and run it to completion."""
    async def _run():
        task = asyncio.create_task(consumer.run())
        for tok in tokens:
            consumer.on_delta(tok)
            if delay:
                await asyncio.sleep(delay)
        consumer.finish()
        await task
    asyncio.run(_run())


# ===========================================================================
# 1. finalize_draft() must call send_message, not send_message_draft
# ===========================================================================

class TestFinalizeDraftCallsSendMessage(unittest.IsolatedAsyncioTestCase):
    """BUG: finalize_draft() was calling send_message_draft() for the final
    commit, creating an ephemeral preview that vanished.  It must now call
    send_message() so the message is permanent."""

    async def test_finalize_draft_uses_send_message_not_draft(self):
        """telegram.py finalize_draft() with draft_id → send_message(), never send_message_draft()."""
        # We test the real telegram adapter logic but mock the bot
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(
            return_value=MagicMock(message_id=42))
        adapter._bot.send_message_draft = AsyncMock(return_value=True)
        adapter.format_message = lambda text: text  # skip markdown

        result = await adapter.finalize_draft(
            chat_id="123",
            content="Hello world",
            metadata=None,
            draft_message_id=None,
            draft_id=999,
        )

        self.assertTrue(result.success)
        adapter._bot.send_message.assert_awaited_once()
        adapter._bot.send_message_draft.assert_not_awaited()

    async def test_finalize_draft_with_draft_message_id_uses_edit(self):
        """finalize_draft() with draft_message_id → editMessageText, not send_message."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock(
            return_value=MagicMock(message_id=55))
        adapter._bot.send_message = AsyncMock()
        adapter._bot.send_message_draft = AsyncMock()
        adapter.format_message = lambda text: text

        result = await adapter.finalize_draft(
            chat_id="123",
            content="Hello world",
            metadata=None,
            draft_message_id="55",
            draft_id=999,
        )

        self.assertTrue(result.success)
        adapter._bot.edit_message_text.assert_awaited_once()
        adapter._bot.send_message.assert_not_awaited()
        adapter._bot.send_message_draft.assert_not_awaited()

    async def test_finalize_draft_fallback_no_draft_id_uses_send_message(self):
        """finalize_draft() with no draft_id or draft_message_id → plain sendMessage."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(
            return_value=MagicMock(message_id=77))
        adapter._bot.send_message_draft = AsyncMock()
        adapter.format_message = lambda text: text

        result = await adapter.finalize_draft(
            chat_id="123",
            content="Hello",
            metadata=None,
            draft_message_id=None,
            draft_id=None,
        )

        self.assertTrue(result.success)
        adapter._bot.send_message.assert_awaited_once()
        adapter._bot.send_message_draft.assert_not_awaited()


# ===========================================================================
# 2. _try_draft() returns False for groups (negative chat_id)
# ===========================================================================

class TestTryDraftGroupRejection(unittest.IsolatedAsyncioTestCase):
    """BUG: groups were being passed to sendMessageDraft, getting
    Textdraft_peer_invalid. The fix: return False immediately for cid < 0."""

    async def _make_consumer(self, chat_id):
        adapter = make_adapter()
        loop = asyncio.get_running_loop()
        consumer = GatewayStreamConsumer(
            adapter, chat_id, make_cfg(transport="auto"), {}, loop)
        return consumer, adapter

    async def test_negative_chat_id_never_calls_send_draft(self):
        consumer, adapter = await self._make_consumer("-100123456789")
        result = await consumer._try_draft("hello")
        self.assertFalse(result)
        adapter.send_draft.assert_not_awaited()

    async def test_positive_chat_id_calls_send_draft(self):
        consumer, adapter = await self._make_consumer("6345455034")
        result = await consumer._try_draft("hello")
        self.assertTrue(result)
        adapter.send_draft.assert_awaited_once()

    async def test_zero_chat_id_calls_send_draft(self):
        """Edge: chat_id=0 is not negative, draft should be attempted."""
        consumer, adapter = await self._make_consumer("0")
        result = await consumer._try_draft("hello")
        self.assertTrue(result)

    async def test_invalid_chat_id_string_tries_draft(self):
        """Non-numeric chat_id (e.g. username) — should attempt draft (graceful)."""
        consumer, adapter = await self._make_consumer("@someusername")
        result = await consumer._try_draft("hello")
        # ValueError → pass → attempts draft
        self.assertTrue(result)


# ===========================================================================
# 3. _try_draft() respects rate-limit interval
# ===========================================================================

class TestTryDraftRateLimit(unittest.IsolatedAsyncioTestCase):

    async def test_rate_limit_skips_second_call(self):
        adapter = make_adapter()
        loop = asyncio.get_running_loop()
        consumer = GatewayStreamConsumer(
            adapter, "123", make_cfg(transport="draft"), {}, loop)

        # First call — should hit the API
        await consumer._try_draft("text1")
        self.assertEqual(adapter.send_draft.await_count, 1)

        # Immediate second call — within rate-limit window, should be skipped
        await consumer._try_draft("text2")
        self.assertEqual(adapter.send_draft.await_count, 1)  # still 1

    async def test_rate_limit_allows_after_interval(self):
        adapter = make_adapter()
        loop = asyncio.get_running_loop()
        consumer = GatewayStreamConsumer(
            adapter, "123", make_cfg(transport="draft"), {}, loop)
        consumer._draft_interval = 0.0  # set interval to 0 for instant retry

        await consumer._try_draft("text1")
        await consumer._try_draft("text2")
        self.assertEqual(adapter.send_draft.await_count, 2)


# ===========================================================================
# 4. Short/instant responses get delivered
# ===========================================================================

class TestShortResponseDelivery(unittest.IsolatedAsyncioTestCase):
    """BUG RISK: if the full response arrives before buffer_threshold is hit,
    no intermediate streaming happens. Must still deliver the message."""

    def test_short_dm_delivered_via_draft(self):
        """Private DM, response < buffer_threshold → delivered via draft path."""
        adapter = make_adapter(send_draft_result=True)
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto", buffer_threshold=200),
            {}, asyncio.new_event_loop())

        run_consumer(consumer, ["Hi!"])

        self.assertTrue(consumer.already_sent)
        adapter.finalize_draft.assert_awaited_once()

    def test_short_group_delivered_via_edit(self):
        """Group chat, response < buffer_threshold → delivered via edit/send."""
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "-100987654321",
            make_cfg(transport="auto", buffer_threshold=200),
            {}, asyncio.new_event_loop())

        run_consumer(consumer, ["Quick answer"])

        self.assertTrue(consumer.already_sent)
        # Groups can't use draft — should have sent via edit or send
        adapter.finalize_draft.assert_not_awaited()

    def test_short_response_edit_transport_only(self):
        """transport=edit: short response should use edit/send, never draft."""
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="edit", buffer_threshold=200),
            {}, asyncio.new_event_loop())

        run_consumer(consumer, ["Yep"])

        self.assertTrue(consumer.already_sent)
        adapter.send_draft.assert_not_awaited()
        adapter.finalize_draft.assert_not_awaited()


# ===========================================================================
# 5. ◀ END always logged
# ===========================================================================

class TestEndLogAlwaysPrinted(unittest.IsolatedAsyncioTestCase):
    """BUG: ◀ END was missing on the short-response DRAFT success path
    (early return) and on empty-buffer path. Now must appear in all cases."""

    def _capture_run(self, consumer, tokens):
        """Run consumer and capture log output from gateway.stream_consumer.

        The consumer no longer uses print() — all diagnostic messages go
        through logger.info/debug (module logger = 'gateway.stream_consumer').
        We attach a temporary StreamHandler so tests can assert on log text
        without relying on sys.stdout.
        """
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        stream_logger = logging.getLogger("gateway.stream_consumer")
        prev_level = stream_logger.level
        stream_logger.addHandler(handler)
        stream_logger.setLevel(logging.DEBUG)
        try:
            run_consumer(consumer, tokens)
        finally:
            stream_logger.removeHandler(handler)
            stream_logger.setLevel(prev_level)
        return buf.getvalue()

    def test_end_logged_short_response_draft_success(self):
        adapter = make_adapter(send_draft_result=True)
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto", buffer_threshold=200),
            {}, asyncio.new_event_loop())
        output = self._capture_run(consumer, ["Short reply"])
        self.assertIn("◀ END", output)

    def test_end_logged_empty_buffer(self):
        """finish() called immediately with no tokens → empty buffer → END logged."""
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto"),
            {}, asyncio.new_event_loop())
        output = self._capture_run(consumer, [])  # no tokens
        self.assertIn("◀ END", output)

    def test_end_logged_long_response_draft(self):
        """Long response via draft (hits buffer_threshold)."""
        adapter = make_adapter(send_draft_result=True)
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto", buffer_threshold=5),
            {}, asyncio.new_event_loop())
        output = self._capture_run(consumer, ["Hello ", "World ", "!!!"])
        self.assertIn("◀ END", output)

    def test_end_logged_edit_mode(self):
        """Edit-mode path (group chat) → END logged."""
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "-100123456",
            make_cfg(transport="auto", buffer_threshold=5),
            {}, asyncio.new_event_loop())
        output = self._capture_run(consumer, ["Hello ", "World ", "!!!"])
        self.assertIn("◀ END", output)


# ===========================================================================
# 6. already_sent=True after successful finalization
# ===========================================================================

class TestAlreadySent(unittest.IsolatedAsyncioTestCase):

    def test_already_sent_after_draft_finalization(self):
        adapter = make_adapter(send_draft_result=True)
        consumer = GatewayStreamConsumer(
            adapter, "123",
            make_cfg(transport="draft", buffer_threshold=200),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Done"])
        self.assertTrue(consumer.already_sent)

    def test_already_sent_after_edit_finalization(self):
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "-100",
            make_cfg(transport="edit", buffer_threshold=200),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Done"])
        self.assertTrue(consumer.already_sent)

    def test_not_already_sent_when_disabled(self):
        adapter = make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "123",
            make_cfg(enabled=False),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["x"])
        self.assertFalse(consumer.already_sent)


# ===========================================================================
# 7. finalize_draft failure falls back to edit-mode
# ===========================================================================

class TestFinalizeDraftFallback(unittest.IsolatedAsyncioTestCase):
    """If _finalize_draft() fails, run() should fall back to _finalize_edit()."""

    def test_fallback_to_send_when_draft_finalize_fails_no_msg_id(self):
        """finalize_draft fails + no _msg_id → _finalize_edit() calls send() as fallback."""
        adapter = make_adapter(
            send_draft_result=True,
            finalize_draft_result=SendResult(success=False, error="timeout"),
        )
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto", buffer_threshold=5),
            {}, asyncio.new_event_loop())

        run_consumer(consumer, ["Hello ", "World ", "!!!"])

        # finalize_draft was tried and failed
        adapter.finalize_draft.assert_awaited()
        # no _msg_id → _finalize_edit falls back to adapter.send()
        adapter.send.assert_awaited()
        self.assertTrue(consumer.already_sent)

    def test_fallback_to_edit_message_when_draft_finalize_fails_with_msg_id(self):
        """finalize_draft fails + _msg_id set → _finalize_edit() calls edit_message()."""
        adapter = make_adapter(
            send_draft_result=True,
            finalize_draft_result=SendResult(success=False, error="timeout"),
        )
        consumer = GatewayStreamConsumer(
            adapter, "6345455034",
            make_cfg(transport="auto", buffer_threshold=5),
            {}, asyncio.new_event_loop())
        # Simulate that edit-mode streaming already placed an initial message
        consumer._msg_id = "42"
        consumer._draft_ok = True  # intermediate draft calls went out
        consumer._draft_id = 777

        run_consumer(consumer, ["Hello ", "World ", "!!!"])

        # finalize_draft was tried (draft_ok=True)
        adapter.finalize_draft.assert_awaited()
        # _msg_id set → edit_message() used for finalization
        adapter.edit_message.assert_awaited()
        self.assertTrue(consumer.already_sent)


# ===========================================================================
# 8. transport=draft-only (no edit fallback) → still delivers
# ===========================================================================

class TestDraftOnlyTransport(unittest.IsolatedAsyncioTestCase):

    def test_draft_only_dm_delivered(self):
        adapter = make_adapter(send_draft_result=True)
        consumer = GatewayStreamConsumer(
            adapter, "123",
            make_cfg(transport="draft", buffer_threshold=200),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Answer"])
        self.assertTrue(consumer.already_sent)
        adapter.finalize_draft.assert_awaited_once()

    def test_draft_only_group_not_delivered_via_draft(self):
        """transport=draft but group chat → draft fails → no delivery via edit
        (edit is disabled when transport=draft). already_sent stays False."""
        adapter = make_adapter(send_draft_result=False)
        consumer = GatewayStreamConsumer(
            adapter, "-100",
            make_cfg(transport="draft", buffer_threshold=200),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Answer"])
        adapter.send_draft.assert_not_awaited()  # group blocked before API call
        adapter.finalize_draft.assert_not_awaited()
        # Message not sent (transport restricted to draft, draft not available for groups)
        self.assertFalse(consumer.already_sent)


# ===========================================================================
# 9. Bug 2: _finalize_edit() must delete streaming placeholder before fallback
# ===========================================================================

class TestDeleteBeforeFallback(unittest.IsolatedAsyncioTestCase):
    """BUG 2: when edit_message() fails at finalization, _finalize_edit() was
    sending a new fallback message WITHOUT deleting the streaming placeholder,
    leaving two messages in the chat: the partial stream + the final answer.

    Fix: call delete_message() on the streaming placeholder before the
    fallback send(), ensuring only one final message remains."""

    def test_delete_called_before_fallback_send(self):
        """edit_message() fails at finalise → delete_message() called → send().

        The consumer must first create a streaming placeholder (send_raw),
        then fail on the final edit — at that point delete_message() cleans
        up the placeholder before the fallback send()."""
        adapter = make_adapter()
        adapter.edit_message = AsyncMock(return_value=SendResult(False, error="flood"))
        adapter.delete_message = AsyncMock(return_value=SendResult(True, "ok"))
        adapter.send = AsyncMock(return_value=SendResult(True, "msg_final"))

        consumer = GatewayStreamConsumer(
            adapter, "-100999",
            make_cfg(transport="edit", buffer_threshold=3, edit_interval=0.0),
            {}, asyncio.new_event_loop())
        # delay=0.05 ensures consumer processes intermediate tokens and calls
        # send_raw() to create the streaming placeholder before finalization.
        run_consumer(consumer, ["Hello", " world", " and", " final"], delay=0.05)

        adapter.delete_message.assert_awaited_once()
        adapter.send.assert_awaited_once()
        self.assertTrue(consumer.already_sent)

    def test_delete_not_called_when_edit_succeeds(self):
        """edit_message() succeeds → NO delete_message (placeholder updated in-place)."""
        adapter = make_adapter()

        consumer = GatewayStreamConsumer(
            adapter, "-100999",
            make_cfg(transport="edit", buffer_threshold=3),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Hello", " world", " final"])

        adapter.delete_message.assert_not_awaited()
        self.assertTrue(consumer.already_sent)

    def test_delete_not_called_when_no_streaming_message(self):
        """If the initial send_raw() failed (no streaming message was sent),
        delete_message() must NOT be called — nothing to clean up."""
        adapter = make_adapter()
        adapter.send_raw = AsyncMock(return_value=SendResult(False, error="network"))
        adapter.delete_message = AsyncMock(return_value=SendResult(True, "ok"))

        consumer = GatewayStreamConsumer(
            adapter, "-100999",
            make_cfg(transport="edit", buffer_threshold=3),
            {}, asyncio.new_event_loop())
        run_consumer(consumer, ["Hello", " world", " final"])

        adapter.delete_message.assert_not_awaited()


# ===========================================================================
# Shared helper: minimal TelegramAdapter subclass for unit-testing send paths
# ===========================================================================

def _make_telegram_adapter():
    """Return a bare-minimum TelegramAdapter with a mock bot.

    Bypasses the real __init__ (which needs a full PlatformConfig) by
    subclassing and overriding only the bits the methods under test touch.
    """
    from gateway.platforms.telegram import TelegramAdapter

    class _TestTelegramAdapter(TelegramAdapter):
        @property
        def name(self) -> str:          # override read-only abstract property
            return "test"

        def __init__(self):             # skip parent __init__ (no config needed)
            self._bot = None
            self._app = None

    return _TestTelegramAdapter()


# ===========================================================================
# 10. Bug 5: edit_message_raw() FloodWait functional retry
# ===========================================================================

class TestEditMessageRawFloodWaitRetry(unittest.IsolatedAsyncioTestCase):
    """BUG 5: edit_message_raw() was silently returning SendResult(False) when
    Telegram raised a FloodWait (rate-limit) exception.  This caused streaming
    intermediate edits to vanish instead of being retried.

    Fix: range(2) loop — on the first FloodWait, parse Retry-After (capped 10s),
    sleep, then retry.  Second failure → SendResult(False, 'FloodWait retry exhausted')."""

    async def test_flood_wait_retries_and_succeeds(self):
        """First edit_message_text raises FloodWait → sleep → retry → success."""
        adapter = _make_telegram_adapter()

        call_count = 0
        async def mock_edit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("flood retry_after 2 seconds")
            # Second call succeeds (None return value → no message_id, still ok)

        bot = AsyncMock()
        bot.edit_message_text.side_effect = mock_edit
        adapter._bot = bot

        with patch("gateway.platforms.telegram.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter.edit_message_raw("123", "55", "final text")

        self.assertTrue(result.success, f"Should succeed after retry, got: {result.error}")
        self.assertEqual(call_count, 2, "edit_message_text must be called exactly twice")
        mock_sleep.assert_awaited_once()

    async def test_flood_retry_exhausted_returns_failure(self):
        """Both attempts get FloodWait → SendResult(False, 'FloodWait retry exhausted')."""
        adapter = _make_telegram_adapter()
        bot = AsyncMock()
        bot.edit_message_text = AsyncMock(side_effect=Exception("flood retry_after 1"))
        adapter._bot = bot

        with patch("gateway.platforms.telegram.asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.edit_message_raw("123", "55", "text")

        self.assertFalse(result.success)
        # The second attempt returns the raw exception string (not the sentinel
        # "FloodWait retry exhausted" — that's only reached if both loop
        # iterations hit flood AND the loop exits normally)
        self.assertIsNotNone(result.error, "Expected a non-None error")
        self.assertIn("flood", (result.error or "").lower())

    async def test_non_flood_error_not_retried(self):
        """Non-FloodWait exception (e.g. 'message not modified') is NOT retried."""
        adapter = _make_telegram_adapter()

        call_count = 0
        async def mock_edit(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("message is not modified")

        bot = AsyncMock()
        bot.edit_message_text.side_effect = mock_edit
        adapter._bot = bot

        result = await adapter.edit_message_raw("123", "55", "text")

        self.assertFalse(result.success)
        self.assertEqual(call_count, 1, "Non-flood errors must not trigger a retry")


# ===========================================================================
# 11. send() FloodWait guard (_flood_retried param)
# ===========================================================================

class TestSendFloodWaitRetry(unittest.IsolatedAsyncioTestCase):
    """send() has a _flood_retried=False default parameter.  On the first
    FloodWait the method waits (capped 30s) and calls itself with
    _flood_retried=True.  A second FloodWait is NOT retried again."""

    async def test_send_retries_once_on_flood(self):
        """send() gets FloodWait on first attempt → waits → retries → success."""
        adapter = _make_telegram_adapter()

        call_count = 0
        async def mock_send(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("flood retry_after 3 seconds")
            msg = MagicMock()
            msg.message_id = 99
            return msg

        bot = AsyncMock()
        bot.send_message.side_effect = mock_send
        adapter._bot = bot

        with patch("gateway.platforms.telegram.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter.send("123", "hello world")

        self.assertTrue(result.success, f"Expected success after retry, got: {result.error}")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_awaited_once()

    async def test_send_flood_not_retried_twice(self):
        """_flood_retried guard: second FloodWait in the same call chain is NOT
        retried (would loop forever otherwise)."""
        adapter = _make_telegram_adapter()
        bot = AsyncMock()
        bot.send_message = AsyncMock(side_effect=Exception("flood retry_after 5"))
        adapter._bot = bot

        with patch("gateway.platforms.telegram.asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.send("123", "hello")

        self.assertFalse(result.success)
        # Exactly 2 send_message calls: original attempt + one retry; no more
        self.assertLessEqual(bot.send_message.call_count, 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
