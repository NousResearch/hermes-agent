"""Regression coverage for the gateway final-send flood-control race.

Incident (2026-08-05 15:15 ET, Telegram group, 4070-char final response
delivered twice):

    15:15:10 MarkdownV2 edit failed, falling back to plain text: Flood
             control exceeded. Retry in 37 seconds
    15:15:10 Telegram flood control, waiting 37.0s
    15:15:10 Telegram flood control on send (attempt 1/3), retrying in
             37.0s: Flood control exceeded
    15:15:15 gateway.platforms.base: Sending response (4070 chars) to
             -1003725014629
    15:15:15 Telegram flood control on send (attempt 1/3), retrying in
             32.0s

The stream consumer's finalize edit hit Telegram flood control (fast,
"Retry in 37 seconds") and fell back to a fresh ``send()`` — and *that*
send hit flood control too, entering a retry sleep governed by Telegram's
own ``retry_after`` (37s). Meanwhile ``gateway/run.py`` only waited a flat
5 seconds (``asyncio.wait_for(stream_task, timeout=5.0)``) for the whole
final-delivery attempt before giving up, cancelling the consumer's task,
and falling through to the duplicate-send decision with
``final_response_sent`` / ``final_content_delivered`` still False (the
attempt hadn't resolved yet). The gateway then sent its own "final"
response — and because the abandoned send can still reach the platform
after being locally cancelled/abandoned, the user got the answer twice.

The fix (``gateway/run.py``: ``_await_stream_task_before_final_decision``;
``gateway/stream_consumer.py``: ``GatewayStreamConsumer.
final_delivery_in_progress``) extends the wait, bounded, whenever the
consumer reports a final-delivery attempt still in flight, instead of
racing a fixed short timeout against an arbitrarily long platform-mandated
retry.

Determinism: the mocked flood-controlled ``send()`` does not sleep for a
fixed duration and race that duration against a scaled timeout (that was
the source of the original flakiness — two independently-timed waits
compared against each other under real scheduling jitter). Instead it
blocks on an ``asyncio.Event`` (``send_release``) that only the test
controls, and signals ``send_started`` the instant it begins blocking.
The test awaits ``send_started`` (no timing guess needed — this resolves
the moment the send is genuinely in flight), lets the gateway's *base*
timeout genuinely elapse (a single one-sided sleep, safe because nothing
else is racing it — the send cannot complete on its own), and only then
sets ``send_release``. Resolution after that point is driven by the event
loop waking the extended ``wait_for`` on task completion, not by more
sleeping.
"""

import asyncio
import importlib
import time

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import _DONE, GatewayStreamConsumer, StreamConsumerConfig

FULL_RESPONSE = "y" * 200


class FloodyFallbackAdapter(BasePlatformAdapter):
    """Telegram-shaped adapter reproducing the incident's exact call shape.

    The cosmetic finalize edit ("MarkdownV2 edit failed... Retry in 37
    seconds") always fails with flood control immediately — matching the
    incident, where the edit failure itself resolved fast and it was the
    *fallback* send that entered the long retry sleep. ``FALLBACK_ON_
    FINAL_EDIT_FLOOD`` + ``RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK`` route
    that fallback through a fresh ``send()`` call, which blocks on
    ``send_release`` until the test lets it through — modeling a
    flood-controlled retry that's still in flight when the gateway's
    duplicate-send decision runs. ``asyncio.shield`` models the fact that
    a request already reached Telegram: a local ``cancel()`` of the
    waiting task cannot un-send it, so delivery still completes once
    ``send_release`` is set even if the gateway gave up waiting.
    """

    REQUIRES_EDIT_FINALIZE = True
    FALLBACK_ON_FINAL_EDIT_FLOOD = True
    RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.edits = []
        self.deleted = set()
        self._next_id = 0
        self.send_started = asyncio.Event()
        self.send_release = asyncio.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        self._next_id += 1
        return f"m-{self._next_id}"

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        async def _deliver():
            self.send_started.set()
            await self.send_release.wait()
            message_id = self._mint_id()
            self.sent.append({"chat_id": chat_id, "content": content, "message_id": message_id})
            return SendResult(success=True, message_id=message_id)

        return await asyncio.shield(_deliver())

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None,
    ) -> SendResult:
        if not finalize:
            self.edits.append(
                {"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": finalize}
            )
            return SendResult(success=True, message_id=message_id)
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": True}
        )
        return SendResult(
            success=False,
            error="Flood control exceeded. Retry in 37 seconds",
            retry_after=37.0,
        )

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.add(str(message_id))
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    def visible_full_response_messages(self):
        """Messages still visible in the chat carrying the exact final text."""
        return [
            m for m in self.sent
            if m["content"] == FULL_RESPONSE and m["message_id"] not in self.deleted
        ]


def _make_consumer(adapter):
    consumer = GatewayStreamConsumer(adapter, "chat-1", StreamConsumerConfig(cursor=" ▉"))
    # Streaming already showed the complete answer (a real preview message
    # exists and matches the final text exactly) — only the cosmetic
    # finalize edit is left to run. Setting this up directly (rather than
    # depending on exactly how a delta callback and the completion sentinel
    # happen to interleave through a background thread) keeps the test
    # deterministic: it exercises got_done's finalize/fallback handling in
    # run() every time.
    consumer._message_id = "preview-stale"
    consumer._last_sent_text = FULL_RESPONSE
    consumer._already_sent = True
    consumer._accumulated = FULL_RESPONSE
    consumer._message_created_ts = time.monotonic() - 1000.0
    consumer._queue.put(_DONE)
    return consumer


async def _legacy_wait_then_decide(stream_task, base_timeout: float):
    """Pre-fix gateway/run.py logic: flat short timeout, then cancel and give up.

    This is the literal control flow every call site used to run (only the
    numeric timeout is shrunk for test speed — the shape is unchanged: no
    extension, no in-flight check).
    """
    try:
        await asyncio.wait_for(stream_task, timeout=base_timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass


async def _run_flood_race(monkeypatch, *, base_timeout: float, max_timeout: float):
    gateway_run = importlib.import_module("gateway.run")

    adapter = FloodyFallbackAdapter()
    consumer = _make_consumer(adapter)
    stream_task = asyncio.create_task(consumer.run())

    # Resolves the instant the flood-controlled send actually starts
    # blocking — no timing guess, no race.
    await adapter.send_started.wait()

    helper = getattr(gateway_run, "_await_stream_task_before_final_decision", None)
    if helper is not None:
        monkeypatch.setattr(gateway_run, "_STREAM_FINAL_WAIT_BASE_TIMEOUT_SECONDS", base_timeout)
        monkeypatch.setattr(gateway_run, "_STREAM_FINAL_WAIT_MAX_TIMEOUT_SECONDS", max_timeout)
        decision_task = asyncio.create_task(helper(stream_task, consumer))
    else:
        decision_task = asyncio.create_task(_legacy_wait_then_decide(stream_task, base_timeout))

    # One-sided wait: guarantees the base timeout has genuinely elapsed.
    # Nothing races this — the send cannot complete on its own, it is
    # blocked on send_release, which we haven't set yet.
    await asyncio.sleep(base_timeout + 0.05)

    # Now let the "flood-controlled" send resolve. Fixed code is still
    # inside its extended wait at this point (bounded by max_timeout) and
    # will observe completion via the event loop, not more sleeping.
    # Unpatched code has already given up and moved on.
    adapter.send_release.set()

    await decision_task

    already_confirmed = bool(consumer.final_response_sent or consumer.final_content_delivered)
    if not already_confirmed:
        # Mimic the real outer caller: it only performs its own "normal
        # final send" when the gateway did not confirm streamed delivery —
        # exactly the decision that races the in-flight fallback send.
        await adapter.send(chat_id="chat-1", content=FULL_RESPONSE, metadata=None)

    # Let the (now-released) shielded delivery finish landing.
    for _ in range(200):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)

    return adapter, already_confirmed


@pytest.mark.asyncio
async def test_flood_controlled_finalize_delivers_final_response_once(monkeypatch):
    """The completed answer must reach the chat exactly once even when the
    stream consumer's flood-controlled fallback send is still in flight at
    the moment the gateway decides whether to send its own final response.
    """
    adapter, already_confirmed = await _run_flood_race(
        monkeypatch, base_timeout=0.05, max_timeout=1.0,
    )

    visible = adapter.visible_full_response_messages()
    assert len(visible) == 1, (
        f"final response visible in {len(visible)} separate messages "
        f"(expected exactly 1): {visible!r}; all sends={adapter.sent!r}"
    )
    assert already_confirmed is True, (
        "gateway did not wait for the in-flight fallback send to resolve "
        "before deciding whether to send its own duplicate final response"
    )
