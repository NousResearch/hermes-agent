"""F1 — a bounded PRE-TURN inbound enrichment pipeline (kanban t_d2133284).

2026-10-23 gateway DM outage: a Telegram DM arrived at 22:18:16 and the gateway went
silent for 27 minutes. The last log line was "Pre-analyzing 1 image(s) via
vision_analyze"; the temp image was written completely, so the hang was inside the
prep steps. The turn never started, so no reply was ever produced for that chat.

Root cause: the pre-turn enrichment path (image pre-analysis, STT, ``@`` context
expansion) runs BEFORE ``AIAgent.run_conversation()``, so it is outside the
turn-liveness watchdog's scope entirely, and it had no deadline of its own.

These tests pin the fix:
  * the pipeline deadline abandons a hung enrichment and CONTINUES (degraded, delivered);
  * an explicit refusal (context expansion -> None) still drops the message;
  * a deadline expiry never drops the message;
  * the budget resolves from config / env / default with sane clamping;
  * the image-prep step is bounded independently.

The project does not enable pytest-asyncio; async tests use ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.inbound_preprocess_deadline import (
    DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S,
    EXPIRED,
    MAX_INBOUND_PREPROCESS_TIMEOUT_S,
    MIN_INBOUND_PREPROCESS_TIMEOUT_S,
    resolve_inbound_preprocess_timeout,
    run_with_inbound_preprocess_deadline,
)


# --------------------------------------------------------------------------
# Budget resolution
# --------------------------------------------------------------------------

def test_default_budget_is_used_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", raising=False)
    assert resolve_inbound_preprocess_timeout(None) == DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S


def test_config_value_wins_over_env(monkeypatch):
    monkeypatch.setenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", "45")

    class _Cfg:
        inbound_preprocess_timeout_s = 90.0

    assert resolve_inbound_preprocess_timeout(_Cfg()) == 90.0


def test_env_is_used_when_config_has_no_value(monkeypatch):
    monkeypatch.setenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", "45")

    class _Cfg:
        inbound_preprocess_timeout_s = None

    assert resolve_inbound_preprocess_timeout(_Cfg()) == 45.0


def test_zero_opts_out(monkeypatch):
    monkeypatch.delenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", raising=False)

    class _Cfg:
        inbound_preprocess_timeout_s = 0

    assert resolve_inbound_preprocess_timeout(_Cfg()) == 0.0


def test_out_of_range_value_is_clamped(monkeypatch):
    monkeypatch.delenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", raising=False)

    class _TooBig:
        inbound_preprocess_timeout_s = 99_999.0

    class _TooSmall:
        inbound_preprocess_timeout_s = 1.0

    assert resolve_inbound_preprocess_timeout(_TooBig()) == MAX_INBOUND_PREPROCESS_TIMEOUT_S
    assert resolve_inbound_preprocess_timeout(_TooSmall()) == MIN_INBOUND_PREPROCESS_TIMEOUT_S


def test_nan_value_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("HERMES_INBOUND_PREPROCESS_TIMEOUT_S", raising=False)

    class _Cfg:
        inbound_preprocess_timeout_s = float("nan")

    assert resolve_inbound_preprocess_timeout(_Cfg()) == DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S


# --------------------------------------------------------------------------
# The deadline helper
# --------------------------------------------------------------------------

def test_successful_await_returns_its_value():
    async def _ok():
        return "enriched"

    assert asyncio.run(
        run_with_inbound_preprocess_deadline(_ok(), timeout_s=5.0)
    ) == "enriched"


def test_hung_await_returns_expired_sentinel():
    async def _hang():
        await asyncio.sleep(3600)

    async def _run():
        started = time.monotonic()
        result = await run_with_inbound_preprocess_deadline(_hang(), timeout_s=0.25)
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(_run())

    assert result is EXPIRED
    assert elapsed < 5.0, "the deadline must fire promptly, not hang with the awaitable"


def test_zero_budget_runs_unbounded():
    """Opt-out: the awaitable runs to completion with no deadline."""

    async def _slow():
        await asyncio.sleep(0.2)
        return "done"

    assert asyncio.run(
        run_with_inbound_preprocess_deadline(_slow(), timeout_s=0.0)
    ) == "done"


def test_cancellation_propagates_rather_than_becoming_a_timeout():
    """A shutdown cancellation must not be swallowed as a deadline expiry."""

    async def _cancelled():
        raise asyncio.CancelledError()

    async def _run():
        await run_with_inbound_preprocess_deadline(_cancelled(), timeout_s=5.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run())


def test_expired_is_distinguishable_from_none():
    """``None`` is a legitimate result for several enrichment paths; EXPIRED is not."""

    async def _returns_none():
        return None

    async def _run():
        return (await run_with_inbound_preprocess_deadline(_returns_none(), timeout_s=5.0),
                await run_with_inbound_preprocess_deadline(_returns_none(), timeout_s=5.0))

    first, second = asyncio.run(_run())
    assert first is None
    assert second is not EXPIRED


# --------------------------------------------------------------------------
# The inbound pipeline uses the deadline (behavioural, via the real mixin)
# --------------------------------------------------------------------------

def test_hung_image_enrichment_does_not_drop_the_message(monkeypatch):
    """The 2026-10-23 failure mode: enrichment hangs -> the message still gets a turn.

    The regression this pins: the inbound pipeline awaited the enrichment with no
    deadline, so a hung vision call held the message forever and it was never
    answered.
    """
    from gateway.run_inbound import GatewayInboundMixin

    class _Runner(GatewayInboundMixin):
        def __init__(self):
            self.config = type("C", (), {"inbound_preprocess_timeout_s": 0.25})()

        def _session_key_for_source(self, source):
            return "sess-1"

        def _consume_pending_native_image_paths(self, session_key):
            pass

        def _prefix_inbound_sender_context(self, event, source, text):
            return text

        def _classify_inbound_media(self, event, pending_stt):
            return (["/tmp/img.jpg"], [], [], [])

        async def _enrich_inbound_images(self, source, session_key, text, image_paths):
            await asyncio.sleep(3600)  # the 2026-10-23 hang
            return text + " [enriched]"

        async def _enrich_inbound_voice(self, event, source, text, audio_paths):
            return text

        def _prepend_inbound_media_file_notes(self, text, audio, video):
            return text

        def _prepend_inbound_document_notes(self, event, text):
            return text

        def _prepend_inbound_reply_context(self, event, source, text):
            return text

    runner = _Runner()
    from gateway.platforms.event import MessageEvent
    event = MessageEvent(text="hello")

    # The config value is clamped to the production floor (30s), which is far too
    # slow for a unit test. Patch the resolver so the deadline mechanism is
    # exercised with a short budget; the clamp itself is covered by the
    # budget-resolution tests above.
    import gateway.run_inbound as run_inbound

    monkeypatch.setattr(
        run_inbound, "resolve_inbound_preprocess_timeout", lambda _cfg: 0.25,
    )

    async def _run():
        started = time.monotonic()
        result = await runner._prepare_inbound_message_text(
            event=event, source=object(), history=[],
        )
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(_run())

    assert result == "hello", "the raw text must survive the enrichment timeout"
    assert elapsed < 5.0, "the pipeline must not wait for the hung enrichment"


def test_context_expansion_refusal_still_drops_the_message(monkeypatch):
    """A refusal (``None``) is NOT a timeout: the message is still dropped.

    Guard against the sentinel change accidentally turning a security refusal into
    a delivered message.
    """
    from gateway.run_inbound import GatewayInboundMixin

    class _Runner(GatewayInboundMixin):
        def __init__(self):
            self.config = type("C", (), {"inbound_preprocess_timeout_s": 30.0})()

        def _session_key_for_source(self, source):
            return "sess-1"

        def _consume_pending_native_image_paths(self, session_key):
            pass

        def _prefix_inbound_sender_context(self, event, source, text):
            return text

        def _classify_inbound_media(self, event, pending_stt):
            return ([], [], [], [])

        async def _enrich_inbound_voice(self, event, source, text, audio_paths):
            return text

        def _prepend_inbound_media_file_notes(self, text, audio, video):
            return text

        def _prepend_inbound_document_notes(self, event, text):
            return text

        def _prepend_inbound_reply_context(self, event, source, text):
            return text

        async def _expand_inbound_context_references(self, source, session_key, text):
            return None  # refused

    runner = _Runner()
    from gateway.platforms.event import MessageEvent
    event = MessageEvent(text="see @secret.env")

    result = asyncio.run(runner._prepare_inbound_message_text(
        event=event, source=object(), history=[],
    ))
    assert result is None, "an explicit refusal must still drop the message"


# --------------------------------------------------------------------------
# Image prep is bounded independently (F1b)
# --------------------------------------------------------------------------

def test_image_prep_is_bounded(monkeypatch):
    """``_prepare_image`` must not wait forever on a source that never resolves."""
    from tools import vision_tools

    monkeypatch.setattr(vision_tools, "_VISION_PREP_TIMEOUT", 0.25)

    async def _hang(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr("tools.image_source.resolve_image_source", _hang)

    from tools.vision_tools import _ImagePrepError

    async def _run():
        return await vision_tools._prepare_image(
            "https://example.invalid/x.jpg", None, None, validate_decode=False,
        )

    with pytest.raises(_ImagePrepError, match="Timed out"):
        asyncio.run(_run())
