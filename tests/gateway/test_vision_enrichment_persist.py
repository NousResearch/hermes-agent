"""Text-mode image turns must not persist the vision-enrichment prose.

The enrichment ("[The user sent an image~ …]") is a model instruction, not authored text; the
durable user row keeps the caption plus ``@image:<path>`` directive lines — the desktop renders
its attachment thumbnail from those directives (``embedded-images.ts``), so persisting the prose
loses the image from history (#142500). Mirrors the Discord triggering-note contract in
``test_discord_triggering_note_persistence.py``: model text rides, authored text persists.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    return runner


def _telegram_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="dm1", chat_type="dm", user_id="u1", user_name="Owen")


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_telegram_source(), message_id="m1")


ENRICHED = (
    "[The user sent an image~ Here's what I can see:\nA stack trace.]\n"
    "[If you need a closer look, use vision_analyze with image_url: {img} ~]\n\n"
    "look at this error")


@pytest.mark.asyncio
async def test_enriched_image_turn_persists_caption_and_refs_not_prose(monkeypatch, tmp_path):
    """Model text keeps the enrichment; the durable row keeps caption + @image directives."""
    runner = _runner()
    key = "telegram:dm1"
    img = tmp_path / "shot.png"
    img.write_bytes(b"png")
    caption = "look at this error"
    monkeypatch.setattr(runner, "_decide_image_input_mode", lambda **kwargs: "text")
    monkeypatch.setattr(
        runner, "_enrich_message_with_vision",
        AsyncMock(return_value=ENRICHED.format(img=img)))

    model_text = await runner._enrich_inbound_images(
        _telegram_source(), key, caption, [str(img)])
    assert "The user sent an image" in model_text  # enrichment rides the model text
    assert runner._session_state(key).persistent.enriched_image_persist == (
        caption, [str(img)])  # buffered for the persist seam

    message_text, persist_user_message, _ts = runner._hmwa_apply_message_timestamp(
        _event(caption), model_text, session_key=key)
    assert "The user sent an image" in message_text
    assert "The user sent an image" not in persist_user_message
    assert persist_user_message == f"look at this error\n@image:{img}"
    # Consumed one-shot: a second consume (double persist, next turn) must find nothing.
    assert runner._consume_pending_enriched_image_persist(key) is None


@pytest.mark.asyncio
async def test_plain_turn_persist_unchanged():
    """Without an enrichment buffer the persist seam behaves exactly as before."""
    runner = _runner()
    message_text, persist_user_message, _ts = runner._hmwa_apply_message_timestamp(
        _event("hello there"), "hello there", session_key="telegram:dm1")
    assert message_text == "hello there"
    assert persist_user_message == "hello there"


@pytest.mark.asyncio
async def test_stale_buffer_cleared_by_next_prepare(monkeypatch):
    """A turn that dies between prepare and persist must not leak its caption into the next
    turn's durable row: the next inbound prepare clears the buffer, like native_image_paths."""
    runner = _runner()
    key = "telegram:dm1"
    runner._session_state(key).persistent.enriched_image_persist = ("stale caption", ["/gone.png"])

    await runner._prepare_inbound_message_text(
        event=_event("next message"), source=_telegram_source(), history=[], session_key=key)

    assert runner._session_state(key).persistent.enriched_image_persist is None
