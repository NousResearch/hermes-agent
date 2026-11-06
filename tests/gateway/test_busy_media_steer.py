"""Busy media steer: opt-in text-mode preparation and gating."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

def _runner(*, mode: str = "text") -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._busy_media_steer = mode
    return runner

def _source(chat_id: str = "media-steer") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name="tester",
    )

def _photo(source: SessionSource, *, text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/busy-media.png"],
        media_types=["image/png"],
    )

@pytest.mark.asyncio
async def test_for_steer_prep_forces_text_images_and_keeps_native_buffer():
    runner = _runner()
    source = _source()
    session_key = build_session_key(source)
    runner._decide_image_input_mode = lambda **_: "native"
    runner._enrich_message_with_vision = AsyncMock(return_value="VISION DESCRIPTION")
    runner._session_state(session_key).persistent.native_image_paths = ["/tmp/prior.png"]

    event = _photo(source)
    prepared = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        session_key=session_key,
        for_steer=True,
    )

    assert prepared is not None
    assert "VISION DESCRIPTION" in prepared
    assert runner._consume_pending_native_image_paths(session_key) == ["/tmp/prior.png"]
    assert event._gateway_busy_steer_prepared_text == prepared

@pytest.mark.asyncio
async def test_busy_media_steer_reuses_prepared_text_on_fallback():
    runner = _runner()
    source = _source()
    event = _photo(source, text="look")

    async def _prepare(**kwargs):
        kwargs["event"]._gateway_busy_steer_prepared_text = "PREPARED"
        return "PREPARED"

    runner._prepare_inbound_message_text = AsyncMock(side_effect=_prepare)

    first = await runner._prepare_busy_steer_text(event, "key")
    second = await runner._prepare_busy_steer_text(event, "key")

    assert first == "PREPARED"
    assert second == "PREPARED"
    assert runner._prepare_inbound_message_text.await_count == 1

    event._gateway_busy_steer_prepared_text = "CACHED"
    third = await runner._prepare_busy_steer_text(event, "key")
    assert third == "CACHED"
    assert runner._prepare_inbound_message_text.await_count == 1

@pytest.mark.asyncio
async def test_busy_media_steer_gate_allows_media_when_enabled():
    runner = _runner(mode="text")
    runner._prepare_busy_steer_text = AsyncMock(return_value="PREPARED")
    runner._try_agent_verb = lambda *args, **kwargs: True
    runner._fold_into_running_turn = lambda *args, **kwargs: None
    event = _photo(_source())
    agent = type("Agent", (), {"steer": lambda self, text: True})()

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)

    assert outcome.steered is True
    assert outcome.effective_mode == "steer"

@pytest.mark.asyncio
async def test_busy_media_steer_gate_queues_when_off():
    runner = _runner(mode="off")
    event = _photo(_source())

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", object())

    assert outcome.steered is False
    assert outcome.effective_mode == "queue"


def _voice(source: SessionSource, *, text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/busy-media.ogg"],
        media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
async def test_voice_only_stays_on_stt_branch():
    """A voice note must not enter the shared prep: only that branch writes the STT cache."""
    runner = _runner()
    source = _source()
    event = _voice(source)
    runner._pending_event_audio_paths = lambda ev: list(ev.media_urls)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="PREPARED")
    runner._transcribe_and_echo_pending_voice = AsyncMock(return_value=("VOICE TRANSCRIPT", ["t"]))

    text = await runner._prepare_busy_steer_text(event, "key")

    assert text == "VOICE TRANSCRIPT"
    runner._prepare_profile_scoped_inbound_message_text.assert_not_awaited()
    runner._transcribe_and_echo_pending_voice.assert_awaited_once()
    assert not hasattr(event, "_gateway_busy_steer_prepared_text")


@pytest.mark.asyncio
async def test_mixed_voice_and_image_transcribes_before_shared_prep():
    """Mixed media: the STT pass runs first so prep sees the cache and skips STT."""
    runner = _runner()
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/busy-media.ogg", "/tmp/busy-media.png"],
        media_types=["audio/ogg", "image/png"],
    )
    runner._pending_event_audio_paths = lambda ev: [ev.media_urls[0]]
    seen = {}

    async def _stt(ev, adapter, src, text, **kwargs):
        ev._gateway_pending_stt_text = "VOICE TRANSCRIPT"
        return "VOICE TRANSCRIPT", ["t"]

    async def _prep(**kwargs):
        seen["pending_stt"] = hasattr(kwargs["event"], "_gateway_pending_stt_text")
        kwargs["event"]._gateway_busy_steer_prepared_text = "PREPARED MIXED"
        return "PREPARED MIXED"

    runner._transcribe_and_echo_pending_voice = AsyncMock(side_effect=_stt)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(side_effect=_prep)

    text = await runner._prepare_busy_steer_text(event, "key")

    assert text == "PREPARED MIXED"
    assert seen["pending_stt"] is True
    runner._transcribe_and_echo_pending_voice.assert_awaited_once()


def test_burst_merge_invalidates_prepared_text_cache():
    """A photo merged in after steer prep must drop the derived text, or the burst is stale."""
    from gateway.platforms.base import merge_pending_message_event

    source = _source()
    existing = _photo(source)
    existing._gateway_pending_stt_text = "stale stt"
    existing._gateway_pending_stt_transcripts = ["stale"]
    existing._gateway_busy_steer_prepared_text = "STALE DESCRIPTION"
    pending = {build_session_key(source): existing}

    merge_pending_message_event(pending, build_session_key(source), _photo(source))

    assert pending[build_session_key(source)] is existing
    assert len(existing.media_urls) == 2
    assert not hasattr(existing, "_gateway_busy_steer_prepared_text")
    assert not hasattr(existing, "_gateway_pending_stt_text")


@pytest.mark.asyncio
async def test_for_steer_prep_is_idempotent_for_vision():
    """The queue fallback re-enters prep; the cached payload must short-circuit a second pass."""
    runner = _runner()
    source = _source()
    session_key = build_session_key(source)
    runner._decide_image_input_mode = lambda **_: "native"
    runner._enrich_message_with_vision = AsyncMock(return_value="VISION DESCRIPTION")
    event = _photo(source)

    first = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[], session_key=session_key, for_steer=True,
    )
    second = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[], session_key=session_key, for_steer=True,
    )

    assert first == second
    runner._enrich_message_with_vision.assert_awaited_once()


def test_loader_rejects_unknown_value_with_warning(monkeypatch, caplog):
    import logging

    import gateway.run_config_loaders as loaders
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        GatewayRunner, "_env_or_cfg_str", classmethod(lambda cls, *a, **k: "native"),
    )
    with caplog.at_level(logging.WARNING, logger=loaders.__name__):
        assert GatewayRunner._load_busy_media_steer() == "off"
    assert any("busy_media_steer" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_document_reaches_steer_text_through_shared_prep():
    """A document is not an image or voice: it must prepare as a path note, not fall to queue."""
    runner = _runner()
    source = _source()
    event = MessageEvent(
        text="see attached",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/busy-media.pdf"],
        media_types=["application/pdf"],
    )

    text = await runner._prepare_busy_steer_text(event, "key")

    assert "busy-media.pdf" in text
    assert "see attached" in text
    assert event._gateway_busy_steer_prepared_text == text


@pytest.mark.asyncio
async def test_empty_prepared_text_falls_back_to_queue():
    """Prep can come back empty (rejected @ reference); the gate must queue, not steer blank."""
    runner = _runner()
    runner._prepare_busy_steer_text = AsyncMock(return_value="")
    runner._try_agent_verb = lambda *a, **k: pytest.fail("must not steer an empty payload")
    event = _photo(_source())
    agent = type("Agent", (), {"steer": lambda self, text: True})()

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)

    assert outcome.steered is False
    assert outcome.effective_mode == "queue"
