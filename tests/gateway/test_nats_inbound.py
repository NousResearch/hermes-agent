"""Phase 4 (T4.4): inbound pipeline for the NATS gateway adapter.

Covers, in order of the prompt lifecycle (design doc §6.2):

* ``_extract_x_session`` — JSON branch returns the caller's session, the
  plain-text shorthand shortcut returns None, malformed inputs fall back
  gracefully, and empty / non-string values are rejected.
* ``_unpack_envelope`` — extension routing (image/audio/video/document),
  base64 decode failures surface as ``RuntimeError`` (→ SDK 400), and
  ``media_urls`` / ``media_types`` are aligned with the first entry
  driving ``MessageEvent.message_type``.
* ``_looks_like_command`` — conservative slash heuristic (matches paths,
  double-slashes, empty bodies).
* ``_on_prompt`` end-to-end — registers the stream + in-flight task,
  emits one final ``ResponseChunk`` for non-streaming runs, forwards
  slash commands through ``_message_handler``, unwinds the keep-alive
  and stream registration in the ``finally`` block.
* ``_run_keepalive`` — emits ``StatusChunk(status="ack")`` periodically
  and exits cleanly on cancellation.
* ``_pump_deltas`` — drains a queue, publishes one ResponseChunk per
  delta, returns on the ``None`` sentinel.
* ``send()`` — publishes when a stream is registered, returns a
  descriptive ``SendResult`` when it isn't.

Tests use the conftest ``natsagent`` mock (``ResponseChunk`` and
``StatusChunk`` are simple stand-ins that just record kwargs). The
``AIAgent`` construction inside ``_run_agent_sync`` is exercised via
``_run_text_prompt`` monkeypatches — we don't spin up a real agent.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.nats import (
    NatsAdapter,
    _extract_x_session,
    _final_response_text,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _valid_extra(**overrides) -> dict:
    base = {
        "servers": ["nats://127.0.0.1:4222"],
        "owner": "rene",
        "name": "gateway",
        "ack_keepalive_interval_s": 1,  # fast for tests
    }
    base.update(overrides)
    return base


def _build_adapter(**extra_overrides) -> NatsAdapter:
    return NatsAdapter(PlatformConfig(enabled=True, extra=_valid_extra(**extra_overrides)))


def _fake_stream(raw: bytes = b"") -> MagicMock:
    """Build a PromptStream-shaped MagicMock.

    ``_extract_session`` reads ``stream._request.data`` — wire that up so
    the adapter can peek the raw envelope for the ``x-session`` hack.
    """
    stream = MagicMock()
    stream.send = AsyncMock()
    request = MagicMock()
    request.data = raw
    stream._request = request
    return stream


@pytest.fixture(autouse=True)
def _fresh_natsagent_mock(monkeypatch):
    """Reset ResponseChunk / StatusChunk stand-ins between tests.

    The conftest planter installs classes with ``__init__`` kwargs; re-use
    them here by verifying they persist, but individual tests set up their
    own send side-effects so no per-test re-planting is needed.
    """
    return sys.modules["natsagent"]


# ---------------------------------------------------------------------------
# _extract_x_session
# ---------------------------------------------------------------------------


class TestExtractXSession:
    def test_returns_session_from_json_envelope(self):
        raw = b'{"prompt":"hi","x-session":"alice"}'
        assert _extract_x_session(raw) == "alice"

    def test_tolerates_leading_whitespace(self):
        # Protocol §5.3: first non-whitespace byte must be '{' for JSON.
        raw = b'   \n{"prompt":"hi","x-session":"alice"}'
        assert _extract_x_session(raw) == "alice"

    def test_plain_text_shorthand_returns_none(self):
        # Plain UTF-8 is the §5.3 shorthand ``{"prompt": <text>}`` —
        # there's no x-session field to extract.
        assert _extract_x_session(b"tell me a joke") is None

    def test_malformed_json_returns_none(self):
        assert _extract_x_session(b"{not json") is None

    def test_empty_payload_returns_none(self):
        assert _extract_x_session(b"") is None

    def test_whitespace_only_payload_returns_none(self):
        assert _extract_x_session(b"   \t\n") is None

    def test_non_object_json_returns_none(self):
        # ``[]`` is valid JSON but not an envelope.
        assert _extract_x_session(b"[]") is None

    def test_missing_field_returns_none(self):
        assert _extract_x_session(b'{"prompt":"hi"}') is None

    def test_non_string_field_returns_none(self):
        assert _extract_x_session(b'{"prompt":"hi","x-session":42}') is None

    def test_empty_string_returns_none(self):
        # An empty session is indistinguishable from "missing" for
        # routing purposes — fall back to the settings default.
        assert _extract_x_session(b'{"prompt":"hi","x-session":""}') is None

    def test_whitespace_only_session_returns_none(self):
        assert _extract_x_session(b'{"prompt":"hi","x-session":"   "}') is None


# ---------------------------------------------------------------------------
# _unpack_envelope — attachment routing
# ---------------------------------------------------------------------------


class TestUnpackEnvelope:
    @staticmethod
    def _make_attachment(filename: str, data: bytes):
        """Build a MagicMock that behaves like an SDK Attachment."""
        att = MagicMock()
        att.filename = filename
        att.to_bytes = MagicMock(return_value=data)
        return att

    def test_text_only_envelope_produces_text_message(self):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "hello"
        envelope.attachments = None

        prompt, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert prompt == "hello"
        assert urls == []
        assert types == []
        assert mtype is MessageType.TEXT

    def test_image_attachment_routes_to_photo(self, monkeypatch):
        adapter = _build_adapter()
        # cache_image_from_bytes validates magic bytes — mock it out to
        # avoid dragging image-validation concerns into the inbound test.
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_image_from_bytes",
            lambda data, ext=".jpg": f"/cache/img{ext}",
        )
        envelope = MagicMock()
        envelope.prompt = "look at this"
        envelope.attachments = [self._make_attachment("photo.png", b"PNGDATA")]

        prompt, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert prompt == "look at this"
        assert urls == ["/cache/img.png"]
        assert types == [MessageType.PHOTO.value]
        assert mtype is MessageType.PHOTO

    def test_document_attachment_routes_to_document(self, monkeypatch):
        adapter = _build_adapter()
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_document_from_bytes",
            lambda data, filename: f"/cache/{filename}",
        )
        envelope = MagicMock()
        envelope.prompt = "summarize"
        envelope.attachments = [self._make_attachment("report.pdf", b"PDFDATA")]

        _, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert urls == ["/cache/report.pdf"]
        assert types == [MessageType.DOCUMENT.value]
        assert mtype is MessageType.DOCUMENT

    def test_audio_extension_routes_to_audio(self, monkeypatch):
        adapter = _build_adapter()
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_audio_from_bytes",
            lambda data, ext=".ogg": f"/cache/audio{ext}",
        )
        envelope = MagicMock()
        envelope.prompt = ""
        envelope.attachments = [self._make_attachment("note.mp3", b"MP3DATA")]

        _, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert urls == ["/cache/audio.mp3"]
        assert types == [MessageType.AUDIO.value]
        assert mtype is MessageType.AUDIO

    def test_video_extension_routes_to_video(self, monkeypatch):
        adapter = _build_adapter()
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_video_from_bytes",
            lambda data, ext=".mp4": f"/cache/video{ext}",
        )
        envelope = MagicMock()
        envelope.prompt = ""
        envelope.attachments = [self._make_attachment("clip.webm", b"WEBMDATA")]

        _, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert urls == ["/cache/video.webm"]
        assert types == [MessageType.VIDEO.value]
        assert mtype is MessageType.VIDEO

    def test_first_attachment_drives_message_type(self, monkeypatch):
        # Mixed attachments — the primary MessageType reflects the first
        # entry because that's what the downstream routing (vision tool
        # vs. document reader) hooks off of.
        adapter = _build_adapter()
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_image_from_bytes",
            lambda data, ext=".jpg": f"/cache/img{ext}",
        )
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_document_from_bytes",
            lambda data, filename: f"/cache/{filename}",
        )
        envelope = MagicMock()
        envelope.prompt = ""
        envelope.attachments = [
            self._make_attachment("first.jpg", b"J"),
            self._make_attachment("second.pdf", b"P"),
        ]

        _, urls, types, mtype = adapter._unpack_envelope(envelope)
        assert len(urls) == 2
        assert types == [MessageType.PHOTO.value, MessageType.DOCUMENT.value]
        assert mtype is MessageType.PHOTO

    def test_base64_decode_failure_raises_runtime_error(self):
        adapter = _build_adapter()
        att = MagicMock()
        att.filename = "bad.pdf"
        att.to_bytes = MagicMock(side_effect=ValueError("invalid base64"))
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = [att]

        with pytest.raises(RuntimeError, match="base64 decode failed"):
            adapter._unpack_envelope(envelope)

    def test_non_image_bytes_with_image_extension_raises(self, monkeypatch):
        # The image cache validator rejects non-image bytes as a defense
        # against callers uploading HTML error pages as ``.jpg``. Ensure
        # that surface-level error converts to our RuntimeError so the
        # SDK emits a 400, not a 500 (§9.3).
        adapter = _build_adapter()

        def _fake_cache(data, ext=".jpg"):
            raise ValueError("Refusing to cache non-image data")
        monkeypatch.setattr(
            "gateway.platforms.nats.cache_image_from_bytes",
            _fake_cache,
        )

        att = MagicMock()
        att.filename = "fake.jpg"
        att.to_bytes = MagicMock(return_value=b"<html>oops")
        envelope = MagicMock()
        envelope.prompt = ""
        envelope.attachments = [att]

        with pytest.raises(RuntimeError, match="failed validation"):
            adapter._unpack_envelope(envelope)


# ---------------------------------------------------------------------------
# _enrich_event_with_media — Phase 8 fix (aligned with canonical gateway path)
# ---------------------------------------------------------------------------


class TestEnrichEventWithMedia:
    """Regression: NATS adapter folds ``media_urls`` into ``event.text``.

    Phase 4 cached attachments into ``media_urls`` but ``_run_text_prompt``
    only passed ``event.text`` to ``run_conversation``, so the agent never
    saw them. Phase 8's live T8.4 smoke caught this when
    03-prompt-attachment returned "I don't see an image". The fix mirrors
    :meth:`GatewayRunner._enrich_message_with_vision` (inline vision
    pre-analysis of images with the same output template) plus
    :meth:`GatewayRunner._handle_message`'s document path-note block —
    so the NATS adapter's user-facing contract matches Telegram / Discord
    / Slack byte-for-byte.
    """

    @staticmethod
    def _success_vision_result(description: str) -> str:
        import json as _json
        return _json.dumps({"success": True, "analysis": description})

    @pytest.mark.asyncio
    async def test_no_media_returns_event_unchanged(self):
        adapter = _build_adapter()
        src = MagicMock()
        event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=src)
        result = await adapter._enrich_event_with_media(event)
        assert result is event

    @pytest.mark.asyncio
    async def test_image_pre_analyzed_via_vision_tool(self, monkeypatch):
        adapter = _build_adapter()
        src = MagicMock()
        calls: List[dict] = []

        async def _fake_vision(image_url, user_prompt):
            calls.append({"image_url": image_url, "user_prompt": user_prompt})
            return self._success_vision_result("a blue banner with the word HERMES")

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _fake_vision,
        )

        event = MessageEvent(
            text="what is this?",
            message_type=MessageType.PHOTO,
            source=src,
            media_urls=["/cache/img.png"],
            media_types=[MessageType.PHOTO.value],
        )
        result = await adapter._enrich_event_with_media(event)

        assert len(calls) == 1
        assert calls[0]["image_url"] == "/cache/img.png"
        # Output matches the canonical gateway template (same keywords).
        assert "Here's what I can see" in result.text
        assert "a blue banner with the word HERMES" in result.text
        assert "/cache/img.png" in result.text
        # User's text preserved after the analysis.
        assert "what is this?" in result.text
        assert result.text.rstrip().endswith("what is this?")
        # Metadata passes through.
        assert result.media_urls == ["/cache/img.png"]
        assert result.media_types == [MessageType.PHOTO.value]
        assert result.message_type is MessageType.PHOTO
        assert result.source is src

    @pytest.mark.asyncio
    async def test_image_vision_failure_degrades_to_self_retry_note(self, monkeypatch):
        adapter = _build_adapter()

        async def _boom(image_url, user_prompt):
            raise RuntimeError("model quota exceeded")

        monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _boom)

        event = MessageEvent(
            text="what's in the photo?",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            media_urls=["/cache/img.png"],
            media_types=[MessageType.PHOTO.value],
        )
        result = await adapter._enrich_event_with_media(event)
        # Graceful degradation: the caller gets a fallback note pointing
        # the agent at vision_analyze for a retry, exactly like the
        # gateway's canonical path does.
        assert "vision_analyze" in result.text
        assert "/cache/img.png" in result.text
        assert "what's in the photo?" in result.text

    @pytest.mark.asyncio
    async def test_image_non_success_result_falls_back_to_retry_note(self, monkeypatch):
        adapter = _build_adapter()
        import json as _json

        async def _fail_analyze(image_url, user_prompt):
            return _json.dumps({"success": False, "error": "bad image"})

        monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fail_analyze)

        event = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            media_urls=["/cache/img.png"],
            media_types=[MessageType.PHOTO.value],
        )
        result = await adapter._enrich_event_with_media(event)
        assert "vision_analyze" in result.text
        assert "/cache/img.png" in result.text

    @pytest.mark.asyncio
    async def test_document_gets_context_note_no_vision_call(self, monkeypatch):
        adapter = _build_adapter()
        calls: List[str] = []

        async def _should_not_run(image_url, user_prompt):
            calls.append(image_url)
            return ""

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _should_not_run,
        )

        event = MessageEvent(
            text="summarize",
            message_type=MessageType.DOCUMENT,
            source=MagicMock(),
            media_urls=["/cache/deadbeef_report.pdf"],
            media_types=[MessageType.DOCUMENT.value],
        )
        result = await adapter._enrich_event_with_media(event)
        # Document path MUST NOT trigger vision analysis — matches the
        # gateway's canonical behavior, which only pre-analyzes images.
        assert calls == []
        assert "/cache/deadbeef_report.pdf" in result.text
        assert "document" in result.text.lower()
        assert "read_file" in result.text
        assert "summarize" in result.text

    @pytest.mark.asyncio
    async def test_audio_gets_transcription_hint_no_vision_call(self, monkeypatch):
        adapter = _build_adapter()

        async def _should_not_run(image_url, user_prompt):
            raise AssertionError("vision_analyze must not be called for audio")

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _should_not_run,
        )

        event = MessageEvent(
            text="",
            message_type=MessageType.AUDIO,
            source=MagicMock(),
            media_urls=["/cache/note.mp3"],
            media_types=[MessageType.AUDIO.value],
        )
        result = await adapter._enrich_event_with_media(event)
        assert "/cache/note.mp3" in result.text
        assert "transcription" in result.text.lower()

    @pytest.mark.asyncio
    async def test_empty_user_text_keeps_notes_only(self, monkeypatch):
        adapter = _build_adapter()

        async def _fake_vision(image_url, user_prompt):
            return self._success_vision_result("a PNG with text")

        monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fake_vision)

        event = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            media_urls=["/cache/img.png"],
            media_types=[MessageType.PHOTO.value],
        )
        result = await adapter._enrich_event_with_media(event)
        assert not result.text.endswith("\n\n")
        assert result.text.strip() != ""

    @pytest.mark.asyncio
    async def test_multiple_attachments_image_then_doc(self, monkeypatch):
        adapter = _build_adapter()
        image_calls: List[str] = []

        async def _fake_vision(image_url, user_prompt):
            image_calls.append(image_url)
            return self._success_vision_result(f"description of {image_url}")

        monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fake_vision)

        event = MessageEvent(
            text="look",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            media_urls=["/cache/a.png", "/cache/b.pdf"],
            media_types=[MessageType.PHOTO.value, MessageType.DOCUMENT.value],
        )
        result = await adapter._enrich_event_with_media(event)
        # Only the image triggers vision.
        assert image_calls == ["/cache/a.png"]
        # Both paths appear in the enriched text; user text trails.
        assert "/cache/a.png" in result.text
        assert "/cache/b.pdf" in result.text
        assert "description of /cache/a.png" in result.text
        assert "read_file" in result.text
        assert result.text.rstrip().endswith("look")


# ---------------------------------------------------------------------------
# _looks_like_command
# ---------------------------------------------------------------------------


class TestLooksLikeCommand:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("/help", True),
            ("/status", True),
            ("/stop now", True),
            ("/resume 42", True),
            ("  /help  ", True),  # leading whitespace tolerated
            ("/a", True),
            ("hello", False),
            ("", False),
            ("/", False),  # just a slash, no command token
            ("//", False),  # double slash rejected
            ("/var/log/foo", False),  # path, not command
            ("/123", True),  # numeric first char is allowed
            ("/_hidden", True),  # underscore is allowed per get_command()
        ],
    )
    def test_classification(self, text, expected):
        adapter = _build_adapter()
        assert adapter._looks_like_command(text) is expected


# ---------------------------------------------------------------------------
# _run_keepalive — periodic status:ack emission
# ---------------------------------------------------------------------------


class TestKeepalive:
    @pytest.mark.asyncio
    async def test_emits_status_ack_chunks_on_interval(self):
        adapter = _build_adapter(ack_keepalive_interval_s=1)
        stream = _fake_stream()

        task = asyncio.create_task(adapter._run_keepalive(stream))
        # Drive past one interval so at least one ack lands. We use a
        # real 1.1 s sleep rather than patching asyncio.sleep globally
        # because pytest-asyncio dispatches the loop itself.
        await asyncio.sleep(1.1)
        task.cancel()
        # The loop swallows CancelledError (returns None) by design so
        # ``_teardown_handles`` can ``gather`` across many keep-alives
        # without any of them surfacing as exceptions. Await normally
        # and assert that the task finished.
        await task
        assert task.done()

        # At least one ack must have landed before cancellation.
        assert stream.send.await_count >= 1
        chunk = stream.send.await_args_list[0].args[0]
        assert getattr(chunk, "status", None) == "ack"

    @pytest.mark.asyncio
    async def test_returns_cleanly_when_shutdown_event_set(self):
        adapter = _build_adapter(ack_keepalive_interval_s=1)
        stream = _fake_stream()

        task = asyncio.create_task(adapter._run_keepalive(stream))
        # Wait for the first sleep to land, then signal shutdown.
        await asyncio.sleep(0.05)
        adapter._shutdown_event.set()
        # Cancel to short-circuit the outer sleep and let the shutdown
        # branch run on the next iteration boundary. Cancellation is
        # caught and swallowed inside the loop (by design).
        task.cancel()
        await task
        assert task.done()

    @pytest.mark.asyncio
    async def test_send_failure_does_not_escalate(self, caplog):
        adapter = _build_adapter(ack_keepalive_interval_s=1)
        stream = _fake_stream()
        stream.send = AsyncMock(side_effect=RuntimeError("stream closed"))

        task = asyncio.create_task(adapter._run_keepalive(stream))
        await asyncio.sleep(1.1)
        # Task should have returned cleanly on the send failure, not
        # escalated — cancelling a finished task is a no-op.
        task.cancel()
        # Don't assert on .done()-ness here: if the sleep woke exactly at
        # cancel() rather than past the interval, the task may still be
        # pending. Either way, awaiting it must not re-raise the stream
        # send's RuntimeError.
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# _pump_deltas
# ---------------------------------------------------------------------------


class TestPumpDeltas:
    @pytest.mark.asyncio
    async def test_drains_queue_and_publishes_response_chunks(self):
        adapter = _build_adapter()
        stream = _fake_stream()
        queue: asyncio.Queue = asyncio.Queue()

        await queue.put("hello ")
        await queue.put("world")
        await queue.put(None)  # sentinel

        await adapter._pump_deltas(queue, stream)

        assert stream.send.await_count == 2
        first = stream.send.await_args_list[0].args[0]
        second = stream.send.await_args_list[1].args[0]
        # ResponseChunk is the conftest _FakeResponseChunk which records .text
        assert getattr(first, "text", None) == "hello "
        assert getattr(second, "text", None) == "world"

    @pytest.mark.asyncio
    async def test_exits_cleanly_on_send_failure(self):
        adapter = _build_adapter()
        stream = _fake_stream()
        stream.send = AsyncMock(side_effect=RuntimeError("stream closed"))
        queue: asyncio.Queue = asyncio.Queue()

        await queue.put("first")
        # Pump exits on first send failure rather than trying to drain
        # more — by that point the stream is dead.
        await adapter._pump_deltas(queue, stream)

        assert stream.send.await_count == 1


# ---------------------------------------------------------------------------
# send() — publishes via _active_streams
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_publishes_response_chunk_when_stream_is_registered(self):
        adapter = _build_adapter()
        stream = _fake_stream()
        adapter._active_streams[("alice", id(stream))] = stream

        result = await adapter.send(chat_id="alice", content="hi")

        assert result.success is True
        assert result.message_id
        chunk = stream.send.await_args.args[0]
        assert getattr(chunk, "text", None) == "hi"

    @pytest.mark.asyncio
    async def test_returns_failure_when_no_active_stream(self):
        adapter = _build_adapter()
        result = await adapter.send(chat_id="unknown", content="hi")
        assert result.success is False
        assert "no active NATS stream" in (result.error or "")

    @pytest.mark.asyncio
    async def test_returns_failure_when_stream_send_raises(self):
        adapter = _build_adapter()
        stream = _fake_stream()
        stream.send = AsyncMock(side_effect=RuntimeError("broken pipe"))
        adapter._active_streams[("alice", id(stream))] = stream

        result = await adapter.send(chat_id="alice", content="hi")
        assert result.success is False
        assert "broken pipe" in (result.error or "")


# ---------------------------------------------------------------------------
# _dispatch_command — routes through _message_handler
# ---------------------------------------------------------------------------


class TestDispatchCommand:
    @pytest.mark.asyncio
    async def test_sends_handler_response_as_response_chunk(self):
        adapter = _build_adapter()
        adapter._message_handler = AsyncMock(return_value="✅ session reset")
        stream = _fake_stream()

        event = MessageEvent(
            text="/new",
            message_type=MessageType.COMMAND,
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._dispatch_command(event, stream)

        adapter._message_handler.assert_awaited_once_with(event)
        chunk = stream.send.await_args.args[0]
        assert getattr(chunk, "text", None) == "✅ session reset"

    @pytest.mark.asyncio
    async def test_no_handler_sends_explicit_error(self):
        # An adapter used standalone (no GatewayRunner.set_message_handler)
        # must still respond — silent drop is a support nightmare.
        adapter = _build_adapter()
        adapter._message_handler = None
        stream = _fake_stream()
        event = MessageEvent(
            text="/help",
            message_type=MessageType.COMMAND,
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._dispatch_command(event, stream)

        chunk = stream.send.await_args.args[0]
        text = getattr(chunk, "text", "")
        assert "no message handler" in text.lower() or "not dispatched" in text.lower()

    @pytest.mark.asyncio
    async def test_handler_exception_surfaces_as_error_chunk(self):
        adapter = _build_adapter()
        adapter._message_handler = AsyncMock(side_effect=RuntimeError("boom"))
        stream = _fake_stream()
        event = MessageEvent(
            text="/stop",
            message_type=MessageType.COMMAND,
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._dispatch_command(event, stream)

        chunk = stream.send.await_args.args[0]
        assert "boom" in getattr(chunk, "text", "")

    @pytest.mark.asyncio
    async def test_empty_handler_response_emits_nothing(self):
        # Commands like /stop or /ack sometimes legitimately return
        # None (the gateway handled it out-of-band). Don't publish an
        # empty chunk in that case — callers expect silence = no output.
        adapter = _build_adapter()
        adapter._message_handler = AsyncMock(return_value=None)
        stream = _fake_stream()
        event = MessageEvent(
            text="/stop",
            message_type=MessageType.COMMAND,
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._dispatch_command(event, stream)

        stream.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# _on_prompt — end-to-end happy paths & cleanup invariants
# ---------------------------------------------------------------------------


class TestOnPromptIntegration:
    @pytest.mark.asyncio
    async def test_text_prompt_dispatches_to_text_path(self, monkeypatch):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "hello agent"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"hello agent","x-session":"alice"}')

        text_prompt_calls: list = []

        async def _fake_run_text_prompt(event, s, chat_id):
            text_prompt_calls.append((event, s, chat_id))

        monkeypatch.setattr(adapter, "_run_text_prompt", _fake_run_text_prompt)

        await adapter._on_prompt(envelope, stream)

        assert len(text_prompt_calls) == 1
        event, passed_stream, chat_id = text_prompt_calls[0]
        assert isinstance(event, MessageEvent)
        assert event.text == "hello agent"
        assert event.source.chat_id == "alice"
        assert event.source.platform is Platform.NATS
        assert event.source.chat_type == "dm"
        assert passed_stream is stream
        assert chat_id == "alice"

    @pytest.mark.asyncio
    async def test_slash_command_dispatches_to_command_path(self, monkeypatch):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "/help"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"/help"}')

        dispatched: list = []

        async def _fake_dispatch(event, s):
            dispatched.append((event, s))

        async def _fake_text(*args, **kwargs):
            raise AssertionError("text path must not run for slash commands")

        monkeypatch.setattr(adapter, "_dispatch_command", _fake_dispatch)
        monkeypatch.setattr(adapter, "_run_text_prompt", _fake_text)

        await adapter._on_prompt(envelope, stream)

        assert len(dispatched) == 1
        event, _ = dispatched[0]
        assert event.message_type is MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_command_text_is_lstripped_for_gateway_dispatch(self, monkeypatch):
        # Regression guard: ``_looks_like_command`` tolerates leading
        # whitespace, but ``MessageEvent.is_command`` / ``get_command()``
        # in base.py:732 require literal ``text.startswith("/")``. If we
        # pass the raw prompt with leading whitespace, the gateway's
        # command registry misses the dispatch and the caller sees the
        # text prompt path instead of the command response.
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "  /help"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"  /help"}')

        dispatched: list = []

        async def _fake_dispatch(event, s):
            dispatched.append(event)

        async def _fake_text(*args, **kwargs):
            raise AssertionError("text path must not run for whitespace-prefixed command")

        monkeypatch.setattr(adapter, "_dispatch_command", _fake_dispatch)
        monkeypatch.setattr(adapter, "_run_text_prompt", _fake_text)

        await adapter._on_prompt(envelope, stream)

        assert len(dispatched) == 1
        event = dispatched[0]
        # event.text must start with "/" — otherwise MessageEvent.get_command
        # returns None and the gateway's command registry misses the dispatch.
        assert event.text.startswith("/")
        assert event.get_command() == "help"

    @pytest.mark.asyncio
    async def test_registers_stream_and_cleans_up_on_success(self, monkeypatch):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"hi","x-session":"bob"}')

        observed: dict = {}

        async def _fake_run_text_prompt(event, s, chat_id):
            # Mid-handler: the stream must be registered so tool outputs
            # can publish onto it.
            observed["active_streams_during_run"] = dict(adapter._active_streams)

        monkeypatch.setattr(adapter, "_run_text_prompt", _fake_run_text_prompt)

        await adapter._on_prompt(envelope, stream)

        # T5.0 — the registry is compound-keyed (chat_id, id(stream)) so
        # overlapping x-session prompts can coexist. The contextvar is the
        # race-safe primary lookup; this dict is the diagnostic fallback.
        assert observed["active_streams_during_run"] == {("bob", id(stream)): stream}
        # After the handler returns, the stream must be gone so a later
        # send() to the same chat_id fails fast.
        assert adapter._active_streams == {}
        # Task tracking leaves no leaks either.
        assert adapter._in_flight_handlers == set()

    @pytest.mark.asyncio
    async def test_falls_back_to_default_session_when_x_session_missing(
        self, monkeypatch
    ):
        adapter = _build_adapter(session_default="fallback")
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = None
        # Plain-text shorthand carries no x-session → fallback.
        stream = _fake_stream(b"hi")

        captured: list = []

        async def _fake_run(event, s, chat_id):
            captured.append(chat_id)

        monkeypatch.setattr(adapter, "_run_text_prompt", _fake_run)

        await adapter._on_prompt(envelope, stream)
        assert captured == ["fallback"]

    @pytest.mark.asyncio
    async def test_cleans_up_when_run_text_prompt_raises(self, monkeypatch):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"hi","x-session":"carol"}')

        async def _boom(event, s, chat_id):
            raise RuntimeError("agent exploded")

        monkeypatch.setattr(adapter, "_run_text_prompt", _boom)

        with pytest.raises(RuntimeError, match="agent exploded"):
            await adapter._on_prompt(envelope, stream)

        # The SDK converts our exception into an error frame — but we
        # must leave NO stream / task leaks so the next prompt starts
        # clean.
        assert adapter._active_streams == {}
        assert adapter._in_flight_handlers == set()

    @pytest.mark.asyncio
    async def test_current_task_is_tracked_during_handler(self, monkeypatch):
        adapter = _build_adapter()
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"hi"}')

        tracked: list = []

        async def _inspect(event, s, chat_id):
            tracked.append(set(adapter._in_flight_handlers))

        monkeypatch.setattr(adapter, "_run_text_prompt", _inspect)

        await adapter._on_prompt(envelope, stream)

        assert len(tracked) == 1
        assert len(tracked[0]) == 1  # the current handler task
        assert adapter._in_flight_handlers == set()

    @pytest.mark.asyncio
    async def test_keepalive_task_cancelled_after_handler_returns(
        self, monkeypatch
    ):
        # If the keep-alive leaks, it keeps publishing after the SDK's
        # terminator fires — spec violation. Verify it's cancelled in
        # the finally block of _on_prompt.
        adapter = _build_adapter(ack_keepalive_interval_s=1)
        envelope = MagicMock()
        envelope.prompt = "hi"
        envelope.attachments = None
        stream = _fake_stream(b'{"prompt":"hi"}')

        captured_tasks: list = []

        real_create = asyncio.create_task

        def _spy(coro, *args, **kwargs):
            task = real_create(coro, *args, **kwargs)
            name = kwargs.get("name") or ""
            if name.startswith("nats-keepalive"):
                captured_tasks.append(task)
            return task

        async def _fast_run(event, s, chat_id):
            # Yield briefly so the keep-alive task has a chance to start.
            await asyncio.sleep(0)

        monkeypatch.setattr(adapter, "_run_text_prompt", _fast_run)
        monkeypatch.setattr("asyncio.create_task", _spy)

        await adapter._on_prompt(envelope, stream)

        assert len(captured_tasks) == 1
        assert captured_tasks[0].done()


# ---------------------------------------------------------------------------
# _final_response_text
# ---------------------------------------------------------------------------


class TestFinalResponseText:
    def test_reads_final_response_from_dict(self):
        assert _final_response_text({"final_response": "ok"}) == "ok"

    def test_accepts_bare_string(self):
        assert _final_response_text("bare") == "bare"

    def test_none_folds_to_empty_string(self):
        assert _final_response_text(None) == ""

    def test_missing_key_folds_to_empty_string(self):
        assert _final_response_text({"other": "value"}) == ""


# ---------------------------------------------------------------------------
# Streaming fallback — _run_text_prompt delivers final text when no deltas
# streamed (e.g. streaming disabled, tool-only turn)
# ---------------------------------------------------------------------------


class TestRunTextPromptFallback:
    @pytest.mark.asyncio
    async def test_final_text_delivered_when_no_deltas_streamed(
        self, monkeypatch
    ):
        # When ``stream_delta_callback`` is never called, the final text
        # still needs to land on the caller. Verify the fallback send.
        adapter = _build_adapter()
        stream = _fake_stream()

        def _fake_run_agent_sync(event, chat_id, cb, loop):
            # Don't call ``cb`` — simulate a non-streaming run.
            return {"final_response": "final answer"}

        monkeypatch.setattr(adapter, "_run_agent_sync", _fake_run_agent_sync)

        event = MessageEvent(
            text="question",
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._run_text_prompt(event, stream, "alice")

        # Exactly one ResponseChunk for the final answer.
        assert stream.send.await_count == 1
        chunk = stream.send.await_args.args[0]
        assert getattr(chunk, "text", None) == "final answer"

    @pytest.mark.asyncio
    async def test_final_text_skipped_when_deltas_already_streamed(
        self, monkeypatch
    ):
        # If the pump saw deltas, the final text is already on the wire
        # — don't duplicate it.
        adapter = _build_adapter()
        stream = _fake_stream()

        def _fake_run_agent_sync(event, chat_id, cb, loop):
            # Simulate one streamed delta.
            cb("streamed ")
            cb("answer")
            return {"final_response": "streamed answer"}

        monkeypatch.setattr(adapter, "_run_agent_sync", _fake_run_agent_sync)

        event = MessageEvent(
            text="question",
            source=adapter.build_source(chat_id="alice"),
        )

        await adapter._run_text_prompt(event, stream, "alice")

        # Two ResponseChunks — one per delta; NO extra final-text chunk.
        assert stream.send.await_count == 2
        chunks = [c.args[0] for c in stream.send.await_args_list]
        texts = [getattr(c, "text", None) for c in chunks]
        assert texts == ["streamed ", "answer"]
