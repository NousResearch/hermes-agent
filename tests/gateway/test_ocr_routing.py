"""Gateway OCR routing tests for gateway/run.py.

Covers the caption-intent trigger that splits inbound image messages between the
verbatim OCR path (ocr_image) and the describe path (vision_analyze):

- _caption_requests_ocr: explicit OCR markers and EMPTY captions → True;
  describe-style captions → False.
- _enrich_message_with_ocr: invokes the ocr_image tool (mocked async_call_llm at
  the tool boundary) and prepends the verbatim transcription to the message.
- Routing parity check: "transcribe" routes to OCR; "what's in this picture"
  does not (stays on the describe path).

These exercise the library surface of GatewayRunner without standing up an
adapter, a live agent, or the message guard.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import Platform, SessionSource


def _runner():
    """Bare GatewayRunner instance (no __init__) for unit-testing methods."""
    return GatewayRunner.__new__(GatewayRunner)


_EMPTY_PLACEHOLDER = "(The user sent a message with no text content)"


# ---------------------------------------------------------------------------
# _caption_requests_ocr — intent detection
# ---------------------------------------------------------------------------


class TestCaptionRequestsOcr:
    @pytest.mark.parametrize("caption", [
        "transcribe",
        "Transcribe this please",
        "read this",
        "read the document",
        "what does this say",
        "what does it say?",
        "OCR this",
        "can you ocr it",
    ])
    def test_explicit_ocr_intent_true(self, caption):
        assert GatewayRunner._caption_requests_ocr(caption) is True

    @pytest.mark.parametrize("caption", [
        "",
        "   ",
        None,
        _EMPTY_PLACEHOLDER,
    ])
    def test_empty_caption_defaults_to_ocr(self, caption):
        assert GatewayRunner._caption_requests_ocr(caption) is True

    @pytest.mark.parametrize("caption", [
        "what's in this picture",
        "describe this image",
        "who is in this photo",
        "is this a cat or a dog",
        "make this funnier",
    ])
    def test_describe_intent_false(self, caption):
        assert GatewayRunner._caption_requests_ocr(caption) is False


# ---------------------------------------------------------------------------
# _enrich_message_with_ocr — prepends verbatim transcription
# ---------------------------------------------------------------------------


class TestEnrichMessageWithOcr:
    @pytest.mark.asyncio
    async def test_prepends_transcription(self):
        runner = _runner()
        ok = json.dumps({"success": True, "text": "HELLO WORLD", "pages": 1,
                         "model": "qwen/qwen3-vl-32b-instruct"})
        with patch(
            "tools.vision_tools.ocr_image_tool",
            new_callable=AsyncMock,
            return_value=ok,
        ) as mock_tool:
            out = await runner._enrich_message_with_ocr(
                "transcribe", ["/tmp/img.png"]
            )
        mock_tool.assert_awaited_once()
        assert mock_tool.await_args.kwargs["image_url"] == "/tmp/img.png"
        assert "HELLO WORLD" in out
        # Original caption is preserved after the prepended transcription.
        assert out.rstrip().endswith("transcribe")

    @pytest.mark.asyncio
    async def test_empty_caption_yields_only_transcription(self):
        runner = _runner()
        ok = json.dumps({"success": True, "text": "JUST TEXT", "pages": 1})
        with patch(
            "tools.vision_tools.ocr_image_tool",
            new_callable=AsyncMock,
            return_value=ok,
        ):
            out = await runner._enrich_message_with_ocr(
                _EMPTY_PLACEHOLDER, ["/tmp/img.png"]
            )
        assert "JUST TEXT" in out
        assert _EMPTY_PLACEHOLDER not in out

    @pytest.mark.asyncio
    async def test_failure_falls_back_to_recovery_note(self):
        runner = _runner()
        bad = json.dumps({"success": False, "error": "boom", "text": ""})
        with patch(
            "tools.vision_tools.ocr_image_tool",
            new_callable=AsyncMock,
            return_value=bad,
        ):
            out = await runner._enrich_message_with_ocr(
                "transcribe", ["/tmp/img.png"]
            )
        # Recovery note references the tool + path so the agent can retry.
        assert "ocr_image" in out
        assert "/tmp/img.png" in out

    @pytest.mark.asyncio
    async def test_exception_is_caught(self):
        runner = _runner()
        with patch(
            "tools.vision_tools.ocr_image_tool",
            new_callable=AsyncMock,
            side_effect=RuntimeError("kaboom"),
        ):
            out = await runner._enrich_message_with_ocr(
                "transcribe", ["/tmp/img.png"]
            )
        assert "ocr_image" in out
        assert "/tmp/img.png" in out


# ---------------------------------------------------------------------------
# Routing parity: which enrich path a caption selects
# ---------------------------------------------------------------------------


class TestRoutingParity:
    @pytest.mark.asyncio
    async def test_transcribe_goes_to_ocr_not_vision(self):
        runner = _runner()
        assert GatewayRunner._caption_requests_ocr("transcribe") is True

        ok = json.dumps({"success": True, "text": "T", "pages": 1})
        with (
            patch(
                "tools.vision_tools.ocr_image_tool",
                new_callable=AsyncMock,
                return_value=ok,
            ) as mock_ocr,
        ):
            await runner._enrich_message_with_ocr("transcribe", ["/tmp/a.png"])
        mock_ocr.assert_awaited_once()

    def test_describe_caption_stays_on_vision_path(self):
        # The describe caption must NOT select the OCR branch; the dispatch
        # site only calls _enrich_message_with_ocr when this returns True.
        assert (
            GatewayRunner._caption_requests_ocr("what's in this picture")
            is False
        )


# ---------------------------------------------------------------------------
# Dispatch-branch precedence (finding 1): OCR intent overrides native vision
# ---------------------------------------------------------------------------


def _photo_event(caption: str, paths):
    src = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="u1",
        user_name="alice",
    )
    return MessageEvent(
        text=caption,
        message_type=MessageType.PHOTO,
        source=src,
        media_urls=list(paths),
        media_types=["image/png"] * len(paths),
    ), src


def _wire_runner_for_prepare(runner):
    """Stub out the runner internals the image-routing branch touches so we can
    drive _prepare_inbound_message_text without standing up an adapter."""
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._session_key_for_source = MagicMock(return_value="telegram:123")
    runner._consume_pending_native_image_paths = MagicMock(return_value=[])
    # Sentinel return values let us assert which enrich path ran.
    runner._enrich_message_with_ocr = AsyncMock(return_value="OCR_RAN")
    runner._enrich_message_with_vision = AsyncMock(return_value="VISION_RAN")
    runner._enrich_message_with_transcription = AsyncMock()


class TestNativeModeOcrOverride:
    @pytest.mark.asyncio
    async def test_ocr_intent_overrides_native_mode(self):
        # HIGH finding 1: even when the model supports native vision, an explicit
        # OCR-intent caption must route to ocr_image (NOT be buffered for inline
        # native attachment, NOT go to vision_analyze).
        runner = _runner()
        _wire_runner_for_prepare(runner)
        event, src = _photo_event("transcribe this", ["/tmp/a.png"])

        with patch.object(runner, "_decide_image_input_mode",
                          return_value="native"):
            out = await runner._prepare_inbound_message_text(
                event=event, source=src, history=[],
            )

        runner._enrich_message_with_ocr.assert_awaited_once()
        runner._enrich_message_with_vision.assert_not_awaited()
        # No native buffer was created for this session.
        assert not getattr(
            runner, "_pending_native_image_paths_by_session", {}
        ).get("telegram:123")
        assert "OCR_RAN" in out

    @pytest.mark.asyncio
    async def test_empty_caption_overrides_native_mode(self):
        runner = _runner()
        _wire_runner_for_prepare(runner)
        event, src = _photo_event("", ["/tmp/a.png"])

        with patch.object(runner, "_decide_image_input_mode",
                          return_value="native"):
            await runner._prepare_inbound_message_text(
                event=event, source=src, history=[],
            )
        runner._enrich_message_with_ocr.assert_awaited_once()
        runner._enrich_message_with_vision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_describe_caption_uses_native_buffer(self):
        # Parity: a describe caption on a native model must still buffer for
        # inline attachment (NOT OCR, NOT vision_analyze pre-run).
        runner = _runner()
        _wire_runner_for_prepare(runner)
        event, src = _photo_event("what's in this picture", ["/tmp/a.png"])

        with patch.object(runner, "_decide_image_input_mode",
                          return_value="native"):
            await runner._prepare_inbound_message_text(
                event=event, source=src, history=[],
            )
        runner._enrich_message_with_ocr.assert_not_awaited()
        runner._enrich_message_with_vision.assert_not_awaited()
        assert runner._pending_native_image_paths_by_session["telegram:123"] == [
            "/tmp/a.png"
        ]

    @pytest.mark.asyncio
    async def test_ocr_intent_text_mode_still_ocr(self):
        # Regression: OCR intent in text mode keeps routing to OCR.
        runner = _runner()
        _wire_runner_for_prepare(runner)
        event, src = _photo_event("please ocr this", ["/tmp/a.png"])

        with patch.object(runner, "_decide_image_input_mode",
                          return_value="text"):
            await runner._prepare_inbound_message_text(
                event=event, source=src, history=[],
            )
        runner._enrich_message_with_ocr.assert_awaited_once()
        runner._enrich_message_with_vision.assert_not_awaited()
