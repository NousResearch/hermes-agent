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
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run import GatewayRunner


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
