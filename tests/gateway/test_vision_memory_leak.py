"""Tests for _enrich_message_with_vision — regression for #5719.

The auxiliary vision LLM can echo system-prompt memory-context back into
its analysis output.  The boundary fix in gateway/run.py runs the generic
sanitize_context helper over the description so the fenced wrapper and
its system-note are removed before the description reaches the user.

Plugin-specific header cleanup (e.g. "## Honcho Context") belongs at the
provider boundary, not in this shared gateway path.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def gateway_runner():
    """Minimal GatewayRunner stub with the methods exercised by these
    tests bound directly off the real class.

    Includes the auto-retry helpers introduced by #28972 because
    ``_enrich_message_with_vision`` now delegates to them — without
    these bindings the stub would AttributeError on the very first
    image and the sanitize-context regression tests below would
    silently exercise the wrong code path.
    """
    from gateway.run import GatewayRunner

    class _Stub:
        _enrich_message_with_vision = GatewayRunner._enrich_message_with_vision
        _vision_analyze_with_auto_retry = GatewayRunner._vision_analyze_with_auto_retry
        _vision_auto_retry_count = staticmethod(GatewayRunner._vision_auto_retry_count)
        _vision_failure_is_retryable = staticmethod(GatewayRunner._vision_failure_is_retryable)
        _VISION_AUTO_RETRY_COUNT_DEFAULT = GatewayRunner._VISION_AUTO_RETRY_COUNT_DEFAULT
        _VISION_AUTO_RETRY_INITIAL_BACKOFF_S = GatewayRunner._VISION_AUTO_RETRY_INITIAL_BACKOFF_S
        _VISION_AUTO_RETRY_MAX_BACKOFF_S = GatewayRunner._VISION_AUTO_RETRY_MAX_BACKOFF_S
        _VISION_NONRETRYABLE_HINTS = GatewayRunner._VISION_NONRETRYABLE_HINTS

    return _Stub()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.new_event_loop().run_until_complete(coro)


class TestEnrichMessageWithVision:
    def test_clean_description_passes_through(self, gateway_runner):
        """Vision output without leaked memory is embedded unchanged."""
        fake_result = json.dumps({
            "success": True,
            "analysis": "A photograph of a sunset over the ocean.",
        })
        with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=fake_result)):
            out = _run(gateway_runner._enrich_message_with_vision("caption", ["/tmp/img.jpg"]))
        assert "sunset over the ocean" in out

    def test_memory_context_fence_stripped(self, gateway_runner):
        """<memory-context>...</memory-context> fenced block is scrubbed."""
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n\n"
            "User details and preferences here.\n"
            "</memory-context>\n"
            "A photograph of a cat."
        )
        fake_result = json.dumps({"success": True, "analysis": leaked})
        with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=fake_result)):
            out = _run(gateway_runner._enrich_message_with_vision("caption", ["/tmp/img.jpg"]))
        assert "photograph of a cat" in out
        assert "<memory-context>" not in out
        assert "User details and preferences" not in out
        assert "System note" not in out

    def test_fenced_leak_stripped_plugin_header_preserved(self, gateway_runner):
        """The fenced wrapper is stripped; plugin-specific text outside the
        fence (e.g. a "## Honcho Context" header) is left to the plugin layer.
        Gateway core stays plugin-agnostic."""
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n"
            "fenced leak\n"
            "</memory-context>\n"
            "A photograph of a dog."
        )
        fake_result = json.dumps({"success": True, "analysis": leaked})
        with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=fake_result)):
            out = _run(gateway_runner._enrich_message_with_vision("caption", ["/tmp/img.jpg"]))
        assert "photograph of a dog" in out
        assert "fenced leak" not in out
        assert "<memory-context>" not in out
