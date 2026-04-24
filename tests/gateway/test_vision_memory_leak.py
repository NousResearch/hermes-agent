"""Tests for _enrich_message_with_vision — regression for #5719.

The auxiliary vision LLM can echo system-prompt Honcho memory back into
its analysis output. When that echo reaches the user as the enriched
image description, recalled memory context (personal facts, dialectic
output) surfaces into a user-visible message.

The boundary fix in gateway/run.py strips both <memory-context>...</memory-context>
fenced blocks AND any "## Honcho Context" section from vision descriptions
before they're embedded into the enriched user message.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def gateway_runner():
    """Minimal GatewayRunner stub with just the method under test bound."""
    from gateway.run import GatewayRunner

    class _Stub:
        _enrich_message_with_vision = GatewayRunner._enrich_message_with_vision

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

    def test_honcho_context_header_stripped(self, gateway_runner):
        """'## Honcho Context' section and everything after is removed."""
        leaked = (
            "A photograph of a sunset.\n\n"
            "## Honcho Context\n"
            "User prefers concise answers, works at Plastic Labs,\n"
            "uses OPSEC pseudonyms.\n"
        )
        fake_result = json.dumps({"success": True, "analysis": leaked})
        with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=fake_result)):
            out = _run(gateway_runner._enrich_message_with_vision("caption", ["/tmp/img.jpg"]))
        assert "sunset" in out
        assert "Honcho Context" not in out
        assert "Plastic Labs" not in out
        assert "OPSEC" not in out

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

    def test_both_leak_patterns_together_stripped(self, gateway_runner):
        """A vision output containing both leak shapes is fully scrubbed."""
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n"
            "fenced leak\n"
            "</memory-context>\n"
            "A photograph of a dog.\n\n"
            "## Honcho Context\n"
            "header leak\n"
        )
        fake_result = json.dumps({"success": True, "analysis": leaked})
        with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=fake_result)):
            out = _run(gateway_runner._enrich_message_with_vision("caption", ["/tmp/img.jpg"]))
        assert "photograph of a dog" in out
        assert "fenced leak" not in out
        assert "header leak" not in out
        assert "Honcho Context" not in out
        assert "<memory-context>" not in out
