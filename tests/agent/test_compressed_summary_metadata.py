"""Regression tests for the compressed-summary metadata flag (#38389).

The compressor marks summary messages with ``COMPRESSED_SUMMARY_METADATA_KEY``
so frontends (CLI, Desktop, gateway, TUI) can distinguish them from real
assistant/user messages without content-prefix heuristics.

Two invariants:
1. The flag is present on exactly the summary-bearing message after compress()
   (standalone insertion AND merge-into-tail).
2. The key is underscore-prefixed so the chat-completions wire sanitizer
   strips it — strict gateways (Fireworks, Mistral, Moonshot/Kimi,
   opencode-go) reject unknown message keys with "Extra inputs are not
   permitted", poisoning the session.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)


def _make_compressor():
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=8000
    ):
        return ContextCompressor(
            model="test-model", quiet_mode=True, config_context_length=8000
        )


def _make_messages(n_turns=30):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 400})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 400})
    return msgs


def _compress(cc, msgs):
    resp = MagicMock()
    resp.choices[0].message.content = "## Active Task\nstuff"
    with patch("agent.context_compressor.call_llm", return_value=resp):
        return cc.compress(msgs, current_tokens=100_000, force=True)


class TestMetadataFlagSet:
    def test_exactly_one_flagged_message_after_compress(self):
        cc = _make_compressor()
        out = _compress(cc, _make_messages())
        flagged = [
            m for m in out
            if isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        assert len(flagged) == 1
        # The flagged message is the one carrying the compaction handoff.
        assert "[CONTEXT COMPACTION" in flagged[0]["content"]

    def test_helper_detects_flag(self):
        assert ContextCompressor._has_compressed_summary_metadata(
            {COMPRESSED_SUMMARY_METADATA_KEY: True}
        )
        assert not ContextCompressor._has_compressed_summary_metadata(
            {"role": "assistant", "content": "hi"}
        )
        assert not ContextCompressor._has_compressed_summary_metadata("not a dict")
        assert not ContextCompressor._has_compressed_summary_metadata(None)


class TestMetadataFlagNeverReachesWire:
    def test_key_is_underscore_prefixed(self):
        """The wire sanitizers strip every top-level message key starting
        with '_'. A bare key would reach strict gateways (Fireworks etc.)
        and 400 with 'Extra inputs are not permitted'."""
        assert COMPRESSED_SUMMARY_METADATA_KEY.startswith("_")

    def test_chat_completions_transport_strips_flag(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        cc = _make_compressor()
        out = _compress(cc, _make_messages())
        wire = ChatCompletionsTransport().convert_messages(out, model="some-model")
        assert not any(
            isinstance(m, dict) and COMPRESSED_SUMMARY_METADATA_KEY in m
            for m in wire
        )
        # Sanitization must not destroy the in-process flag on the originals.
        assert any(
            isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
            for m in out
        )
