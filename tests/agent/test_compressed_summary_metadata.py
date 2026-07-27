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
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
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
        assert COMPRESSED_SUMMARY_HAS_USER_TURN_KEY.startswith("_")

    def test_chat_completions_transport_strips_flag(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        cc = _make_compressor()
        out = _compress(cc, _make_messages())
        wire = ChatCompletionsTransport().convert_messages(out, model="some-model")
        assert not any(
            isinstance(m, dict)
            and (
                COMPRESSED_SUMMARY_METADATA_KEY in m
                or COMPRESSED_SUMMARY_HAS_USER_TURN_KEY in m
            )
            for m in wire
        )
        # Sanitization must not destroy the in-process flag on the originals.
        assert any(
            isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
            for m in out
        )


class TestClassifySummaryContent:
    """classify_summary_content distinguishes standalone handoffs from
    merge-into-tail messages so wire consumers (ACP replay) can flag them
    differently — collapsing a merged message would hide the preserved
    tail content that precedes the summary."""

    def test_standalone_summary(self):
        from agent.context_compressor import SUMMARY_PREFIX

        content = SUMMARY_PREFIX + "\n## Active Task\nstuff"
        assert ContextCompressor.classify_summary_content(content) == "standalone"
        assert ContextCompressor._is_context_summary_content(content) is True

    def test_legacy_and_historical_prefixes_are_standalone(self):
        from agent.context_compressor import (
            LEGACY_SUMMARY_PREFIX,
            _HISTORICAL_SUMMARY_PREFIXES,
        )

        assert ContextCompressor.classify_summary_content(
            LEGACY_SUMMARY_PREFIX + " body"
        ) == "standalone"
        for prefix in _HISTORICAL_SUMMARY_PREFIXES:
            assert ContextCompressor.classify_summary_content(
                prefix + " body"
            ) == "standalone"

    def test_merged_tail_summary(self):
        from agent.context_compressor import (
            SUMMARY_PREFIX,
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            _SUMMARY_END_MARKER,
        )

        merged = (
            _MERGED_PRIOR_CONTEXT_HEADER + "\n"
            "old tail content\n\n"
            + _MERGED_SUMMARY_DELIMITER + "\n\n"
            + SUMMARY_PREFIX + "\nBODY\n\n"
            + _SUMMARY_END_MARKER
        )
        assert ContextCompressor.classify_summary_content(merged) == "merged"
        assert ContextCompressor._is_context_summary_content(merged) is True

    def test_plain_messages_classify_none(self):
        assert ContextCompressor.classify_summary_content("just a question") is None
        assert ContextCompressor.classify_summary_content("") is None
        assert ContextCompressor.classify_summary_content(None) is None

    def test_delimiter_without_summary_prefix_is_none(self):
        """A message merely quoting the merged delimiter (e.g. a user pasting
        logs) is not a summary unless a handoff prefix follows it."""
        from agent.context_compressor import _MERGED_SUMMARY_DELIMITER

        content = "look at this:\n" + _MERGED_SUMMARY_DELIMITER + "\nnot a summary"
        assert ContextCompressor.classify_summary_content(content) is None
        assert ContextCompressor._is_context_summary_content(content) is False


class TestClassifyAgreesWithPredicatesOnLiveEmissions:
    """Behavior contract: classify_summary_content must agree with the
    boolean summary predicates (``_is_context_summary_content`` and the
    module-level ``is_compaction_summary_message``) on every handoff shape
    the CURRENT compressor actually emits — not just hand-built fixtures.

    If the emission format drifts (prefix rewording, new merge framing),
    these tests fail on the real output rather than on a stale snapshot,
    signalling that ACP replay flagging and the internal summary detectors
    have diverged.
    """

    @staticmethod
    def _live_compress(msgs):
        # Match the emission-probe compressor shape (wide context, minimal
        # protection) so the transcript geometry — not protection budgets —
        # decides the merge-vs-standalone path deterministically.
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=100_000,
        ):
            cc = ContextCompressor(
                model="test-model",
                threshold_percent=0.85,
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )
        out = _compress(cc, msgs)
        flagged = [
            m for m in out
            if isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        return flagged

    @staticmethod
    def _assert_agreement(message):
        from agent.context_compressor import is_compaction_summary_message

        kind = ContextCompressor.classify_summary_content(message.get("content"))
        detected = ContextCompressor._is_context_summary_content(
            message.get("content")
        )
        assert (kind is not None) == detected
        # is_compaction_summary_message also honors the metadata flag, so it
        # must detect every message classify flags — and on a flag-stripped
        # copy (the DB-reload shape) content classification alone must carry.
        assert is_compaction_summary_message(message) is True
        stripped = {
            k: v for k, v in message.items()
            if k != COMPRESSED_SUMMARY_METADATA_KEY
        }
        assert is_compaction_summary_message(stripped) == (kind is not None)
        return kind

    def test_merged_emission_classifies_merged_and_predicates_agree(self):
        """Alternating transcripts take the merge-into-tail path on current
        main — the emitted handoff must classify 'merged'."""
        flagged = self._live_compress(_make_messages())
        assert len(flagged) == 1
        kind = self._assert_agreement(flagged[0])
        assert kind == "merged"

    def test_standalone_emission_classifies_standalone_and_predicates_agree(self):
        """A degenerate all-user transcript forces the standalone-insertion
        path — the emitted handoff must classify 'standalone'."""
        msgs = [{"role": "system", "content": "sys"}]
        msgs.extend(
            {"role": "user", "content": f"user only {i} " + "x" * 400}
            for i in range(60)
        )
        flagged = self._live_compress(msgs)
        assert len(flagged) == 1
        kind = self._assert_agreement(flagged[0])
        assert kind == "standalone"

    def test_non_summary_messages_agree_on_none(self):
        from agent.context_compressor import is_compaction_summary_message

        for msg in (
            {"role": "user", "content": "plain question"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
        ):
            assert ContextCompressor.classify_summary_content(msg["content"]) is None
            assert ContextCompressor._is_context_summary_content(msg["content"]) is False
            assert is_compaction_summary_message(msg) is False
