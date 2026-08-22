"""Regression tests for the gateway fallback secret-redaction patterns.

Issue #81073: ``_GATEWAY_SECRET_PATTERNS`` anchored with ``\\b``, which under
Python 3's default Unicode semantics treats CJK/fullwidth letters as word
characters. A token glued to a CJK character (``xx中sk-...``) is therefore NOT
at a ``\\b`` boundary, so the fallback pattern pass silently left it
unredacted. The anchors are now ASCII word lookarounds
(``(?<![A-Za-z0-9_])`` / ``(?![A-Za-z0-9_])``) so Unicode-glued tokens are
caught the same as ASCII-neighbored ones.
"""

import pytest

from gateway.run import _GATEWAY_SECRET_PATTERNS

# Each entry is (text with token, expected-absence substring).
_FAKE_TOKENS = [
    ("sk-" + "a" * 20, "sk-"),
    ("ghp_" + "b" * 20, "ghp_"),
    ("xapp-1-" + "c" * 20, "xapp-"),
    ("xoxb-" + "d" * 20, "xoxb-"),
    ("hf_" + "e" * 20, "hf_"),
    ("glpat-" + "f" * 20, "glpat-"),
    ("Bearer " + "g" * 20, "Bearer g"),
]


def _pattern_pass(text: str) -> str:
    """Run exactly the fallback pass ``_redact_gateway_user_facing_secrets``
    applies after the Tirith-grade redactor."""
    redacted = text
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]",
            redacted,
        )
    return redacted


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_cjk_glued_tokens_are_redacted(token, prefix):
    """A token glued directly to a CJK character must be redacted (#81073)."""
    redacted = _pattern_pass("xx中" + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"CJK-glued token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_fullwidth_digit_glued_tokens_are_redacted(token, prefix):
    """A token glued to a fullwidth digit (Unicode ``\\w``) must be redacted."""
    redacted = _pattern_pass("xx１" + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"fullwidth-glued token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_ascii_neighbor_tokens_still_redacted(token, prefix):
    """ASCII-neighbored tokens keep redacting (no regression)."""
    redacted = _pattern_pass("xx " + token)
    assert "[REDACTED]" in redacted
    assert prefix not in redacted, f"ASCII-neighbor token leaked: {redacted!r}"


@pytest.mark.parametrize("token,prefix", _FAKE_TOKENS)
def test_embedded_in_ascii_word_not_matched(token, prefix):
    """A token embedded inside an ASCII identifier must NOT match — same
    boundary semantics as ``\\b``."""
    redacted = _pattern_pass("xx" + token + "yy")
    assert "[REDACTED]" not in redacted, (
        f"embedded token over-matched: {redacted!r}"
    )


def test_bearer_prefix_is_preserved():
    """The Bearer prefix survives redaction (group(1) preserved)."""
    redacted = _pattern_pass("Authorization: Bearer " + "z" * 20)
    assert "Bearer " in redacted
    assert "Bearer zzz" not in redacted


# Unicode case-fold characters that Python's IGNORECASE maps onto ASCII
# letters: U+0130 İ, U+0131 ı, U+017F ſ, U+212A K. With the blanket ``(?i)``
# these folded into the ASCII word lookarounds, so a token glued beside one
# was NOT redacted (the boundary assertion failed). The fix scopes
# case-insensitivity to the Bearer literal via ``(?i:Bearer)`` (#81073).
_CASE_FOLD_NEIGHBORS = ("İ", "ı", "ſ", "K")


@pytest.mark.parametrize("neighbor", _CASE_FOLD_NEIGHBORS)
def test_bearer_redacted_beside_unicode_case_fold_left(neighbor):
    """A bare Bearer token glued after a case-fold character must redact."""
    redacted = _pattern_pass("xx" + neighbor + "Bearer " + "g" * 20)
    assert "[REDACTED]" in redacted, f"left {neighbor!r} leaked: {redacted!r}"
    assert "Bearer g" not in redacted, f"left {neighbor!r} leaked: {redacted!r}"


@pytest.mark.parametrize("neighbor", _CASE_FOLD_NEIGHBORS)
def test_bearer_redacted_beside_unicode_case_fold_right(neighbor):
    """A bare Bearer token followed by a case-fold character must redact."""
    redacted = _pattern_pass("Bearer " + "g" * 20 + neighbor)
    assert "[REDACTED]" in redacted, f"right {neighbor!r} leaked: {redacted!r}"
    assert "Bearer g" not in redacted, f"right {neighbor!r} leaked: {redacted!r}"


@pytest.mark.parametrize("prefix", ["Bearer", "BEARER", "bearer", "BeArEr"])
def test_bearer_literal_case_variants_still_redact(prefix):
    """Case-insensitivity of the Bearer literal is preserved (scoped flag)."""
    redacted = _pattern_pass("Authorization: " + prefix + " " + "z" * 20)
    assert "[REDACTED]" in redacted
    assert prefix + " zzz" not in redacted


def test_prose_without_tokens_unchanged():
    """Ordinary prose must pass through untouched."""
    text = "Hello, world. Nothing secret here."
    assert _pattern_pass(text) == text


# ── Streamed / interim redaction (P1 81097) ─────────────────────────────

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.run import _sanitize_gateway_final_response


class TestStreamedSecretRedaction:
    """Streamed deltas and interim commentary must be redacted before reaching the adapter.

    The non-streaming fallback (``_sanitize_gateway_final_response``) already
    redacts, but deltas wired directly to ``GatewayStreamConsumer.on_delta``
    bypassed it.  The consumer now applies the same forced redaction with a
    small tail hold-back so a token split across deltas (e.g. ``sk-`` in one
    delta and the body in the next) does not leak.
    """

    @pytest.mark.asyncio
    async def test_cjk_glued_token_split_across_deltas_is_redacted(self):
        """A CJK-glued token split across two deltas must not leak.

        Example from the review: ``xx中sk-...`` was visible in human chat when
        ``sk-`` and the body arrived in separate deltas, even though the
        non-streaming fallback would redact.
        """
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
        # Split the credential across deltas to exercise the stateful tail.
        token = "sk-" + "a" * 20
        # Use a longer prefix so the first delta alone exceeds the tail window
        # and would be sent as a preview before the second delta arrives.
        # Without the fix the first preview would leak ``sk-``.
        prefix = "Hello world, this is a long preamble that exceeds the tail window. xx中"
        consumer.on_delta(prefix + "sk-")
        # Give the run loop a chance to send the first preview
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("a" * 20)
        consumer.finish()
        await task

        all_texts = []
        for call in adapter.send.call_args_list:
            all_texts.append(call[1].get("content", ""))
        for call in adapter.edit_message.call_args_list:
            all_texts.append(call[1].get("content", ""))
        combined = "\n".join(all_texts)
        assert token not in combined, f"stream leaked raw token: {combined!r}"
        # Redacted form may be [REDACTED] (fallback) or masked sk-... (agent redactor)
        assert "[REDACTED]" in combined or "..." in combined

    @pytest.mark.asyncio
    async def test_bearer_split_across_deltas_is_redacted(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
        prefix = "Long preamble to exceed tail window for bearer test. "
        consumer.on_delta(prefix + "Bearer ")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("g" * 20)
        consumer.finish()
        await task

        all_texts = [c[1].get("content", "") for c in adapter.send.call_args_list] + [
            c[1].get("content", "") for c in adapter.edit_message.call_args_list
        ]
        combined = "\n".join(all_texts)
        assert "Bearer g" not in combined
        assert "[REDACTED]" in combined
        # Prefix preserved (group 1)
        assert "Bearer " in combined

    @pytest.mark.asyncio
    async def test_interim_commentary_is_redacted(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
        consumer.on_commentary("interim note Bearer " + "z" * 20 + " end")
        consumer.finish()
        await consumer.run()

        sent = [c[1].get("content", "") for c in adapter.send.call_args_list]
        assert sent, "commentary not sent"
        assert "Bearer z" not in sent[0]
        assert "[REDACTED]" in sent[0]

    @pytest.mark.asyncio
    async def test_sanitized_streamed_final_suppresses_duplicate_final(self):
        """Only suppress the final send when sanitized payloads match.

        The gateway records the streamed payload sanitized the same way as
        ``_sanitize_gateway_final_response``.  A raw streamed ``sk-...`` that was
        redacted to ``[REDACTED]`` must reconcile as matching the sanitized
        final ``[REDACTED]``, not as a mismatch that would cause a duplicate.
        """
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
        token = "sk-" + "a" * 20
        raw_final = "answer " + token + " done"
        sanitized_final = _sanitize_gateway_final_response("telegram", raw_final)
        assert token not in sanitized_final, f"raw token leaked in sanitized: {sanitized_final!r}"
        assert sanitized_final != raw_final

        # Simulate streaming the raw final (as the model would)
        consumer.on_delta(raw_final)
        consumer.finish()
        await consumer.run()

        # The consumer's recorded payload is sanitized, so it should match the
        # sanitized final.
        assert consumer.delivered_final_matches(sanitized_final) is True
        # And also match the raw final (since has_delivered normalizes via sanitized)
        assert consumer.delivered_final_matches(raw_final) is True
        # A different final should be a mismatch
        assert consumer.delivered_final_matches("different answer") is False

    @pytest.mark.asyncio
    async def test_normal_prose_still_streams_without_delay(self):
        """Non-secret prose must not be held back by the tail window."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
        consumer.on_delta("Hello world, no secrets here.")
        consumer.finish()
        await consumer.run()

        all_texts = [c[1].get("content", "") for c in adapter.send.call_args_list] + [
            c[1].get("content", "") for c in adapter.edit_message.call_args_list
        ]
        combined = "\n".join(all_texts)
        assert "Hello world" in combined
