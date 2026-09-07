"""Regression tests: HTTP 413 recovery must score progress in BYTES.

Bug (#88960 / #47339): a 413 is a *byte*-size error, but the recovery loop in
``agent/conversation_loop.py`` scored compression progress with
``estimate_messages_tokens_rough``, which deliberately prices every image at
a flat per-image token cost so screenshots don't trigger premature
compaction.  When the payload is image-dominated that progress test can never
be satisfied: in the reporting session two ``vision_analyze`` results were
5,627,202 bytes — 96.6% of the request body — while contributing only ~3K of
the ~80K token estimate.  Compaction (post-#97160) frees those megabytes, but
the token-scored check reported "no progress", burned all three attempts, and
wedged the session permanently at 13% context usage.

The fix: the 413 no-progress check measures ``serialized_messages_bytes``
(exact, free) before and after each compression pass, never the token
estimate.  These tests assert that invariant directly.
"""

import pytest

from agent.message_sanitization import serialized_messages_bytes
from agent.model_metadata import estimate_messages_tokens_rough


def _data_url_image(size_bytes: int) -> dict:
    """An image part whose inline data URL is ~``size_bytes`` long."""
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + ("A" * size_bytes)},
    }


def _tool_msg_with_image(size_bytes: int, text: str = "screenshot captured") -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": [
            {"type": "text", "text": text},
            _data_url_image(size_bytes),
        ],
    }


def _image_aged_out(msg: dict) -> dict:
    """The message after compaction replaced its image with a placeholder."""
    out = dict(msg)
    out["content"] = [
        p for p in msg["content"] if p.get("type") != "image_url"
    ] + [{"type": "text", "text": "[image removed during compaction]"}]
    return out


class TestSerializedMessagesBytes:
    def test_counts_inline_data_url_payloads(self):
        small = serialized_messages_bytes([_tool_msg_with_image(1_000)])
        huge = serialized_messages_bytes([_tool_msg_with_image(3_000_000)])
        assert huge - small == pytest.approx(3_000_000 - 1_000, abs=64)

    def test_is_exact_not_an_estimate(self):
        """Same input, same answer — a measurement, not a heuristic."""
        messages = [_tool_msg_with_image(50_000), {"role": "user", "content": "hi"}]
        assert serialized_messages_bytes(messages) == serialized_messages_bytes(
            messages
        )

    def test_utf8_bytes_not_codepoints(self):
        ascii_msgs = [{"role": "user", "content": "aaaa"}]
        utf8_msgs = [{"role": "user", "content": "éééé"}]  # 2 bytes each in UTF-8
        assert serialized_messages_bytes(utf8_msgs) > serialized_messages_bytes(
            ascii_msgs
        )

    def test_degenerate_input(self):
        assert serialized_messages_bytes([]) == 0
        assert serialized_messages_bytes("not-a-list") == 0  # type: ignore[arg-type]

    def test_never_raises_on_non_serializable_content(self):
        class Weird:
            pass

        messages = [{"role": "tool", "content": Weird()}]
        assert serialized_messages_bytes(messages) > 0


class TestTokenEstimateIsBlindToImageBytes:
    """The root cause, asserted directly.

    This is why a token-scored progress check can never clear an
    image-dominated 413: the estimate barely moves regardless of how many
    megabytes are on the wire.
    """

    def test_estimate_barely_moves_as_image_bytes_explode(self):
        small = [_tool_msg_with_image(1_000)]
        huge = [_tool_msg_with_image(3_000_000)]

        small_tokens = estimate_messages_tokens_rough(small)
        huge_tokens = estimate_messages_tokens_rough(huge)

        # ~3000x more bytes on the wire...
        assert len(huge[0]["content"][1]["image_url"]["url"]) > 2_000_000

        # ...but the token estimate is essentially unchanged, so a
        # "did compression make progress?" check scored in tokens
        # (new < original * 0.95) can never be satisfied by freeing images.
        assert huge_tokens < small_tokens * 2


class TestByteScoredProgressCheck:
    """The invariant the fix installs: 413 progress is judged in bytes.

    Mirrors the exact decision expression in the 413 handler:
    ``len(messages) < original_len or new_bytes < original_bytes * 0.95``.
    """

    def _decision(self, before: list, after: list, *, metric: str) -> bool:
        if len(after) < len(before):
            return True
        if metric == "tokens":
            o = estimate_messages_tokens_rough(before)
            n = estimate_messages_tokens_rough(after)
        else:
            o = serialized_messages_bytes(before)
            n = serialized_messages_bytes(after)
        return n > 0 and n < o * 0.95

    def _image_dominated_session(self):
        """The real-world shape that wedged a session: ~190 substantive text
        turns (~77K token estimate), two multi-MB vision results = 96%+ of
        the serialized body but a tiny slice of the token estimate."""
        messages = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"turn {i} " + ("x" * 900),
            }
            for i in range(190)
        ]
        messages.insert(50, _tool_msg_with_image(2_756_000))
        messages.insert(80, _tool_msg_with_image(2_871_000))
        return messages

    def test_token_scoring_wedges_on_image_dominated_payload(self):
        """BEFORE-behavior pin: compaction frees megabytes, token check
        still says no-progress -> attempts burn -> session wedges."""
        before = self._image_dominated_session()
        # Compaction ages out the older tool image (#97160) — same message
        # count, ~2.7MB freed.
        after = [
            _image_aged_out(m)
            if isinstance(m.get("content"), list) and m is before[50]
            else m
            for m in before
        ]
        freed = serialized_messages_bytes(before) - serialized_messages_bytes(after)
        assert freed > 2_000_000, "compaction really freed megabytes"
        assert self._decision(before, after, metric="tokens") is False, (
            "token yardstick is blind to the freed bytes — this is the bug"
        )

    def test_byte_scoring_sees_the_same_reduction(self):
        before = self._image_dominated_session()
        after = [
            _image_aged_out(m)
            if isinstance(m.get("content"), list) and m is before[50]
            else m
            for m in before
        ]
        assert self._decision(before, after, metric="bytes") is True, (
            "byte yardstick must recognize a multi-MB reduction as progress"
        )

    def test_text_only_compression_still_scores_progress_in_bytes(self):
        """Non-image 413s keep working: summarizing text shrinks bytes too."""
        before = [
            {"role": "user", "content": "x" * 10_000} for _ in range(50)
        ]
        after = [
            {"role": "user", "content": "x" * 10_000} for _ in range(10)
        ]
        # message-count branch fires first, but the byte branch alone would
        # also pass:
        assert serialized_messages_bytes(after) < (
            serialized_messages_bytes(before) * 0.95
        )
        assert self._decision(before, after, metric="bytes") is True

    def test_true_no_progress_is_still_terminal(self):
        """When nothing actually shrank, byte scoring must NOT fake progress."""
        before = self._image_dominated_session()
        after = list(before)  # identical payload
        assert self._decision(before, after, metric="bytes") is False


class TestConversationLoopWiring:
    """The handler really uses the byte metric (source-level contract)."""

    def test_413_branch_scores_bytes_not_tokens(self):
        import inspect

        import agent.conversation_loop as loop

        src = inspect.getsource(loop)
        # The byte measurement is taken before and after the 413 compression
        # pass and drives the progress decision.
        assert "original_bytes = serialized_messages_bytes(messages)" in src
        assert "new_bytes = serialized_messages_bytes(messages)" in src
        assert "new_bytes < original_bytes * 0.95" in src
        # The old token-scored expression is gone from the 413 branch's
        # decision. Isolate the 413 handler region: from its status line to
        # its terminal error. (Token scoring survives in the
        # context-overflow branches, which ARE token-budget errors.)
        start = src.index("Request payload too large (413) — compression attempt")
        end = src.index("Payload too large and cannot compress further")
        branch = src[start:end]
        assert "new_tokens < original_tokens * 0.95" not in branch
        assert "original_bytes = serialized_messages_bytes" in branch
        assert "new_bytes = serialized_messages_bytes" in branch
