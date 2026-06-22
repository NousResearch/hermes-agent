"""Tests for gateway.text_sanitize — strip model chain-of-thought before
messaging-platform delivery.

Reasoning models (``minimax-m3:cloud``, ``sonar-reasoning-pro``, Claude
with extended thinking, etc.) include the model's internal reasoning
blocks in the final assistant turn. The gateway relays that to messaging
platforms verbatim, which means Discord and Slack users see the full
chain-of-thought before the actual answer.

These tests pin the stripper's behavior. The actual integration lives in
``gateway.run._sanitize_gateway_final_response``; this module is the
pure-function kernel that the integration calls.
"""
from __future__ import annotations

import pytest

from gateway.text_sanitize import strip_reasoning_blocks, strip_for_platform


# The XML-like opening/closing tags in the test fixtures are written using
# \u003c / \u003e escapes so the source file itself never contains the raw
# angle-bracket markup. Some editors / renderers will otherwise collapse or
# hide the literal "<" and ">" in test data, leading to tests that pass on
# the wrong string.
THINK_OPEN = "\u003c" + "think" + "\u003e"
THINK_CLOSE = "\u003c/" + "think" + "\u003e"
HTML_COMMENT_OPEN = "\u003c!--"


class TestStripReasoningBlocks:
    """Pure-function tests for the CoT stripper."""

    def test_strips_think_block(self):
        raw = f"{THINK_OPEN}hidden reasoning{THINK_CLOSE}actual answer"
        out = strip_reasoning_blocks(raw)
        assert out == "actual answer"

    def test_strips_empty_think_block(self):
        raw = f"{THINK_OPEN}\n\n{THINK_CLOSE}actual answer"
        out = strip_reasoning_blocks(raw)
        assert out == "actual answer"

    def test_think_block_in_middle(self):
        raw = f"prefix{THINK_OPEN}hidden{THINK_CLOSE}suffix"
        out = strip_reasoning_blocks(raw)
        assert out == "prefixsuffix"

    def test_think_block_at_end(self):
        raw = f"actual answer{THINK_OPEN}hidden{THINK_CLOSE}"
        out = strip_reasoning_blocks(raw)
        assert out == "actual answer"

    def test_multiple_think_blocks(self):
        raw = (
            f"{THINK_OPEN}one{THINK_CLOSE}middle"
            f"{THINK_OPEN}two{THINK_CLOSE}end"
        )
        out = strip_reasoning_blocks(raw)
        assert out == "middleend"

    def test_strips_html_comment_reasoning(self):
        raw = f"{HTML_COMMENT_OPEN} reasoning: hidden --\u003eactual answer"
        out = strip_reasoning_blocks(raw)
        assert out == "actual answer"

    def test_preserves_legitimate_xml(self):
        # We intentionally only strip the role-like tag whitelist, not all XML.
        raw = "error: \u003cParseError\u003eline 5\u003c/ParseError\u003e"
        out = strip_reasoning_blocks(raw)
        assert "ParseError" in out
        assert "line 5" in out

    def test_no_reasoning_block_returns_unchanged(self):
        text = "this is a normal response with no CoT wrapping"
        assert strip_reasoning_blocks(text) == text

    def test_collapses_excess_whitespace_after_strip(self):
        raw = f"{THINK_OPEN}hidden{THINK_CLOSE}\n\n\n\n\nactual"
        out = strip_reasoning_blocks(raw)
        assert out == "actual"

    def test_strips_case_insensitively(self):
        # The opening tag with mixed case still matches.
        mixed_open = "\u003c" + "ThInK" + "\u003e"
        raw = f"{mixed_open}HIDDEN{THINK_CLOSE}answer"
        out = strip_reasoning_blocks(raw)
        assert out == "answer"

    def test_multiline_think_block(self):
        raw = (
            f"{THINK_OPEN}\n"
            "step 1: think about things\n"
            "step 2: think some more\n"
            "step 3: conclude\n"
            f"{THINK_CLOSE}\n"
            "the actual answer"
        )
        out = strip_reasoning_blocks(raw)
        assert out == "the actual answer"

    def test_only_think_block_returns_empty(self):
        raw = f"{THINK_OPEN}only reasoning, no answer{THINK_CLOSE}"
        out = strip_reasoning_blocks(raw)
        # The stripper returns empty string; the caller (strip_for_platform)
        # is responsible for substituting the user-facing fallback.
        assert out == ""

    def test_strips_gateway_prepended_reasoning_block(self):
        # The gateway itself sometimes prepends a "💭 **Reasoning:**\n```\n...\n```\n\n"
        # block to the response. The stripper should eat that too.
        raw = (
            "💭 **Reasoning:**\n"
            "```\n"
            "step 1\nstep 2\n"
            "```\n"
            "\n"
            "the actual answer"
        )
        out = strip_reasoning_blocks(raw)
        assert out == "the actual answer"


class TestStripForPlatform:
    """Tests the platform-aware wrapper that the gateway calls."""

    def test_discord_strips_think_block(self):
        raw = f"{THINK_OPEN}hidden{THINK_CLOSE}hello"
        out = strip_for_platform("discord", raw)
        assert out == "hello"

    def test_slack_strips_think_block(self):
        raw = f"{THINK_OPEN}hidden{THINK_CLOSE}hello"
        out = strip_for_platform("slack", raw)
        assert out == "hello"

    def test_telegram_does_not_strip(self):
        # Telegram is a known-working path; the stripper is opt-in for
        # discord/slack only in this PR.
        raw = f"{THINK_OPEN}hidden{THINK_CLOSE}hello"
        out = strip_for_platform("telegram", raw)
        assert out == raw

    def test_unknown_platform_does_not_strip(self):
        raw = f"{THINK_OPEN}hidden{THINK_CLOSE}hello"
        out = strip_for_platform("irc", raw)
        assert out == raw

    def test_discord_empty_after_strip_returns_fallback(self):
        raw = f"{THINK_OPEN}only reasoning{THINK_CLOSE}"
        out = strip_for_platform("discord", raw)
        # The wrapper substitutes a user-facing message rather than sending empty.
        assert out == "(the model produced no visible response)"

    def test_discord_handles_none_text(self):
        # Defensive: if a caller passes None, don't crash.
        assert strip_for_platform("discord", None) is None

    def test_discord_handles_empty_text(self):
        assert strip_for_platform("discord", "") == ""
