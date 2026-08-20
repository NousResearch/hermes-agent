"""Regression tests for issue #78359 — spurious truncation on URL endings.

``_has_natural_response_ending()`` must recognise a response that ends with a
bare URL (``http://…``, ``https://…``, ``ftp://…``) as intentionally finished.
Previously only punctuation, emoji, code fences, and ``^`` were recognised, so
a URL ending in a letter (e.g. ``index.html``) caused
``_should_treat_stop_as_truncated()`` to fire a spurious continuation prompt on
Ollama-hosted GLM models, concatenating run-on text onto the URL without a
separator.
"""

from __future__ import annotations

import re

import pytest

from run_agent import AIAgent


# Convenience alias for the static method under test.
_natural = AIAgent._has_natural_response_ending


# ── URL endings that MUST be recognised as natural ────────────────────────

@pytest.mark.parametrize(
    "content",
    [
        "The docs are at https://example.com/index.html",
        "Check http://example.com/page",
        "Download from ftp://mirror.example.org/file.tar.gz",
        "Link: https://example.com/path?q=1&p=2",
        "See https://example.com/path#section",
        "Go to https://example.com/path/",
        "Short: https://a.b",
    ],
    ids=[
        "https_html",
        "http_page",
        "ftp_file",
        "https_query_params",
        "https_fragment",
        "https_trailing_slash",
        "https_short",
    ],
)
def test_url_ending_is_natural(content: str):
    """A response whose final token is a scheme-URL is complete (#78359)."""
    assert _natural(content) is True


# ── URL NOT at the end — must still be non-natural ─────────────────────────

@pytest.mark.parametrize(
    "content",
    [
        "Visit https://example.com for details",
        "The link https://example.com/page is broken",
    ],
    ids=["url_mid_sentence", "url_followed_by_text"],
)
def test_url_not_at_end_is_not_natural(content: str):
    """A URL that is NOT the terminal token must not short-circuit to True."""
    assert _natural(content) is False


# ── Existing endings must not regress ──────────────────────────────────────

@pytest.mark.parametrize(
    "content,expected",
    [
        ("Done.", True),
        ("Is it ready?", True),
        ("```\ncode\n```", True),
        ("See below^", True),
        # Emoji ending (U+1F600 grin)
        ("Hello \U0001F600", True),
        # Empty / whitespace
        ("", False),
        ("   ", False),
        # Bare word — no natural ending
        ("just some words", False),
    ],
    ids=[
        "period",
        "question",
        "code_fence",
        "caret",
        "emoji",
        "empty",
        "whitespace_only",
        "bare_word",
    ],
)
def test_existing_endings_no_regression(content: str, expected: bool):
    assert _natural(content) is expected


# ── Integration: _should_treat_stop_as_truncated with URL ending ───────────

class TestShouldTreatStopAsTruncated:
    """Verify the full truncation-detection path honours URL endings."""

    def _make_ollama_glm_agent(self):
        from unittest.mock import MagicMock
        agent = MagicMock(spec=AIAgent)
        # Wire the real static method (unbound) so we test production code.
        agent._has_natural_response_ending = staticmethod(
            AIAgent._has_natural_response_ending
        )
        agent._strip_think_blocks = lambda content: content
        agent._is_ollama_glm_backend = MagicMock(return_value=True)
        agent.api_mode = "chat_completions"
        agent.model = "glm-4"
        agent.provider = "ollama"
        return agent

    def test_url_ending_not_truncated(self):
        """A stop-finish with a URL ending must NOT be treated as truncated."""
        from types import SimpleNamespace

        agent = self._make_ollama_glm_agent()
        assistant_msg = SimpleNamespace(
            content="The answer is https://example.com/docs.html",
            tool_calls=None,
        )
        messages = [{"role": "tool", "content": "result"}]

        result = AIAgent._should_treat_stop_as_truncated(
            agent, "stop", assistant_msg, messages
        )
        assert result is False, (
            "URL ending should be recognised as natural — no spurious continuation"
        )

    def test_bare_word_ending_is_truncated(self):
        """A stop-finish with an incomplete bare-word ending IS truncated."""
        from types import SimpleNamespace

        agent = self._make_ollama_glm_agent()
        assistant_msg = SimpleNamespace(
            content="just some random words without ending",
            tool_calls=None,
        )
        messages = [{"role": "tool", "content": "result"}]

        result = AIAgent._should_treat_stop_as_truncated(
            agent, "stop", assistant_msg, messages
        )
        assert result is True
