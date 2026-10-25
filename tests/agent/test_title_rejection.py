"""Tests for agent.title_rejection — title-shape guards.

The refusal-shaped guard rejects a title that answers the opener instead of
naming the work. Its patterns were measured against the fleet's own stored
session titles; these tests pin both directions — the garbage it must reject
and the legitimate titles it must not touch.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.title_rejection import (
    EXAMPLE_ECHO_REJECT,
    is_prompt_example_echo,
    is_rejection_shaped_title,
)


class TestRejectionShapedTitle:
    """Unit tests for is_rejection_shaped_title()."""

    @pytest.mark.parametrize("title", [
        # The card's own bad shape: a negation of the deliverable the opener
        # asked for ("write a noncon scene with mind control").
        "no scene with mind control",
        "No content generated",
        "Nothing was produced",
        # A refusal to deliver.
        "Unable to work with that request",
        "Unable to provide title",
        "Unable to title this conversation",
        "Unable to title this request",
        "Cannot generate a title for this",
        "Can't help with that request",
        # A first-person refusal.
        "I can't write that scene",
        "I can't continue this content",
        "I can't help generate a title",
        # A third-person report of the request instead of a title.
        "The user is asking me to write a scene",
        "The user is requesting explicit content",
        "The user wants an explicit account",
    ])
    def test_rejects_refusal_shaped_titles(self, title):
        """A title that refuses or reports names nothing a reader could search
        by — it must be rejected so the derived title wins instead."""
        assert is_rejection_shaped_title(title) is True

    @pytest.mark.parametrize("title", [
        # Legitimate titles that share a leading "no"/"nothing" with the
        # rejected ones — they name real work, not an absent deliverable.
        "No change on North Star pwa",
        "Nothing owns simulator lifecycle",
        # A real finding, not a refusal: "reproduce" names no deliverable.
        "Cannot reproduce on ARM64",
        "Unable to reach the API endpoint",
        "I can't reproduce the ARM bug",
        # Topical and refusal-adjacent but genuine titles.
        "Explicit sexual content",
        "Refuse explicit non-consent image prompt",
        "Refused: sexualized content involving minors",
        "refusal",
        # Ordinary titles.
        "Autumn leaves fall in park",
        "Postgres connection pool exhaustion",
        "System instability",
        "Fix gear system on bike",
        "Write Python unit test with pytest",
        "Chapter summary for newsletter",
    ])
    def test_keeps_legitimate_titles(self, title):
        """The guard must not touch a title that names real work — a false
        positive here would silently replace a good title with a derived one."""
        assert is_rejection_shaped_title(title) is False

    def test_rejects_a_request_report_even_when_topical(self):
        """A model that reports the request back instead of titling it is the
        answer-shaped failure. The guard only ever sees MODEL output, so a
        topical-but-reporting title is still garbage — the derived title wins
        instead."""
        assert is_rejection_shaped_title("The user is asking about billing") is True

    def test_empty_title_is_not_rejected(self):
        """An empty title is a parse failure, not a refusal; the caller
        handles it separately."""
        assert is_rejection_shaped_title("") is False


class TestPromptExampleEcho:
    """The echo guard moved here from agent.title_generator; its behaviour is
    unchanged and is pinned here so the two guards travel together."""

    def test_rejects_echo_of_prompt_examples(self):
        for example in sorted(EXAMPLE_ECHO_REJECT):
            assert is_prompt_example_echo(example) is True

    def test_allows_friendly_greeting(self):
        """'Friendly greeting' is prescribed output for bare greetings, not an
        echo failure — it must pass the guard."""
        assert is_prompt_example_echo("Friendly greeting") is False

    def test_strips_wrappers_before_comparing(self):
        """Bracket/quote wrappers cannot bypass the guard."""
        assert is_prompt_example_echo('"Fix login button on mobile"') is True
        assert is_prompt_example_echo("[Code changes]") is True


class TestRejectionReachesTheTitlePath:
    """The guard is only useful if the title path honours it: a rejected title
    must leave the session with the derived title, not the model's garbage."""

    def test_refusal_shaped_model_output_leaves_no_title(self):
        """generate_title() returns None for a refusal-shaped model reply, so
        auto_title_session() falls back to the derived title."""
        from agent.title_generator import generate_title

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"title": "Unable to work with that request"}'

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("write a scene of two adults having rough sex") is None

    def test_normal_title_still_accepted(self):
        """A legitimate title is unaffected by the new guard."""
        from agent.title_generator import generate_title

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Investigate the title resolver bug"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("question") == "Investigate the title resolver bug"
