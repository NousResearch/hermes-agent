"""Reasoning summary-part boundary repair (agent/reasoning_summaries.py)."""

from agent.reasoning_summaries import (
    REASONING_SUMMARY_MAX_TITLE_CHARS,
    REASONING_SUMMARY_MAX_TITLES,
    reasoning_summary_titles,
    separate_glued_reasoning_blocks,
)


def test_reasoning_summary_titles_prefers_ordered_unique_headings():
    assert reasoning_summary_titles(
        "**Inspecting files**\nDetails\n\n**Running tests**\nMore details\n\n**inspecting files**"
    ) == ["Inspecting files", "Running tests"]


def test_reasoning_summary_titles_uses_plain_provider_summary():
    assert reasoning_summary_titles("  Planning the implementation  \nSupporting detail") == [
        "Planning the implementation"
    ]
    assert reasoning_summary_titles(None) == []


def test_reasoning_summary_titles_bound_output_and_clean_emphasis():
    titles = reasoning_summary_titles(
        "\n".join(
            ["***Italic heading***"]
            + [f"**Title {index} {'x' * 200}**" for index in range(20)]
        )
    )

    assert titles[0] == "Italic heading"
    assert len(titles) == REASONING_SUMMARY_MAX_TITLES
    assert all(len(title) <= REASONING_SUMMARY_MAX_TITLE_CHARS for title in titles)


def _stream(deltas):
    """Accumulate *deltas* the way the chat-completions stream loop does."""
    parts: list[str] = []
    for delta in deltas:
        parts.append(
            separate_glued_reasoning_blocks(parts[-1] if parts else "", delta)
        )
    return "".join(parts)


def test_heading_only_parts_do_not_glue_into_one_run():
    # The shape observed live on Nous Portal's openai/gpt-5.6-sol: each delta
    # is a bare heading, so consecutive parts produce a `****` run.
    text = _stream(
        [
            "**Investigating likely culprit PRs**",
            "**Inspecting message schema and tool_calls content**",
            "**Analyzing interrupted tool call impact**",
        ]
    )

    assert "****" not in text
    assert text.splitlines() == [
        "**Investigating likely culprit PRs**",
        "",
        "**Inspecting message schema and tool_calls content**",
        "",
        "**Analyzing interrupted tool call impact**",
    ]


def test_prose_body_does_not_glue_onto_the_next_heading():
    # vercel/ai#6742's repro: a part ends in prose and the next heading butts
    # straight onto it, with no `****` run to key off.
    text = _stream(
        [
            "**Simulating a greeting stream**\n\nIt feels like a streaming interaction!",
            "**Simulating a greeting stream**\n\nI want to meet the request.",
        ]
    )

    assert "interaction!**" not in text
    assert "interaction!\n\n**Simulating" in text


def test_token_streamed_reasoning_is_untouched():
    deltas = ["Looking at", " the session", " logs, I see", " one bold word."]

    assert _stream(deltas) == "".join(deltas)


def test_bold_word_mid_sentence_is_not_a_boundary():
    # Emphasis inside token-streamed prose arrives after a space.
    assert separate_glued_reasoning_blocks("I see the ", "**signature**") == "**signature**"


def test_unclosed_emphasis_fragment_is_not_a_boundary():
    # A token stream splitting emphasis across deltas opens but never closes.
    assert separate_glued_reasoning_blocks("weighing", "**") == "**"


def test_boundary_needs_a_bold_opener():
    assert separate_glued_reasoning_blocks("**Closing**", "plain head") == "plain head"


def test_empty_operands_pass_through():
    assert separate_glued_reasoning_blocks("", "**first**") == "**first**"
    assert separate_glued_reasoning_blocks("**first**", "") == ""
