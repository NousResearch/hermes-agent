"""Reasoning summary-part boundary repair (agent/reasoning_summaries.py)."""

from agent.reasoning_summaries import separate_glued_reasoning_blocks


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


def test_list_shaped_delta_flattens_instead_of_crashing():
    # Grok / custom OpenAI-compat gateways stream `reasoning_content` as a list of
    # content parts; the heading-boundary check used to raise
    # `'list' object has no attribute 'startswith'` (#104711).
    text = _stream(
        [
            "**Weighing the request**",
            [{"type": "text", "text": "**Planning the reply**"}],
        ]
    )

    assert "****" not in text
    assert text == "**Weighing the request**\n\n**Planning the reply**"


def test_list_shaped_delta_plain_strings_keep_boundary_semantics():
    # A bare-string list must behave exactly like its flattened text, both as delta
    # and as the trailing operand.
    assert separate_glued_reasoning_blocks(
        ["**first", " part**"], ["**second part**"]
    ) == "\n\n**second part**"
