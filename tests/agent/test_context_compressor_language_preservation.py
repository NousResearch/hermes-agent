"""Tests for language preservation in context-compaction summaries.

The summarizer prompt carries the language directive twice: in the preamble
and again at the end of the section template (immediately before "Write only
the summary body"), so the rule is also the LAST instruction the model reads.
A preamble-only rule was observed to be ignored in production (a summary came
back with entire sections in the wrong language), and a language-flipped
checkpoint poisons every later turn of the session because the summary is
injected into subsequent context.

These exercise ``_generate_summary`` directly -- the function that builds the
summarizer prompt. ``test_context_compressor_summary_continuity`` already
proves ``compress()`` routes into ``_generate_summary``.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _turns():
    return [
        {"role": "user", "content": "do the first thing"},
        {"role": "assistant", "content": "did the first thing"},
        {"role": "user", "content": "do the second thing"},
        {"role": "assistant", "content": "did the second thing"},
    ]


def test_language_rule_appears_in_preamble_and_at_end_of_prompt():
    """The rule must also be the last instruction the summarizer reads, not
    only a preamble line thousands of characters before it writes anything."""
    compressor = _compressor()
    with patch(
        "agent.context_compressor.call_llm", return_value=_response("summary")
    ) as mock_call:
        compressor._generate_summary(_turns())

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert prompt.count("same language the user was using") == 2
    last = prompt.rfind("same language the user was using")
    assert len(prompt) - last < 400  # inside the final 400 characters


def test_sections_and_headings_clause_present():
    """Headings were part of the observed failure (whole sections flipped),
    so the end-of-prompt rule must also cover structure, not just prose."""
    compressor = _compressor()
    with patch(
        "agent.context_compressor.call_llm", return_value=_response("summary")
    ) as mock_call:
        compressor._generate_summary(_turns())

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "Sections and headings follow that language" in prompt
    # The end-of-template anchor still closes the prompt.
    assert prompt.rstrip().endswith("Do not include any preamble or prefix.")
