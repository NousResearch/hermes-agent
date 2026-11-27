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

import contextlib
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


def test_end_of_prompt_rule_in_no_user_turn_variant():
    """The no-user-turn variant has its own language directive (different
    preamble wording, same protective intent). Its language + structure rule
    must also sit at the end of the prompt, right before the closing anchor."""
    compressor = _compressor()
    compressor._summary_has_user_turn = False
    with patch(
        "agent.context_compressor.call_llm", return_value=_response("summary")
    ) as mock_call, contextlib.suppress(Exception):
        # The mocked summary body doesn't match the no-user-turn sentinel, so
        # post-generation validation raises; the prompt under test was already built.
        compressor._generate_summary([])

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    # The variant's own language directive is present...
    assert "no user-authored turns" in prompt
    # ...and the structure rule is the last instruction before the closing anchor.
    assert "Sections and headings follow that language; only quoted literal values stay verbatim." in prompt
    last = prompt.rfind("Sections and headings follow that language")
    assert len(prompt) - last < 400
    assert prompt.rstrip().endswith("Do not include any preamble or prefix.")
