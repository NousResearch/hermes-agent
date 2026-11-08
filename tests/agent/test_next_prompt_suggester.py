"""Behaviour contract of the next-prompt suggester: what it asks the model and what it keeps."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import next_prompt_suggester as nps


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _suggest(text, user="run the tests", assistant="All green."):
    with patch.object(nps, "call_llm", return_value=_resp(text)) as m:
        return nps.generate_next_prompt(user, assistant), m


def test_plain_answer_is_returned_and_uses_next_prompt_task():
    out, m = _suggest("commit that as a PR")
    assert out == "commit that as a PR"
    kwargs = m.call_args.kwargs
    assert kwargs["task"] == "next_prompt"
    assert kwargs["max_tokens"] <= 256


@pytest.mark.parametrize("raw", ["NONE", "none", " None. ", "NONE\n", ""])
def test_none_and_empty_show_nothing(raw):
    assert _suggest(raw)[0] is None


@pytest.mark.parametrize("raw,expected", [
    ('"commit it"', "commit it"),
    ("'commit it'", "commit it"),
    ("`commit it`", "commit it"),
    ("“commit it”", "commit it"),
])
def test_wrapping_quotes_are_stripped(raw, expected):
    assert _suggest(raw)[0] == expected


def test_multiline_output_keeps_first_nonempty_line():
    assert _suggest("\n\nrun the full suite\nthen diff against the baseline")[0] == "run the full suite"


def test_input_is_clipped_to_the_tail():
    _, m = _suggest("x", user="u" * 50_000, assistant="a" * 50_000)
    sent = "".join(msg["content"] for msg in m.call_args.kwargs["messages"])
    assert len(sent) < 10_000
    assert sent.count("a") > 0  # the tail of the assistant reply survives


def test_blank_conversation_makes_no_call():
    with patch.object(nps, "call_llm") as m:
        assert nps.generate_next_prompt("", "") is None
    m.assert_not_called()


def test_failure_logs_at_debug_and_shows_nothing(caplog):
    with patch.object(nps, "call_llm", side_effect=TimeoutError("slow")), caplog.at_level("DEBUG"):
        assert nps.generate_next_prompt("hi", "hello") is None
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]


def test_last_exchange_uses_only_the_latest_assistant_message():
    history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "new reply"},
    ]
    assert nps.last_exchange(history) == ("q2", "new reply")
    # The latest assistant message is tool-call-only: nothing to follow up on, never the older reply.
    history[-1] = {"role": "assistant", "content": "", "tool_calls": [{}]}
    assert nps.last_exchange(history) == ("q2", "")
    assert nps.last_exchange([]) == ("", "")
