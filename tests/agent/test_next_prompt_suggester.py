"""Behaviour contract of the next-prompt suggester: what it asks the model and what it keeps."""

import json
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


@pytest.mark.parametrize("n,kept", [(20, True), (21, False)])
def test_output_over_the_word_limit_shows_nothing(n, kept):
    text = " ".join(["word"] * n)
    assert nps.clean_suggestion(text) == (text if kept else None)


def test_rambling_first_line_is_dropped_not_truncated():
    rambling = "Nothing appears yet. I'll try a couple of turns, then tell you what I see after that happens and whether it is working as intended"
    assert nps.clean_suggestion(f'"{rambling}"\nrun it') is None


@pytest.mark.parametrize("n,kept", [(120, True), (121, False)])
def test_output_over_the_char_limit_shows_nothing(n, kept):
    text = "あ" * n  # no spaces, so the word cap cannot catch it
    assert nps.clean_suggestion(text) == (text if kept else None)


@pytest.mark.parametrize("line", [
    "Nothing has appeared yet, so I'll ask the assistant to check the Haiku call",
    "The user wants to see the turn-settle hook",
    "Ask the assistant to retry",
    "the AI should answer first",
    "It failed, so the model needs another try",
    "The bot is not responding",
])
def test_narration_about_the_parties_is_dropped(line):
    assert nps.clean_suggestion(line) is None


@pytest.mark.parametrize("line", [
    "rename the assistant module",
    "add a test for the user model",
    "show the user's profile page",
    "switch the model to opus",
    "restart the bot",
    "yes, go ahead",
    "nothing yet",
    "done",
])
def test_real_user_requests_pass(line):
    assert nps.clean_suggestion(line) == line


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


_CLARIFY = ('{"responses": [{"question": "Should I delete it?", "choices_offered": ["No, keep it", '
            '"Yes, delete it"], "status": "answered", "user_response": "No, keep it"}], "outcome": "submitted"}')


def _clarify_history(tool_msg, call=None):
    return [
        {"role": "user", "content": "list the files, then ask me whether to delete the oldest backup"},
        {"role": "assistant", "content": "", "tool_calls": [
            call or {"id": "c1", "type": "function", "function": {"name": "clarify", "arguments": "{}"}}]},
        tool_msg,
        {"role": "assistant", "content": "I kept 2026-11-08-backup.tar.gz."},
    ]


def _model_input(history):
    user, assistant = nps.last_exchange(history)
    with patch.object(nps, "call_llm", return_value=_resp("NONE")) as m:
        nps.generate_next_prompt(user, assistant)
    return "\n".join(msg["content"] for msg in m.call_args.kwargs["messages"][1:])


@pytest.mark.parametrize("tool_msg", [
    {"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": _CLARIFY},
    {"role": "tool", "name": "clarify", "tool_call_id": "c1", "content": _CLARIFY},
    {"role": "tool", "tool_call_id": "c1", "content": _CLARIFY},  # name only on the assistant tool_calls
])
def test_clarify_answer_reaches_the_model(tool_msg):
    sent = _model_input(_clarify_history(tool_msg))
    assert "Should I delete it?" in sent and "No, keep it" in sent
    assert "I kept 2026-11-08-backup.tar.gz." in sent


def test_history_without_clarify_is_unchanged():
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert nps.last_exchange(history) == ("q", "a")


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"responses": "x"}', '{"responses": [{"status": "answered"}]}', None])
def test_malformed_clarify_content_is_ignored(content):
    history = _clarify_history({"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": content})
    assert nps.last_exchange(history) == (history[0]["content"], "I kept 2026-11-08-backup.tar.gz.")


def test_other_tools_and_unanswered_clarify_are_ignored():
    other = {"role": "tool", "tool_name": "terminal", "tool_call_id": "c1", "content": _CLARIFY}
    assert "No, keep it" not in nps.last_exchange(_clarify_history(other))[0]
    skipped = _CLARIFY.replace('"answered"', '"skipped"')
    msg = {"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": skipped}
    assert "No, keep it" not in nps.last_exchange(_clarify_history(msg))[0]


def test_clarify_from_an_earlier_turn_is_not_included():
    history = _clarify_history({"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": _CLARIFY})
    history += [{"role": "user", "content": "thanks, now run tests"}, {"role": "assistant", "content": "All green."}]
    assert nps.last_exchange(history) == ("thanks, now run tests", "All green.")


def test_clarify_block_is_clipped_and_assistant_tail_survives():
    big = json.dumps({"responses": [
        {"question": "q" * 5000, "status": "answered", "user_response": "a" * 5000} for _ in range(10)]})
    history = _clarify_history({"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": big})
    history[-1]["content"] = "x" * 50_000 + "THE END"
    user, assistant = nps.last_exchange(history)
    assert len(user) < nps.MAX_USER_CHARS + nps.MAX_CLARIFY_CHARS + 10
    with patch.object(nps, "call_llm", return_value=_resp("NONE")) as m:
        nps.generate_next_prompt(user, assistant)
    sent = "".join(msg["content"] for msg in m.call_args.kwargs["messages"])
    assert len(sent) < 6_000 and sent.endswith("THE END")


def test_long_user_message_without_clarify_stays_within_the_user_budget():
    history = [{"role": "user", "content": "u" * 5000}, {"role": "assistant", "content": "done"}]
    user, _ = nps.last_exchange(history)
    assert len(user) == nps.MAX_USER_CHARS


def test_multi_select_answer_is_joined_not_repr():
    content = json.dumps({"responses": [
        {"question": "Which?", "status": "answered", "user_response": ["alpha", "beta"]}]})
    history = _clarify_history({"role": "tool", "tool_name": "clarify", "tool_call_id": "c1", "content": content})
    assert nps.last_exchange(history)[0].endswith("Asked: Which? / User answered: alpha, beta")
