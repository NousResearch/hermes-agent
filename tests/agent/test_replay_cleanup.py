"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_incomplete_reasoning_tail,
    strip_interrupted_tool_tails,
    sanitize_replay_history,
)


def _user(text):
    return {"role": "user", "content": text}


def _assistant_tc(name):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}


def test_is_interrupted_tool_result_markers():
    assert is_interrupted_tool_result("[Command interrupted]")
    assert is_interrupted_tool_result("foo\nexit_code: 130 (interrupt)\nbar")
    assert not is_interrupted_tool_result("exit_code: 0\nclean output")
    assert not is_interrupted_tool_result("ordinary tool output")
    assert not is_interrupted_tool_result(None)


def test_strip_dangling_tool_call_tail_removes_unanswered_read_only_tail():
    history = [_user("hi"), _assistant_tc("read_file")]
    out = strip_dangling_tool_call_tail(history)
    assert out == [_user("hi")]


def test_dangling_side_effect_is_recovered_as_unknown_not_erased():
    history = [_user("hi"), _assistant_tc("write_file")]

    out = strip_dangling_tool_call_tail(history)

    assert out[:-1] == history
    assert out[-1]["role"] == "tool"
    assert out[-1]["tool_call_id"] == "c1"
    assert out[-1]["effect_disposition"] == "unknown"
    assert "may have executed" in out[-1]["content"].lower()


def test_dangling_session_mutation_is_recovered_as_unknown():
    history = [_user("hi"), _assistant_tc("todo")]

    out = strip_dangling_tool_call_tail(history)

    assert out[:-1] == history
    assert out[-1]["effect_disposition"] == "unknown"
    assert "may have executed" in out[-1]["content"].lower()


def test_mixed_dangling_batch_uses_truthful_per_call_wording():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "read", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "write", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    out = strip_dangling_tool_call_tail([_user("hi"), assistant])

    read_result, write_result = out[-2:]
    assert read_result["effect_disposition"] == "none"
    assert "no effect" in read_result["content"].lower()
    assert "unknown" not in read_result["content"].lower()
    assert write_result["effect_disposition"] == "unknown"
    assert "unknown" in write_result["content"].lower()


def test_strip_dangling_tool_call_tail_preserves_answered_pair():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("contents")]
    out = strip_dangling_tool_call_tail(history)
    assert out == history  # answered -> untouched


def test_strip_interrupted_tool_tails_removes_interrupted_read_only_block():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("[Command interrupted]")]
    out = strip_interrupted_tool_tails(history)
    assert out == [_user("hi")]


def test_interrupted_side_effect_is_preserved_as_unknown():
    history = [_user("hi"), _assistant_tc("terminal"), _tool("[Command interrupted]")]

    out = strip_interrupted_tool_tails(history)

    assert out[:-1] == history[:-1]
    assert out[-1]["role"] == "tool"
    assert out[-1]["effect_disposition"] == "unknown"


def test_strip_interrupted_tool_tails_preserves_successful_block():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("ok"),
               {"role": "assistant", "content": "done"}]
    out = strip_interrupted_tool_tails(history)
    assert out == history


def test_strip_interrupted_tool_tails_removes_orphan_interrupted_tool():
    history = [_user("hi"), _tool("[Command interrupted] exit_code: 130 interrupt")]
    out = strip_interrupted_tool_tails(history)
    assert out == [_user("hi")]


def test_sanitize_replay_history_combines_both():
    # interrupted block is removed; a dangling read-only call is safe to erase
    history = [
        _user("first"),
        _assistant_tc("terminal"), _tool("[Command interrupted]"),
        _user("second"),
        _assistant_tc("read_file"),  # dangling
    ]
    out = sanitize_replay_history(history)
    assert out[:2] == [
        _user("first"),
        _assistant_tc("terminal"),
    ]
    assert out[2]["effect_disposition"] == "unknown"
    assert out[-1] == _user("second")


def test_sanitize_replay_history_noop_on_clean_history():
    history = [_user("hi"), {"role": "assistant", "content": "hello"}]
    assert sanitize_replay_history(history) == history


def test_sanitize_replay_history_empty():
    assert sanitize_replay_history([]) == []


# --- strip_incomplete_reasoning_tail (hidden-reasoning-only incomplete loop) ---
#
# When a Codex turn exhausts its continuation retries it ends with an
# assistant message carrying finish_reason=="incomplete" and NO visible answer
# (only hidden reasoning). The gateway deliberately keeps that turn OUT of the
# persisted transcript, but the cached agent still holds it in its live
# _session_messages. The FTS-corruption guard then resurrects the live
# transcript when disk lagged and replays that poisoned tail, seeding another
# incomplete loop. This stripper removes it before provider continuation.

_INCOMPLETE_ASSISTANT = {
    "role": "assistant",
    "content": "",
    "reasoning": "let me think about this",
    "finish_reason": "incomplete",
}


def _nudge():
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE

    return {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE}


def test_strip_incomplete_reasoning_tail_removes_hidden_reasoning_only_tail():
    history = [_user("real question"), dict(_INCOMPLETE_ASSISTANT)]
    out = strip_incomplete_reasoning_tail(history)
    assert out == [_user("real question")]


def test_strip_incomplete_reasoning_tail_removes_interleaved_nudges_and_retries():
    history = [
        _user("real question"),
        dict(_INCOMPLETE_ASSISTANT),
        _nudge(),
        dict(_INCOMPLETE_ASSISTANT),
        _nudge(),
        dict(_INCOMPLETE_ASSISTANT),
    ]
    out = strip_incomplete_reasoning_tail(history)
    assert out == [_user("real question")]


def test_strip_incomplete_reasoning_tail_preserves_visible_incomplete_answer():
    # A partial-but-VISIBLE answer must never be discarded, even if the turn
    # was marked incomplete: the user should still receive that text.
    visible = {
        "role": "assistant",
        "content": "Here is a partial answer",
        "finish_reason": "incomplete",
    }
    history = [_user("q"), visible]
    assert strip_incomplete_reasoning_tail(history) == history


def test_strip_incomplete_reasoning_tail_preserves_completed_answer():
    history = [
        _user("q"),
        {"role": "assistant", "content": "done", "finish_reason": "stop"},
    ]
    assert strip_incomplete_reasoning_tail(history) == history


def test_strip_incomplete_reasoning_tail_only_touches_the_tail():
    # A completed assistant answer earlier in the history is a hard stop:
    # nothing before the last real turn is removed.
    history = [
        _user("q1"),
        {"role": "assistant", "content": "answer 1", "finish_reason": "stop"},
        _user("q2"),
        dict(_INCOMPLETE_ASSISTANT),
    ]
    out = strip_incomplete_reasoning_tail(history)
    assert out == history[:3]


def test_strip_incomplete_reasoning_tail_ignores_incomplete_tool_call_turn():
    # An assistant turn that issued tool_calls is not "reasoning only": leave
    # it for the tool-tail strippers, don't erase it here.
    tc_turn = {
        "role": "assistant",
        "content": "",
        "finish_reason": "incomplete",
        "tool_calls": [
            {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    }
    history = [_user("q"), tc_turn]
    assert strip_incomplete_reasoning_tail(history) == history


def test_strip_incomplete_reasoning_tail_noop_and_identity():
    clean = [
        _user("hi"),
        {"role": "assistant", "content": "hey", "finish_reason": "stop"},
    ]
    assert strip_incomplete_reasoning_tail(clean) is clean
    assert strip_incomplete_reasoning_tail([]) == []


def test_nudge_prefix_stays_in_sync_with_conversation_loop():
    # The stripper matches the nudge by prefix (no import) to avoid a
    # conversation_loop <-> replay_cleanup cycle; guard against drift.
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE
    from agent.replay_cleanup import _CODEX_INCOMPLETE_NUDGE_PREFIX

    assert _CODEX_INCOMPLETE_NUDGE.startswith(_CODEX_INCOMPLETE_NUDGE_PREFIX)


# --- structured-list content (vision / post-compaction turns) ---
#
# After a vision turn or post-compaction rewrite, an assistant message's
# ``content`` is a structured parts list (``[{"type": "text", "text": ...}]``
# or bare strings) instead of a plain string. The hidden-reasoning-only
# detector routes that through _has_visible_text, so the stripper must treat a
# list carrying no visible text exactly like an empty string tail (strip it),
# and a list carrying any visible text like a partial answer (keep it).


def _incomplete_with_content(content):
    return {
        "role": "assistant",
        "content": content,
        "reasoning": "internal only",
        "finish_reason": "incomplete",
    }


def test_incomplete_tail_with_empty_structured_list_is_stripped():
    history = [_user("q"), _incomplete_with_content([])]
    assert strip_incomplete_reasoning_tail(history) == [_user("q")]


def test_incomplete_tail_with_only_nontext_parts_is_stripped():
    # A parts list holding only non-text (e.g. reasoning/image) entries carries
    # no user-visible answer, so it is a hidden-reasoning-only tail.
    history = [
        _user("q"),
        _incomplete_with_content([{"type": "reasoning", "text": "thinking"}]),
    ]
    assert strip_incomplete_reasoning_tail(history) == [_user("q")]


def test_incomplete_tail_with_blank_text_part_is_stripped():
    # A text part whose text is empty/whitespace is not visible content.
    history = [
        _user("q"),
        _incomplete_with_content([{"type": "text", "text": "   "}]),
    ]
    assert strip_incomplete_reasoning_tail(history) == [_user("q")]


def test_incomplete_tail_with_visible_text_part_is_preserved():
    # A visible text part in the structured list is a partial answer the user
    # should still receive, so it must never be stripped.
    history = [
        _user("q"),
        _incomplete_with_content([{"type": "text", "text": "partial answer"}]),
    ]
    assert strip_incomplete_reasoning_tail(history) == history


def test_incomplete_tail_with_bare_string_part_is_preserved():
    # Bare non-empty strings in the parts list also count as visible content.
    history = [_user("q"), _incomplete_with_content(["visible via bare string"])]
    assert strip_incomplete_reasoning_tail(history) == history
