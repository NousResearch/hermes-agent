"""Invariants for ``collapse_continuation_trail`` (length-continuation cleanup).

When a truncated response is continued, fragment / nudge rows accumulate on the
transcript. Exhaustion and the continuation ceiling both collapse that trail into
one assistant row. These tests pin the contracts in the helper's docstring
(#119001): no unanswered synthetic nudge is persisted, earlier-turn rows are never
read, and explicit ``parts`` still collapses even without a turn index.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import strip_think_blocks
from agent.turn_truncation import collapse_continuation_trail


def _agent():
    agent = SimpleNamespace(_session_messages=None, log_prefix="")
    agent._strip_think_blocks = lambda text: strip_think_blocks(agent, text)
    return agent


def _fragment(text: str) -> dict:
    return {"role": "assistant", "content": text, "_length_continuation_fragment": True}


def _nudge(text: str = "Please continue.") -> dict:
    return {"role": "user", "content": text, "_length_continuation_nudge": True}


def test_invalid_turn_index_with_parts_none_is_a_noop():
    """parts=None without a usable turn index must not mutate the transcript."""
    agent = _agent()
    messages = [
        {"role": "user", "content": "ask"},
        _fragment("orphan fragment from nowhere"),
        _nudge(),
    ]
    before = [dict(m) for m in messages]

    assert collapse_continuation_trail(agent, messages, None, finish_reason="length") == ""
    assert messages == before
    assert agent._session_messages is None


def test_valid_turn_without_trail_is_a_noop():
    """A valid turn index with no fragment/nudge rows leaves the transcript alone."""
    agent = _agent()
    messages = [
        {"role": "user", "content": "ask"},  # idx 0
        {"role": "assistant", "content": "done"},
    ]
    before = [dict(m) for m in messages]

    assert collapse_continuation_trail(agent, messages, 0, finish_reason="length") == ""
    assert messages == before


def test_nudge_only_trail_is_dropped_without_persisting_an_assistant_row():
    """An unanswered synthetic nudge must never be persisted as the turn's answer.

    parts=None + a nudge-only trail: the nudge is removed, but no assistant row is
    appended because there is no fragment text to keep.
    """
    agent = _agent()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "ask"},  # idx 1
        _nudge("Continue from where you left off."),
    ]

    assert collapse_continuation_trail(agent, messages, 1, finish_reason="length") == ""
    assert messages == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "ask"},
    ]
    assert agent._session_messages is messages


def test_fragments_and_nudges_collapse_into_one_assistant_row():
    """Fragment rows are joined; nudge rows are discarded; one assistant remains."""
    agent = _agent()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "last ask"},  # idx 3
        _fragment("Hello "),
        _nudge(),
        _fragment("world"),
    ]

    text = collapse_continuation_trail(agent, messages, 3, finish_reason="length")
    assert text == "Hello world"
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert messages[-1]["content"] == "Hello world"
    assert messages[-1]["finish_reason"] == "length"
    assert messages[-1].get("_length_continuation_fragment") is None
    assert messages[-1].get("_length_continuation_nudge") is None
    # System + last user survive the collapse.
    assert messages[0] == {"role": "system", "content": "S"}
    assert messages[3] == {"role": "user", "content": "last ask"}
    assert agent._session_messages is messages


def test_earlier_turn_fragments_are_not_read():
    """With a valid turn index, only the current turn's trail is collapsed."""
    agent = _agent()
    messages = [
        {"role": "user", "content": "first"},  # idx 0
        _fragment("SHOULD NOT APPEAR"),
        {"role": "user", "content": "second"},  # idx 2 — current turn
        _fragment("kept"),
    ]

    text = collapse_continuation_trail(agent, messages, 2, finish_reason="length")
    assert text == "kept"
    assert messages[1]["content"] == "SHOULD NOT APPEAR"
    assert messages[1].get("_length_continuation_fragment") is True
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "kept"


def test_explicit_parts_collapse_without_turn_index():
    """Continuation-ceiling path: explicit parts always append, scanning from 0."""
    agent = _agent()
    messages = [
        {"role": "user", "content": "ask"},
        _fragment("ignored-because-parts-override"),
        _nudge(),
    ]

    text = collapse_continuation_trail(
        agent, messages, None, finish_reason="length", parts=["Alpha", "Beta"],
    )
    # Bare fragments are joined with a newline (see _join_truncated_parts).
    assert text == "Alpha\nBeta"
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "ask"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Alpha\nBeta"
    assert messages[1]["finish_reason"] == "length"
    assert messages[1].get("_length_continuation_fragment") is None
    assert messages[1].get("_length_continuation_nudge") is None


def test_think_tags_are_stripped_from_collapsed_text():
    agent = _agent()
    messages = [
        {"role": "user", "content": "ask"},  # idx 0
        _fragment("<think>secret</think>Visible"),
    ]

    text = collapse_continuation_trail(agent, messages, 0, finish_reason="length")
    assert text == "Visible"
    assert messages[-1]["content"] == "Visible"
