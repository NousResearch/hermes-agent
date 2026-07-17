"""Regression: _interim_assistant_visible_text must handle Anthropic
block-form content (list of thinking/text dicts), not just plain strings.

2026-07-17: after the v0.18 update, extended-thinking responses put
``content`` as a list of blocks; ``_strip_think_blocks``'s ``re.sub`` then
raised ``TypeError: expected string or bytes-like object, got 'list'`` in
the outer conversation loop on every tool round (API call #39-43 spam).
"""

import types

import pytest

from run_agent import AIAgent


@pytest.fixture
def agent():
    a = AIAgent.__new__(AIAgent)
    # _interim_assistant_visible_text needs only these two helpers.
    a._extract_codex_interim_visible_parts = types.MethodType(
        lambda self, msg: [], a
    )
    return a


def _msg(content):
    return {"role": "assistant", "content": content}


def test_plain_string_content_still_works(agent):
    assert agent._interim_assistant_visible_text(_msg("hello")) == "hello"


def test_none_content_returns_empty(agent):
    assert agent._interim_assistant_visible_text(_msg(None)) == ""


def test_block_list_content_extracts_text_blocks(agent):
    content = [
        {"type": "thinking", "thinking": "secret reasoning"},
        {"type": "text", "text": "visible answer"},
    ]
    assert agent._interim_assistant_visible_text(_msg(content)) == "visible answer"


def test_block_list_skips_thinking_and_joins_text(agent):
    content = [
        {"type": "text", "text": "part one "},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "part two"},
    ]
    assert (
        agent._interim_assistant_visible_text(_msg(content))
        == "part one part two"
    )


def test_block_list_with_raw_strings(agent):
    assert agent._interim_assistant_visible_text(_msg(["a", "b"])) == "ab"


def test_block_list_only_thinking_returns_empty(agent):
    content = [{"type": "thinking", "thinking": "only reasoning"}]
    assert agent._interim_assistant_visible_text(_msg(content)) == ""


def test_think_tags_still_stripped_from_block_text(agent):
    content = [{"type": "text", "text": "<think>hidden</think>shown"}]
    assert agent._interim_assistant_visible_text(_msg(content)) == "shown"
