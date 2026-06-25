"""Verify input-token attribution targets only the user/tool prompt tail.

Confirms the conversation_loop invariant (feedback B): the row that receives
a call's packed (total_input, cache_read) is messages[-1] AND only when it is
a user/tool message — never an assistant/system tail, and never clobbering an
already-packed row.
"""
from types import SimpleNamespace

from hermes_token_codec import (
    attribute_input_tokens_to_prompt_tail as _attribute_input_tokens_to_prompt_tail,
    unpack_token_count,
    pack_input_tokens,
)


def _usage(prompt=5000, cache=2000):
    return SimpleNamespace(prompt_tokens=prompt, cache_read_tokens=cache)


def test_packs_onto_user_tail():
    messages = [{"role": "user", "content": "hi"}]
    updated = _attribute_input_tokens_to_prompt_tail(messages, _usage())
    assert updated is messages[-1]
    assert messages[-1]["token_count"] < 0
    assert unpack_token_count(messages[-1]["token_count"]) == {
        "total_input_tokens": 5000, "cache_read_tokens": 2000,
    }


def test_packs_onto_tool_tail():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{}]},
        {"role": "tool", "content": "result"},
    ]
    updated = _attribute_input_tokens_to_prompt_tail(messages, _usage(8000, 6000))
    assert updated is messages[-1]
    assert unpack_token_count(messages[-1]["token_count"])["total_input_tokens"] == 8000


def test_skips_assistant_tail():
    # An assistant-prefill tail must NOT receive input attribution.
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "pre"}]
    assert _attribute_input_tokens_to_prompt_tail(messages, _usage()) is None
    assert "token_count" not in messages[-1]


def test_skips_system_tail():
    messages = [{"role": "system", "content": "sys"}]
    assert _attribute_input_tokens_to_prompt_tail(messages, _usage()) is None
    assert "token_count" not in messages[-1]


def test_does_not_clobber_already_packed_row():
    prepacked = pack_input_tokens(111, 22)
    messages = [{"role": "tool", "content": "r", "token_count": prepacked}]
    assert _attribute_input_tokens_to_prompt_tail(messages, _usage()) is None
    assert messages[-1]["token_count"] == prepacked


def test_overwrites_legacy_nonnegative_token_count():
    # A legacy non-negative count is safe to replace with packed input data.
    messages = [{"role": "user", "content": "q", "token_count": 1234}]
    updated = _attribute_input_tokens_to_prompt_tail(messages, _usage())
    assert updated is messages[-1]
    assert messages[-1]["token_count"] < 0


def test_empty_and_non_dict_tail():
    assert _attribute_input_tokens_to_prompt_tail([], _usage()) is None
    assert _attribute_input_tokens_to_prompt_tail(["not a dict"], _usage()) is None
