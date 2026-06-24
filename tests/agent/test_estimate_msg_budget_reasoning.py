"""Regression for #51800: the tail-protection budget walk must count
provider-visible reasoning fields (reasoning_content / reasoning_details /
reasoning), not only content + tool-call envelopes.

Persisted reasoning is re-sent on every request, so omitting it let
reasoning-heavy turns (e.g. DeepSeek reasoning models) undercount by the entire
reasoning payload — an oversized recent tail then appeared to fit inside the
budget and compaction summarized almost nothing.
"""

from agent.context_compressor import _CHARS_PER_TOKEN, _estimate_msg_budget_tokens

_BIG = "z" * 4000  # ~1000 tokens at _CHARS_PER_TOKEN


def _base():
    return _estimate_msg_budget_tokens({"role": "assistant", "content": "hi"})


def test_reasoning_content_counted():
    got = _estimate_msg_budget_tokens(
        {"role": "assistant", "content": "hi", "reasoning_content": _BIG}
    )
    assert got >= _base() + (len(_BIG) // _CHARS_PER_TOKEN)


def test_reasoning_details_and_reasoning_counted():
    for key in ("reasoning_details", "reasoning"):
        got = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "hi", key: _BIG}
        )
        assert got >= _base() + (len(_BIG) // _CHARS_PER_TOKEN), key


def test_non_string_reasoning_counted_without_crashing():
    # reasoning_details frequently arrives as a list/dict, not a str.
    msg = {
        "role": "assistant",
        "content": "hi",
        "reasoning_details": [{"type": "reasoning.text", "text": "z" * 2000}],
    }
    assert _estimate_msg_budget_tokens(msg) > _base()


def test_absent_or_empty_reasoning_adds_nothing():
    base_msg = {"role": "user", "content": "hello world"}
    assert _estimate_msg_budget_tokens({**base_msg, "reasoning_content": ""}) == (
        _estimate_msg_budget_tokens(base_msg)
    )
    assert _estimate_msg_budget_tokens({**base_msg, "reasoning_content": None}) == (
        _estimate_msg_budget_tokens(base_msg)
    )
