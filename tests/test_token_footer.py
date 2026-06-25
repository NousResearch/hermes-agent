"""gateway.token_footer.build_token_line — decoded per-reply token footer."""
from gateway.token_footer import build_token_line
from hermes_token_codec import pack_input_tokens, pack_assistant_tokens


def test_user_and_assistant_packed():
    res = {"messages": [
        {"role": "user", "content": "q", "token_count": pack_input_tokens(1520, 890)},
        {"role": "assistant", "content": "a", "token_count": pack_assistant_tokens(234, 128)},
    ]}
    assert build_token_line(res) == "📊 in:1520 out:234 reason:128 cache:890"


def test_tool_prompt_tail_used_for_input():
    # Multi-turn: input attribution lands on the tool row; a trailing NULL
    # user row must not shadow it.
    res = {"messages": [
        {"role": "assistant", "content": "", "token_count": pack_assistant_tokens(5, 0)},
        {"role": "tool", "content": "r", "token_count": pack_input_tokens(8000, 6000)},
        {"role": "assistant", "content": "final", "token_count": pack_assistant_tokens(300, 40)},
    ]}
    assert build_token_line(res) == "📊 in:8000 out:300 reason:40 cache:6000"


def test_null_user_row_falls_back_to_last_prompt_tokens():
    res = {"last_prompt_tokens": 999, "messages": [
        {"role": "user", "content": "q"},  # NULL token_count
        {"role": "assistant", "content": "a", "token_count": pack_assistant_tokens(10, 0)},
    ]}
    assert build_token_line(res) == "📊 in:999 out:10 reason:0 cache:0"


def test_legacy_assistant_output_only():
    res = {"messages": [
        {"role": "assistant", "content": "a", "token_count": 1234},  # legacy non-negative
    ]}
    assert build_token_line(res) == "📊 in:0 out:1234 reason:0 cache:0"


def test_empty_returns_blank():
    assert build_token_line({"messages": []}) == ""
    assert build_token_line({}) == ""
