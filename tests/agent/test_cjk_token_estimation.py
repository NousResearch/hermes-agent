from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough


def test_cjk_text_is_not_estimated_as_four_chars_per_token():
    assert estimate_tokens_rough("a" * 400) == 100
    assert estimate_tokens_rough("가" * 400) >= 400


def test_message_estimate_counts_korean_content_as_token_dense():
    messages = [{"role": "user", "content": "압축 테스트 " + ("가" * 1000)}]

    assert estimate_messages_tokens_rough(messages) >= 1000


def test_compressor_tail_budget_uses_cjk_aware_message_estimate():
    korean_msg = {"role": "assistant", "content": "가" * 2000}
    english_msg = {"role": "assistant", "content": "a" * 2000}

    assert _estimate_msg_budget_tokens(korean_msg) > _estimate_msg_budget_tokens(english_msg)


def test_cjk_tail_does_not_expand_to_english_char_budget():
    with patch("agent.context_compressor.get_model_context_length", return_value=65536):
        compressor = ContextCompressor(
            "test/model",
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.2,
            quiet_mode=True,
        )

    messages = [
        {"role": "user", "content": "head 1"},
        {"role": "assistant", "content": "head 2"},
        {"role": "user", "content": "head 3"},
    ]
    for idx in range(40):
        role = "assistant" if idx % 2 else "user"
        messages.append({"role": role, "content": "가" * 1200})

    compress_start = compressor._align_boundary_forward(
        messages,
        compressor._protect_head_size(messages),
    )
    compress_end = compressor._find_tail_cut_by_tokens(messages, compress_start)

    assert len(messages) - compress_end < 31
