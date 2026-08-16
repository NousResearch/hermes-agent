from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.model_metadata import (
    _is_cjk_token_dense_char,
    estimate_messages_tokens_rough,
    estimate_tokens_rough,
)




def test_message_estimate_counts_korean_content_as_token_dense():
    messages = [{"role": "user", "content": "압축 테스트 " + ("가" * 1000)}]

    assert estimate_messages_tokens_rough(messages) >= 1000




def test_cjk_tail_does_not_expand_to_english_char_budget():
    with patch("agent.context_compressor.get_model_context_length", return_value=65536):
        compressor = ContextCompressor(
            "test/model",
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.2,
            quiet_mode=True,
        )
        # Resolve while the mock is active (lazy init, #32221).
        _ = compressor.context_length

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


def _reference_per_char_estimate(text: str) -> int:
    """Per-character reference: CJK ~1 token/char, everything else UTF-8
    bytes/4 (the byte width corrects Cyrillic/Greek/Arabic under-counting)."""
    dense = 0
    sparse_bytes = 0
    for ch in text:
        if _is_cjk_token_dense_char(ch):
            dense += 1
        else:
            sparse_bytes += len(ch.encode("utf-8"))
    return dense + ((sparse_bytes + 3) // 4)


def test_perf_gated_estimator_matches_per_char_reference():
    samples = [
        "",
        "ab",
        "a" * 400,
        "가" * 400,
        "압축 테스트 " + ("가" * 1000),
        "café résumé naïve",  # non-ASCII, no CJK
        "hello 안녕 world",
        "ｱｲｳｴｵ ﾃｽﾄ",  # halfwidth kana (fullwidth-forms block)
        "漢字とかな交じり文です。",
        "русский текст",  # Cyrillic — non-ASCII, non-CJK
    ]
    for text in samples:
        assert estimate_tokens_rough(text) == _reference_per_char_estimate(text), repr(text)




def test_cyrillic_counts_by_utf8_bytes():
    # «русский текст» = 12 Cyrillic chars (2 bytes each) + 1 ASCII space:
    # 25 bytes -> ceil(25/4) = 7 tokens; the old chars/4 rule said 4 —
    # the ~2x under-count that let real prompts ride the context ceiling.
    from agent.model_metadata import estimate_tokens_rough
    assert estimate_tokens_rough("русский текст") == 7
    # Pure ASCII unchanged.
    assert estimate_tokens_rough("a" * 400) == 100
