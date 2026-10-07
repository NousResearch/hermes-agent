"""Invariant: every rough token estimate in the tree derives from ``estimate_tokens_rough`` /
``CHARS_PER_TOKEN`` in ``agent/model_metadata.py``, so the ``/context`` breakdown's static
categories, the conversation slice and native-compaction retention agree on non-Latin text.
"""

from agent.context_breakdown import _bytes_to_tokens, _chars_to_tokens
from agent.model_metadata import CHARS_PER_TOKEN, estimate_tokens_rough
from agent.native_compaction import _approx_tokens


CYRILLIC = "Привет мир, это проверка оценки токенов. " * 40
CJK = "これは日本語のテキストです。" * 40


def test_breakdown_and_retention_use_the_canonical_estimator():
    for text in (CYRILLIC, CJK, "plain ascii text " * 40):
        canonical = estimate_tokens_rough(text)
        assert _chars_to_tokens(text) == canonical
        assert _approx_tokens(text) == canonical
    # The old chars//4 shape under-counted these by ~2x; the canonical must not.
    assert _chars_to_tokens(CYRILLIC) > (len(CYRILLIC) + 3) // 4 * 1.5
    assert _chars_to_tokens(CJK) >= len(CJK)


def test_byte_and_ratio_consumers_share_one_constant():
    from agent.context_compressor import _CHARS_PER_TOKEN as compressor_ratio
    from tools.budget_config import _CHARS_PER_TOKEN as budget_ratio
    from tools.transcription_command import _PROMPT_CHARS_PER_TOKEN as whisper_ratio

    assert compressor_ratio is budget_ratio is whisper_ratio is CHARS_PER_TOKEN
    assert _bytes_to_tokens(CHARS_PER_TOKEN * 10) == 10
    assert _bytes_to_tokens(None) is None
