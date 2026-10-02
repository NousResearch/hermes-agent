"""After a mid-stream transport cut the continuation must keep tools on the table (#74990)."""
import pytest

from agent import conversation_loop as cl
from agent.context_compressor import ContextCompressor


@pytest.mark.parametrize("dropped", [None, ["write_file"]])
def test_partial_stub_continuation_says_tools_remain_available(dropped):
    prompt = cl._get_continuation_prompt(True, dropped).lower()
    assert "tools" in prompt and "available" in prompt
    assert "finish the answer directly" not in prompt


@pytest.mark.parametrize("dropped", [None, ["write_file"]])
def test_new_and_legacy_stub_nudges_are_recognized_as_synthetic(dropped):
    for text in (cl._get_continuation_prompt(True, dropped), cl._LEGACY_LENGTH_CONTINUATION_NETWORK_STUB):
        assert ContextCompressor._is_synthetic_compression_user_turn({"role": "user", "content": text})
