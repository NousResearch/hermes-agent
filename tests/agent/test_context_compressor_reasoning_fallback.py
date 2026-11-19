"""Reasoning-model summaries must not trip the empty-content failure path.

Local / thinking backends (DeepSeek, Qwen, Kimi) often return content=""
with the usable text in reasoning / reasoning_content. _generate_summary
should accept that via extract_content_or_reasoning, bound the fallback
so a CoT dump cannot grow the transcript, and still fail closed when
both fields are empty (#11978).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _compressor(**overrides):
    kwargs = dict(model="test/model", quiet_mode=True, tail_mode="legacy")
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(**kwargs)


def _turns():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


def test_empty_content_uses_reasoning_content():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            reasoning_content="kept reasoning summary",
        ))]
    )
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert "kept reasoning summary" in out
    assert out.startswith(SUMMARY_PREFIX)


def test_whitespace_content_falls_back_for_dict_and_object():
    class _Msg:
        content = " "
        reasoning_content = "object reasoning"

    for message, needle in (
        ({"content": " ", "reasoning_content": "dict reasoning"}, "dict reasoning"),
        (_Msg(), "object reasoning"),
    ):
        response = {"choices": [{"message": message}]}
        with patch("agent.context_compressor.call_llm", return_value=response):
            out = _compressor()._generate_summary(_turns())
        assert out is not None
        assert needle in out


def test_oversized_reasoning_fallback_is_truncated():
    reasoning = "t" * 20_000
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            reasoning_content=reasoning,
        ))]
    )
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert reasoning not in out
    assert len(out) < 15_000


def test_empty_content_without_reasoning_still_fails():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    with patch("agent.context_compressor.call_llm", return_value=mock_response):
        out = _compressor()._generate_summary(_turns())
    assert out is None
