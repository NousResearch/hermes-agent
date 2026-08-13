"""Reasoning-model summaries must not trip the empty-content failure path.

Local / thinking backends (DeepSeek, Qwen, Kimi) often return content=""
with the usable text in reasoning / reasoning_content. _generate_summary
should accept that via extract_content_or_reasoning, bound the fallback
so a CoT dump cannot grow the transcript, and still fail closed when
both fields are empty (#11978).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, HISTORICAL_TASK_HEADING, SUMMARY_PREFIX
from agent.context_compressor_continuation import (
    CURRENT_SUBTASK_HEADING,
    GOVERNING_OUTCOME_HEADING,
    LATEST_USER_CORRECTION_HEADING,
    NEXT_OUTCOME_STEP_HEADING,
)


def _valid_summary(evidence):
    # Keep the schema ahead of the reasoning excerpt so bounding it preserves the fields.
    return (
        f"{HISTORICAL_TASK_HEADING}\nUser asked: 'do something'.\n\n"
        f"{GOVERNING_OUTCOME_HEADING}\nUnknown.\n\n"
        f"{CURRENT_SUBTASK_HEADING}\nNone.\n\n"
        f"{LATEST_USER_CORRECTION_HEADING}\nNone.\n\n"
        f"{NEXT_OUTCOME_STEP_HEADING}\nNone.\n\n"
        f"## Summary Evidence\n{evidence}"
    )


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
            reasoning_content=_valid_summary("kept reasoning summary"),
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
        reasoning_content = _valid_summary("object reasoning")

    for message, needle in (
        ({"content": " ", "reasoning_content": _valid_summary("dict reasoning")}, "dict reasoning"),
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
            reasoning_content=_valid_summary(reasoning),
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
