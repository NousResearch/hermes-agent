"""Regression coverage for zero-user compaction integrity (#64539)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    ContextCompressor,
    _NO_USER_TASK_SENTINEL,
)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _valid_zero_user_summary(label: str = "Checked artifacts.") -> str:
    return f"""{HISTORICAL_TASK_HEADING}
{_NO_USER_TASK_SENTINEL}

## Goal
Historical cron work only.

## Completed Actions
1. {label}

## Resolved Questions
None. No user-authored questions exist.

## Historical Pending User Asks
None. No user-authored requests exist.
"""


def _assistant_tool_turns(start: int, count: int) -> list[dict]:
    turns: list[dict] = []
    for idx in range(start, start + count):
        turns.extend(
            [
                {
                    "role": "assistant",
                    "content": "Continuing scheduled work in English.",
                    "tool_calls": [
                        {
                            "id": f"call-{idx}",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{idx}",
                    "content": "/workspace/project\n" + ("x" * 300),
                },
            ]
        )
    return turns


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=0,
            protect_last_n=2,
            quiet_mode=True,
        )
    instance.tail_token_budget = 80
    return instance


def test_generate_summary_rejects_fabricated_user_ask(compressor):
    fabricated = f"""{HISTORICAL_TASK_HEADING}
User asked: 'Waar zijn de bestanden gedownload?'

## Goal
Vind de bestanden.
"""

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(fabricated),
    ):
        result = compressor._generate_summary(_assistant_tool_turns(0, 2))

    assert result is None
    assert compressor._previous_summary is None
    assert "invented user attribution" in compressor._last_summary_error


def test_zero_user_prompt_anchors_source_language_and_exact_sentinel(compressor):
    captured_prompt = ""

    def fake_call_llm(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][0]["content"]
        return _response(_valid_zero_user_summary())

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        result = compressor._generate_summary(_assistant_tool_turns(0, 2))

    assert result == f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary().strip()}"
    assert "dominant language of the source turns" in captured_prompt
    assert _NO_USER_TASK_SENTINEL in captured_prompt
    assert "Do not write \"User asked:\"" in captured_prompt


def test_zero_user_provenance_survives_iterative_compaction(compressor):
    messages = _assistant_tool_turns(0, 12)
    first_summary = f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary('First pass').strip()}"

    with patch.object(compressor, "_generate_summary", return_value=first_summary):
        first = compressor.compress(messages, current_tokens=90_000)

    first_handoffs = [
        message
        for message in first
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(first_handoffs) == 1
    assert first_handoffs[0][COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        resumed = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=0,
            protect_last_n=2,
            quiet_mode=True,
        )
    resumed.tail_token_budget = 80
    # SessionDB persists the summary content/role but not arbitrary internal
    # message keys. Simulate that round trip: the exact sentinel must recover
    # false provenance even when both in-process metadata keys are absent.
    persisted_handoff = dict(first_handoffs[0])
    persisted_handoff.pop(COMPRESSED_SUMMARY_METADATA_KEY)
    persisted_handoff.pop(COMPRESSED_SUMMARY_HAS_USER_TURN_KEY)
    second_input = [persisted_handoff, *_assistant_tool_turns(20, 12)]

    def assert_provenance_then_summarize(*_args, **_kwargs):
        assert resumed._summary_has_user_turn is False
        return f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary('Second pass').strip()}"

    with patch.object(
        resumed,
        "_generate_summary",
        side_effect=assert_provenance_then_summarize,
    ):
        second = resumed.compress(second_input, current_tokens=90_000)

    second_handoffs = [
        message
        for message in second
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(second_handoffs) == 1
    assert second_handoffs[0][COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False


def test_zero_user_deterministic_fallback_uses_same_provenance(compressor):
    messages = _assistant_tool_turns(0, 12)

    with patch.object(compressor, "_generate_summary", return_value=None):
        result = compressor.compress(messages, current_tokens=90_000)

    handoff = next(
        message
        for message in result
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert _NO_USER_TASK_SENTINEL in handoff["content"]
    assert "User asked:" not in handoff["content"]
    assert handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False


def test_real_user_turn_sets_provenance_true(compressor):
    messages = [
        {"role": "user", "content": "Please inspect the build artifacts."},
        *_assistant_tool_turns(0, 12),
    ]
    summary = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'Please inspect the build artifacts.'"

    with patch.object(compressor, "_generate_summary", return_value=summary):
        result = compressor.compress(messages, current_tokens=90_000)

    handoff = next(
        message
        for message in result
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is True


def test_session_boundaries_clear_summary_provenance(compressor):
    compressor._summary_has_user_turn = False
    compressor.on_session_reset()
    assert compressor._summary_has_user_turn is None

    compressor._summary_has_user_turn = True
    compressor.on_session_end("cron-session", [])
    assert compressor._summary_has_user_turn is None
