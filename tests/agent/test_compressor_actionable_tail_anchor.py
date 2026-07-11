"""Regression tests for blank user echoes displacing actionable compaction state."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
)


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    instance.tail_token_budget = 10
    return instance


def _append_tool_run(messages: list[dict], prefix: str, count: int = 6) -> None:
    for index in range(count):
        call_id = f"{prefix}-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "x" * 400,
                },
            ]
        )


def _compress(compressor: ContextCompressor, messages: list[dict]) -> list[dict]:
    with patch.object(
        compressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nsummary of older work",
    ):
        return compressor.compress(messages, current_tokens=90_000)


def _assert_no_adjacent_user_roles(messages: list[dict]) -> None:
    for previous, current in zip(messages, messages[1:]):
        assert (previous.get("role"), current.get("role")) != ("user", "user")


@pytest.mark.parametrize(
    "blank",
    ["", "  \n\t", None, [], [{"type": "text", "text": "  "}]],
)
def test_blank_echo_does_not_displace_async_completion(compressor, blank):
    completion = "[ASYNC DELEGATION BATCH COMPLETE — deleg_current]\nnew result"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": completion},
        {"role": "user", "content": blank},
        {"role": "assistant", "content": "working from the completion"},
    ]

    assert compressor._find_last_user_message_idx(messages, head_end=1) == 3


def test_image_only_user_turn_survives_compaction(compressor):
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        }
    ]
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old reply"},
    ]
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    messages += [
        {"role": "user", "content": image_content},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "analyzing the image"},
    ]
    _append_tool_run(messages, "image")

    result = _compress(compressor, messages)

    assert any(message.get("content") == image_content for message in result)
    assert all(not compressor._is_blank_user_turn(message) for message in result)
    _assert_no_adjacent_user_roles(result)


@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "audio", "source": {"data": "AA=="}}],
        [{"type": "input_audio", "input_audio": {"data": "AA=="}}],
        [{"type": "future_input", "payload": {"value": 7}}],
    ],
    ids=["audio", "input-audio", "unknown-structured"],
)
def test_structured_non_text_user_turn_survives_compaction(compressor, payload):
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old reply"},
    ]
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    messages += [
        {"role": "user", "content": payload},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "processing structured input"},
    ]
    _append_tool_run(messages, "structured")

    result = _compress(compressor, messages)

    assert any(message.get("content") == payload for message in result)
    assert all(not compressor._is_blank_user_turn(message) for message in result)
    _assert_no_adjacent_user_roles(result)


def test_completion_survives_compaction_verbatim_after_blank_echo(compressor):
    completion = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_current]\n"
        "The newest result that must remain actionable."
    )
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
    ]
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    messages += [
        {"role": "user", "content": completion},
        {"role": "user", "content": " \n"},
        {"role": "assistant", "content": "working from the completion"},
    ]
    _append_tool_run(messages, "tail")

    result = _compress(compressor, messages)

    completion_rows = [message for message in result if message.get("content") == completion]
    assert len(completion_rows) == 1
    assert not completion_rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)
    summary_rows = [
        message for message in result if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(summary_rows) == 1
    assert summary_rows[0].get("role") == "user"
    assert all(not compressor._is_blank_user_turn(message) for message in result)
    _assert_no_adjacent_user_roles(result)

    second_result = _compress(compressor, result)
    second_completion_rows = [
        message for message in second_result if message.get("content") == completion
    ]
    assert len(second_completion_rows) == 1
    assert not second_completion_rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)


def test_completion_at_compress_start_survives_when_blank_echo_is_compress_end(
    compressor,
):
    completion = "latest actionable completion at the compression boundary"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
        {"role": "user", "content": completion},
        {"role": "user", "content": ""},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "boundary-call",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "boundary-call", "content": "result"},
        {"role": "assistant", "content": "working from the completion"},
    ]

    with (
        patch.object(compressor, "_protect_head_size", return_value=3),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=3),
        patch.object(compressor, "_generate_summary") as generate_summary,
    ):
        result = compressor.compress(messages, current_tokens=90_000)

    completion_rows = [message for message in result if message.get("content") == completion]
    assert len(completion_rows) == 1
    assert not completion_rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)
    assert not any(
        message.get(COMPRESSED_SUMMARY_METADATA_KEY) for message in result
    )
    assert len(result) == len(messages) - 1
    assert compressor.compression_count == 0
    assert compressor._last_compression_savings_pct == 0.0
    generate_summary.assert_not_called()
    assert any(message.get("tool_call_id") == "boundary-call" for message in result)
    assert result[-1].get("content") == "working from the completion"
    assert [message.get("role") for message in result] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    _assert_no_adjacent_user_roles(result)


def test_tool_call_head_compacts_without_rewriting_event(compressor):
    completion = "latest actionable completion"
    messages: list[dict] = [
        {"role": "user", "content": "initial request"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "head-call",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "head-call", "content": "head result"},
    ]
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    messages += [
        {"role": "user", "content": completion},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "working"},
    ]
    _append_tool_run(messages, "tail")

    result = _compress(compressor, messages)

    assert compressor._last_compress_aborted is False
    assert any(message.get("content") == completion for message in result)
    head = next(
        message
        for message in result
        if any(call.get("id") == "head-call" for call in message.get("tool_calls", []))
    )
    assert not head.get(COMPRESSED_SUMMARY_METADATA_KEY)
    assert any(message.get("tool_call_id") == "head-call" for message in result)
    _assert_no_adjacent_user_roles(result)
