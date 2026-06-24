"""Regression tests for iterative context-summary continuity."""

import json
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.side_effect_dedup import (
    build_send_message_record,
    extract_action_log_records,
    render_action_log_block,
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _messages_with_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_existing_previous_summary_is_not_serialized_again_as_new_turn():
    """Same-process iterative compression should not feed the old handoff twice."""
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"
    compressor._previous_summary = old_summary

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_resume_rehydrates_previous_summary_from_handoff_message():
    """After restart/resume, the persisted handoff should regain summary identity."""
    compressor = _compressor()
    old_summary = "RESUMED-SUMMARY-BODY durable continuity facts"
    assert compressor._previous_summary is None

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert "TURNS TO SUMMARIZE:" not in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_handoff_in_protected_head_populates_previous_summary_before_update():
    """A resumed protected-head handoff should restore iterative-summary state."""
    compressor = _compressor()
    old_summary = "PROTECTED-HEAD-SUMMARY durable facts from before restart"
    seen_turns = []

    def fake_generate_summary(turns_to_summarize, focus_topic=None):
        seen_turns.extend(turns_to_summarize)
        return "new summary from resumed turns"

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == "new summary from resumed turns"
    assert seen_turns
    assert all(old_summary not in str(msg.get("content", "")) for msg in seen_turns)


def test_compress_merges_send_message_action_log_records_into_updated_summary():
    compressor = _compressor()
    prior_record = {
        "tool": "send_message",
        "target": "slack:ops",
        "message_sha256": "abc123",
        "fingerprint": "send_message:slack:ops:abc123",
    }
    prior_summary = (
        "OLD-SUMMARY-BODY durable continuity facts"
        + render_action_log_block([prior_record])
    )
    send_args = {
        "action": "send",
        "target": "slack:builds",
        "message": "Build failed on main",
    }
    send_result = {
        "success": True,
        "platform": "slack",
        "chat_id": "C123",
        "message_id": "m456",
    }
    expected_new_record = build_send_message_record(send_args, send_result)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{prior_summary}"},
        {"role": "assistant", "content": "resume ack"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_send_1",
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "arguments": json.dumps(send_args),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_send_1",
            "content": json.dumps(send_result),
        },
        {"role": "user", "content": "continue after notification"},
        {"role": "assistant", "content": "latest tail reply"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]

    with (
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=7),
        patch(
            "agent.context_compressor.call_llm",
            return_value=_response("updated summary body"),
        ) as mock_call,
    ):
        compressed = compressor.compress(messages)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "OLD-SUMMARY-BODY durable continuity facts" in prompt
    assert "[HERMES_ACTION_LOG_BEGIN]" not in prompt

    summary_message = next(
        msg
        for msg in compressed
        if "updated summary body" in str(msg.get("content", ""))
    )
    records = extract_action_log_records(summary_message["content"])
    assert {record["fingerprint"] for record in records} == {
        prior_record["fingerprint"],
        expected_new_record["fingerprint"],
    }
