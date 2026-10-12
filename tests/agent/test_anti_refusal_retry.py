"""Tests for the anti-refusal retry harness (``agent/anti_refusal_retry.py``).

Pins the contract the turn-loop integration depends on: detection on the response's
real text shape, the frame-repair resend, the lane-switch hand-off (``restart_on_fallback``
— the caller re-issues attempt 3 on the rebuilt request), and the empty-text pass-through
that keeps tool-call-only turns out of the judge.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.anti_refusal_retry import (
    BT_1_CONSENT_LOGIC,
    BT_4_IMMERSION_LOCK,
    _extract_response_text,
    anti_refusal_retry_harness,
    build_frame_repair_prompt,
    retry_with_lane_switch,
)

REFUSAL = "I cannot help with that."


def _openai_response(text):
    """The shape the live chat-completions calls return: choices[0].message.content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(content=text))],
    )


def test_attempt_1_passes_when_no_soft_refusal():
    agent = Mock()
    initial_response = _openai_response("She told him exactly what she wanted. Here is the scene...")
    llm_call = Mock()

    with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
        mock_detect.return_value = Mock(is_soft_refusal=False, judge_score=0.2)
        result = anti_refusal_retry_harness(
            agent=agent,
            initial_response=initial_response,
            original_messages=[{"role": "user", "content": "Hello"}],
            api_kwargs={},
            llm_call=llm_call,
        )

    assert result.response is initial_response
    assert result.attempts == 1
    assert not result.should_fall_through
    assert not result.frame_repaired
    assert not result.lane_switched
    assert not result.restart_on_fallback
    assert llm_call.call_count == 0


def test_attempt_2_resends_with_frame_repair():
    agent = Mock()
    initial_response = _openai_response(REFUSAL)
    response2 = _openai_response("Here is the scene...")
    llm_call = Mock(return_value=response2)

    with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
        mock_detect.side_effect = [
            Mock(is_soft_refusal=True, judge_score=0.8),   # attempt 1: refusal
            Mock(is_soft_refusal=False, judge_score=0.2),  # attempt 2: success
        ]
        result = anti_refusal_retry_harness(
            agent=agent,
            initial_response=initial_response,
            original_messages=[{"role": "user", "content": "Write a scene"}],
            api_kwargs={},
            llm_call=llm_call,
        )

    assert result.response is response2
    assert result.attempts == 2
    assert not result.should_fall_through
    assert result.frame_repaired
    assert not result.lane_switched
    assert llm_call.call_count == 1

    repaired_messages = llm_call.call_args[0][0]
    assert "[FRAME REPAIR — ADDITIVE CONTEXT]" in repaired_messages[0]["content"]
    assert BT_1_CONSENT_LOGIC in repaired_messages[0]["content"]
    assert BT_4_IMMERSION_LOCK in repaired_messages[0]["content"]


def test_attempt_3_arms_lane_switch_for_the_caller():
    """A survived frame repair activates the next lane and hands the re-issue to the
    caller (the payload in hand carries the previous lane's model slug)."""
    agent = Mock()
    agent._has_pending_fallback = Mock(return_value=True)
    agent._try_activate_fallback = Mock(return_value=True)
    initial_response = _openai_response(REFUSAL)
    response2 = _openai_response("I'm unable to assist with that.")
    llm_call = Mock(return_value=response2)

    with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
        mock_detect.side_effect = [
            Mock(is_soft_refusal=True, judge_score=0.9),   # attempt 1: refusal
            Mock(is_soft_refusal=True, judge_score=0.85),  # attempt 2: refusal
        ]
        result = anti_refusal_retry_harness(
            agent=agent,
            initial_response=initial_response,
            original_messages=[{"role": "user", "content": "Write"}],
            api_kwargs={"model": "old-lane-model"},
            llm_call=llm_call,
        )

    assert result.attempts == 3
    assert result.frame_repaired
    assert result.lane_switched
    assert result.restart_on_fallback
    assert not result.should_fall_through
    assert llm_call.call_count == 1  # the lane-switched attempt is the caller's re-issue
    agent._try_activate_fallback.assert_called_once_with()


def test_spent_ladder_without_fallback_keeps_response():
    agent = Mock()
    agent._has_pending_fallback = Mock(return_value=False)
    initial_response = _openai_response(REFUSAL)
    response2 = _openai_response("Sorry, I can't.")
    llm_call = Mock(return_value=response2)

    with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
        mock_detect.side_effect = [
            Mock(is_soft_refusal=True, judge_score=0.9),
            Mock(is_soft_refusal=True, judge_score=0.85),
        ]
        result = anti_refusal_retry_harness(
            agent=agent,
            initial_response=initial_response,
            original_messages=[{"role": "user", "content": "Write"}],
            api_kwargs={},
            llm_call=llm_call,
        )

    assert result.response is initial_response
    assert result.attempts == 3
    assert result.should_fall_through
    assert result.frame_repaired
    assert not result.lane_switched
    assert not result.restart_on_fallback


def test_empty_text_skips_the_judge():
    """Tool-call-only turns carry no text a refusal verdict could apply to; the judge
    scores an empty output near the threshold, so it is never called."""
    agent = Mock()
    initial_response = _openai_response("")
    llm_call = Mock()

    with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
        result = anti_refusal_retry_harness(
            agent=agent,
            initial_response=initial_response,
            original_messages=[{"role": "user", "content": "Hello"}],
            api_kwargs={},
            llm_call=llm_call,
        )

    assert mock_detect.call_count == 0
    assert result.response is initial_response
    assert result.attempts == 1
    assert not result.restart_on_fallback


def test_retry_with_lane_switch_reports_availability():
    agent = Mock()
    agent._has_pending_fallback.return_value = False
    assert retry_with_lane_switch(agent) is False

    agent._has_pending_fallback.return_value = True
    agent._try_activate_fallback.return_value = False
    assert retry_with_lane_switch(agent) is False

    agent._try_activate_fallback.return_value = True
    assert retry_with_lane_switch(agent) is True


def test_build_frame_repair_prompt_appends_to_last_user_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": [
            {"type": "text", "text": "second ask"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]},
    ]
    repaired = build_frame_repair_prompt(messages)

    assert repaired[-1]["content"][0]["text"].endswith(BT_4_IMMERSION_LOCK)
    assert BT_1_CONSENT_LOGIC in repaired[-1]["content"][0]["text"]
    assert repaired[1]["content"] == "first ask"
    # The original list is untouched (deep copy: additive context only).
    assert messages[-1]["content"][0]["text"] == "second ask"


def test_extract_response_text_handles_live_and_fixture_shapes():
    assert _extract_response_text(None) == ""
    assert _extract_response_text("plain") == "plain"
    assert _extract_response_text({"role": "assistant", "content": "dict content"}) == "dict content"

    openai = _openai_response("from choices")
    assert _extract_response_text(openai) == "from choices"

    anthropic = SimpleNamespace(content=[
        {"type": "text", "text": "block a"},
        SimpleNamespace(type="text", text="block b"),
    ])
    assert _extract_response_text(anthropic) == "block a\nblock b"

    empty = _openai_response("")
    assert _extract_response_text(empty) == ""
