"""Regression tests for Entity State Tracker and contradiction detection."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


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


def _messages_with_previous_summary(summary_body: str, extra_turns: list = None):
    """Build a message list that looks like a resumed session with previous summary."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
    ]
    if extra_turns:
        messages.extend(extra_turns)
    messages.extend([
        {"role": "user", "content": "latest user turn after resume"},
        {"role": "assistant", "content": "latest assistant response"},
    ])
    return messages


class TestEntityStateTracker:
    """Tests for the Entity State Tracker template field."""

    def test_entity_state_tracker_in_first_compaction_prompt(self):
        """First compaction should include Entity State Tracker in template."""
        compressor = _compressor()
        # Need at least 6 messages (protect_first=1 + 3 tail + 1 middle minimum, but <= 5 skips)
        turns = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Download paper 09Q9ex"},
            {"role": "assistant", "content": "starting download"},
            {"role": "tool", "tool_call_id": "t1", "content": "PDF saved, 63 chunks ingested"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "one more turn to trigger compression"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=_response("summary")) as mock_call:
            result = compressor.compress(turns)

        mock_call.assert_called_once()
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "Entity State Tracker" in prompt, "Template must include Entity State Tracker field"
        assert "TURNS TO SUMMARIZE:" in prompt, "First compaction must use TURNS TO SUMMARIZE"

    def test_entity_state_tracker_in_iterative_update_prompt(self):
        """Iterative update should include Entity State Tracker + contradiction detection."""
        compressor = _compressor()
        previous = "## Active Task\nDownload papers\n\n## Completed Actions\n1. Download 09Q9ex — success [tool: terminal]"
        turns = [
            {"role": "user", "content": "Continue"},
            {"role": "tool", "tool_call_id": "t1", "content": "PDF saved, 63 chunks ingested"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
            compressor.compress(_messages_with_previous_summary(previous, turns))

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "Entity State Tracker" in prompt, "Iterative prompt must include Entity State Tracker"
        assert "PREVIOUS SUMMARY:" in prompt, "Iterative prompt must have PREVIOUS SUMMARY"
        assert "NEW TURNS TO INCORPORATE:" in prompt, "Iterative prompt must have NEW TURNS TO INCORPORATE"


class TestContradictionDetection:
    """Tests for the CONTRADICTION DETECTION instruction in iterative updates."""

    def test_contradiction_detection_present_in_iterative_prompt(self):
        """Iterative update prompt must include CONTRADICTION DETECTION block."""
        compressor = _compressor()
        previous = "## Active Task\nProcess papers\n\n## Completed Actions\n1. Download 09Q9ex — failed (HTML error) [tool: terminal]"
        turns = [
            {"role": "user", "content": "Retry download"},
            {"role": "tool", "tool_call_id": "t1", "content": "PDF re-downloaded successfully, 63 chunks ingested"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
            compressor.compress(_messages_with_previous_summary(previous, turns))

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "CONTRADICTION DETECTION" in prompt, "Iterative prompt must include CONTRADICTION DETECTION"
        assert "PREVIOUS SUMMARY" in prompt, "Must reference PREVIOUS SUMMARY for contradiction check"
        # The LLM should be instructed to mark conflicts, not silently overwrite
        assert "CONFLICT" in prompt or "silent" in prompt.lower(), \
            "Must instruct LLM to handle contradictions explicitly"

    def test_iterative_prompt_has_both_previous_and_new_turns(self):
        """Iterative update must include both PREVIOUS SUMMARY and NEW TURNS."""
        compressor = _compressor()
        unique_previous = "PREVIOUS-SUMMARY-UNIQUE-ID: abc123"
        unique_new = "NEW-TURN-UNIQUE-ID: xyz789"

        compressor._previous_summary = unique_previous
        # Build enough messages to guarantee compression is triggered
        # even after pruning: need > 5 messages (protect_first=1 + 3 tail + 1 middle minimum)
        turns = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": f"{SUMMARY_PREFIX}\n{unique_previous}"},
            {"role": "assistant", "content": "assistant 1"},
            {"role": "tool", "tool_call_id": "t1", "content": unique_new},
            {"role": "assistant", "content": "assistant 2"},
            {"role": "user", "content": "user trigger"},
            {"role": "assistant", "content": "assistant 3"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=_response("summary")) as mock_call:
            result = compressor.compress(turns)

        mock_call.assert_called_once()
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        # Previous summary must appear EXACTLY once (not twice)
        assert prompt.count(unique_previous) == 1, \
            "PREVIOUS SUMMARY content must appear exactly once in prompt"
        assert unique_new in prompt, "NEW TURNS content must appear in prompt"


class TestContentTruncationLimits:
    """Tests for the increased content truncation limits."""

    def test_content_max_increased(self):
        """Verify _CONTENT_MAX has been increased from 6000 to 10000."""
        assert ContextCompressor._CONTENT_MAX == 10000, \
            f"Expected _CONTENT_MAX=10000, got {ContextCompressor._CONTENT_MAX}"

    def test_content_head_increased(self):
        """Verify _CONTENT_HEAD has been increased from 4000 to 6000."""
        assert ContextCompressor._CONTENT_HEAD == 6000, \
            f"Expected _CONTENT_HEAD=6000, got {ContextCompressor._CONTENT_HEAD}"

    def test_content_tail_increased(self):
        """Verify _CONTENT_TAIL has been increased from 1500 to 3000."""
        assert ContextCompressor._CONTENT_TAIL == 3000, \
            f"Expected _CONTENT_TAIL=3000, got {ContextCompressor._CONTENT_TAIL}"

    def test_serialize_preserves_more_content(self):
        """Long tool results should be serialized with head+tail truncation."""
        compressor = _compressor()
        # Create a very long tool result (> 10000 chars)
        long_content = "X" * 12000
        turns = [
            {"role": "tool", "tool_call_id": "t1", "content": long_content},
        ]

        serialized = compressor._serialize_for_summary(turns)

        # Should contain truncated marker
        assert "...[truncated]..." in serialized, \
            "Long content must be truncated with head+tail pattern"
        # Should NOT contain the full content
        assert long_content not in serialized, \
            "Full content must not appear in serialized output"
        # The head portion (6000) + tail portion (3000) + marker should fit in 10000
        assert len(serialized) <= ContextCompressor._CONTENT_MAX + 50, \
            f"Serialized output should fit within _CONTENT_MAX"
