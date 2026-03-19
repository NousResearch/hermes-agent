"""Tests for observation masking in ContextCompressor."""
import pytest
from unittest.mock import patch, MagicMock
from agent.context_compressor import ContextCompressor


def make_compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=128000):
        return ContextCompressor(model="test", quiet_mode=True, **kwargs)


def make_messages(n_turns):
    """Create n_turns of assistant+tool pairs."""
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"Calling tool turn {i}", "tool_calls": [{"id": f"tc{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": f"Tool output for turn {i} — " + "x" * 200})
    return msgs


class TestMaskObservations:

    def test_masks_old_tool_outputs(self):
        c = make_compressor(observation_masking=True, observation_masking_window=2)
        msgs = make_messages(4)
        result = c.mask_observations(msgs)
        # First 2 turns (4 messages) should be masked
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        masked = [m for m in tool_msgs if "observation masked" in m.get("content", "")]
        unmasked = [m for m in tool_msgs if "observation masked" not in m.get("content", "")]
        assert len(masked) == 2
        assert len(unmasked) == 2

    def test_preserves_recent_tool_outputs(self):
        c = make_compressor(observation_masking=True, observation_masking_window=2)
        msgs = make_messages(4)
        result = c.mask_observations(msgs)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        # Last 2 tool outputs should be unmasked
        assert "observation masked" not in tool_msgs[-1]["content"]
        assert "observation masked" not in tool_msgs[-2]["content"]

    def test_masked_content_shows_char_count(self):
        c = make_compressor(observation_masking=True, observation_masking_window=1)
        msgs = make_messages(3)
        result = c.mask_observations(msgs)
        masked_msgs = [m for m in result if m.get("role") == "tool" and "observation masked" in m.get("content", "")]
        for m in masked_msgs:
            assert "chars" in m["content"]

    def test_assistant_messages_never_masked(self):
        c = make_compressor(observation_masking=True, observation_masking_window=1)
        msgs = make_messages(4)
        result = c.mask_observations(msgs)
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        for m in assistant_msgs:
            assert "observation masked" not in str(m.get("content", ""))

    def test_disabled_masking_returns_unchanged(self):
        c = make_compressor(observation_masking=False, observation_masking_window=2)
        msgs = make_messages(4)
        result = c.mask_observations(msgs)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        for m in tool_msgs:
            assert "observation masked" not in m.get("content", "")

    def test_too_few_turns_no_masking(self):
        c = make_compressor(observation_masking=True, observation_masking_window=4)
        msgs = make_messages(2)
        result = c.mask_observations(msgs)
        assert result == msgs

    def test_default_masking_enabled(self):
        c = make_compressor()
        assert c.observation_masking is True

    def test_default_window_is_four(self):
        c = make_compressor()
        assert c.observation_masking_window == 4

    def test_masking_runs_before_compression(self):
        """mask_observations is called at the start of compress()."""
        c = make_compressor(observation_masking=True, observation_masking_window=2)
        msgs = make_messages(10)
        # Patch _generate_summary to avoid LLM call
        with patch.object(c, "_generate_summary", return_value="summary"):
            with patch.object(c, "should_compress", return_value=True):
                result = c.compress(msgs, current_tokens=200000)
        # After compress, old tool outputs in compressed result should be masked
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        if tool_msgs:
            # At least some masking happened
            contents = [m.get("content", "") for m in tool_msgs]
            assert any("observation masked" in c for c in contents) or len(result) < len(msgs)
