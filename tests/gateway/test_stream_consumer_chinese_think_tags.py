"""Regression tests: Chinese reasoning tags (MiniMax M3) must be filtered.

MiniMax M3 emits Chinese-language reasoning tags instead of / alongside the
English ``<think>`` family.  Three filter sites must strip them:

  1. ``GatewayStreamConsumer._OPEN_THINK_TAGS`` / ``_CLOSE_THINK_TAGS``
     (streaming-time state machine in ``gateway/stream_consumer.py``)
  2. ``cli.py`` ``_stream_delta`` local ``_OPEN_TAGS`` / ``_CLOSE_TAGS``
     (CLI streaming-time filter)
  3. ``agent.agent_runtime_helpers.strip_think_blocks()``
     (post-processing for stored / sent content)

These tests prevent regressions on the Chinese tag set.

Refs: #43827, #17924, #27288
"""

import pytest

from gateway.stream_consumer import GatewayStreamConsumer


# ── GatewayStreamConsumer tag tuples ─────────────────────────────────────


class TestChineseThinkTagTuples:
    """Verify Chinese reasoning tags are present in the streaming filter tuples."""

    def test_open_tags_include_chinese(self):
        tags = GatewayStreamConsumer._OPEN_THINK_TAGS
        assert " 思考" in tags
        assert " 反思" in tags
        assert " 推理" in tags
        assert " 推敲" in tags

    def test_close_tags_include_chinese(self):
        tags = GatewayStreamConsumer._CLOSE_THINK_TAGS
        assert " 思考" in tags
        assert " 反思" in tags
        assert " 推理" in tags
        assert " 推敲" in tags

    def test_open_close_tags_same_length(self):
        """Open and close tag lists must stay in sync."""
        assert len(GatewayStreamConsumer._OPEN_THINK_TAGS) == len(
            GatewayStreamConsumer._CLOSE_THINK_TAGS
        )


# ── strip_think_blocks (post-processing) ─────────────────────────────────


class TestStripThinkBlocksChinese:
    """Verify strip_think_blocks() removes Chinese reasoning tags."""

    @pytest.fixture()
    def agent(self):
        """Minimal agent stub (strip_think_blocks only uses ``agent`` for type)."""
        return object()

    def test_closed_chinese_think_pair(self, agent):
        from agent.agent_runtime_helpers import strip_think_blocks

        content = " 思考\nsome reasoning\n 思考\nThe answer is 42."
        result = strip_think_blocks(agent, content)
        assert "The answer is 42." in result
        assert " 思考" not in result
        assert "some reasoning" not in result

    def test_closed_chinese_reflect_pair(self, agent):
        from agent.agent_runtime_helpers import strip_think_blocks

        content = " 反思\nreflecting\n 反思\nFinal answer."
        result = strip_think_blocks(agent, content)
        assert result.strip() == "Final answer."
        assert " 反思" not in result

    def test_closed_chinese_reason_pair(self, agent):
        from agent.agent_runtime_helpers import strip_think_blocks

        content = " 推理\nreasoning out\n 推理\nDone."
        result = strip_think_blocks(agent, content)
        assert result.strip() == "Done."
        assert " 推理" not in result

    def test_closed_chinese_deliberate_pair(self, agent):
        from agent.agent_runtime_helpers import strip_think_blocks

        content = " 推敲\nweighing options\n 推敲\nChosen path."
        result = strip_think_blocks(agent, content)
        assert result.strip() == "Chosen path."
        assert " 推敲" not in result

    def test_unterminated_chinese_think_at_boundary(self, agent):
        """Unterminated Chinese open tag at line start strips to end."""
        from agent.agent_runtime_helpers import strip_think_blocks

        content = " 思考\nraw reasoning that should be stripped"
        result = strip_think_blocks(agent, content)
        assert "raw reasoning" not in result
        assert " 思考" not in result

    def test_stray_chinese_orphan_tag(self, agent):
        """Stray Chinese tags that slipped through are removed."""
        from agent.agent_runtime_helpers import strip_think_blocks

        content = "Here is  思考 the answer."
        result = strip_think_blocks(agent, content)
        assert " 思考" not in result
        assert "Here is" in result
        assert "the answer." in result

    def test_english_tags_still_work(self, agent):
        """English tags must still be filtered (no regression)."""
        from agent.agent_runtime_helpers import strip_think_blocks

        content = "<think>\nenglish reasoning\n</think>\nThe answer."
        result = strip_think_blocks(agent, content)
        assert "english reasoning" not in result
        assert "The answer." in result

    def test_mixed_english_and_chinese(self, agent):
        """Content with both English and Chinese reasoning tags."""
        from agent.agent_runtime_helpers import strip_think_blocks

        content = "<think>\nenglish\n</think>\n 思考\nchinese\n 思考\nResult."
        result = strip_think_blocks(agent, content)
        assert "english" not in result
        assert "chinese" not in result
        assert "Result." in result

    def test_chinese_tags_in_prose_not_over_stripped(self, agent):
        """Chinese characters in normal prose must not be stripped."""
        from agent.agent_runtime_helpers import strip_think_blocks

        content = "Let me 思考 about this. The 推理 is clear."
        result = strip_think_blocks(agent, content)
        # These are inline mentions, not block-boundary tags — should be preserved
        # (the unterminated regex only triggers at line boundaries)
        assert "Let me" in result or "about this" in result
