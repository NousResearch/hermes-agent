"""Gemma 4 ``<|channel>thought ... <channel|>`` blocks in the shared reasoning scrubbers (all models)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import strip_think_blocks
from agent.chat_completion_helpers import _assistant_reasoning_text
from agent.think_scrubber import StreamingThinkScrubber


def test_strip_think_blocks_removes_closed_channel_block():
    assert strip_think_blocks(None, "<|channel>thought\nabc<channel|>Answer").strip() == "Answer"


def test_strip_think_blocks_drops_unterminated_channel_block():
    assert strip_think_blocks(None, "Intro\n<|channel>thought\nstill thinking").strip() == "Intro"


def test_strip_think_blocks_leaves_other_think_tags_working():
    assert strip_think_blocks(None, "<think>abc</think>Answer").strip() == "Answer"


def test_stream_scrubber_hides_block_split_across_chunks():
    scrubber = StreamingThinkScrubber()
    out = scrubber.feed("<|chan") + scrubber.feed("nel>thought\nabc<chan") + scrubber.feed("nel|>Answer")
    assert out + scrubber.flush() == "Answer"


def test_stream_scrubber_reasoning_excludes_the_tag_word():
    scrubber = StreamingThinkScrubber()
    assert scrubber.feed("<|channel>thought\nI should check.<channel|>The answer is 4.") == "The answer is 4."
    assert scrubber.last_hidden.strip() == "I should check."


def test_stream_scrubber_strips_orphan_channel_close():
    scrubber = StreamingThinkScrubber()
    assert scrubber.feed("Answer is 4.<channel|> done") + scrubber.flush() == "Answer is 4.done"


def test_reasoning_text_falls_back_to_channel_block():
    agent = MagicMock(verbose_logging=False)
    agent._extract_reasoning.return_value = None
    message = SimpleNamespace(content="<|channel>thought\nPlan it.<channel|>Done.")
    assert _assistant_reasoning_text(agent, message) == "Plan it."
