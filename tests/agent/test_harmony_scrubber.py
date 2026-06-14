"""Tests for StreamingHarmonyScrubber.

Verifies the stateful suppression of Harmony tags during streaming.
"""

from __future__ import annotations

import pytest

from agent.harmony_scrubber import StreamingHarmonyScrubber


def _drive(scrubber: StreamingHarmonyScrubber, deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    out = [scrubber.feed(d) for d in deltas]
    out.append(scrubber.flush())
    return "".join(out)


class TestHarmonyScrubber:
    def test_passthrough_normal_text(self) -> None:
        s = StreamingHarmonyScrubber()
        assert _drive(s, ["Hello ", "world!"]) == "Hello world!"

    def test_single_final_channel(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["<|channel|>final<|message|>Hello world!<|end|>"]
        assert _drive(s, deltas) == "Hello world!"

    def test_final_channel_split(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["<|chan", "nel|>fi", "nal<|me", "ssage|>Hello ", "world!", "<|e", "nd|>"]
        assert _drive(s, deltas) == "Hello world!"

    def test_suppress_non_final_channel(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["<|channel|>commentary to=skill_view <|constrain|>json<|message|>{\"name\":\"hermes-agent\"}<|end|>"]
        assert _drive(s, deltas) == ""

    def test_suppress_non_final_channel_split(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["<|channel|>com", "mentary to=skill_view<|message|>", "{\"name\":", "\"hermes\"}", "<|end|>"]
        assert _drive(s, deltas) == ""

    def test_sequence_of_channels(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = [
            "<|channel|>commentary to=skill_view<|message|>{\"name\":\"hermes\"}<|end|>",
            "<|channel|>final<|message|>Done!<|end|>"
        ]
        assert _drive(s, deltas) == "Done!"

    def test_sequence_of_channels_split(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = [
            "<|channel|>commentary",
            " to=skill_view<|message|>",
            "{\"name\": \"hermes\"}<|e",
            "nd|><|channel|>final<|message|>Done!",
            "<|endoftext|>"
        ]
        assert _drive(s, deltas) == "Done!"

    def test_orphan_close_tags_stripped(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["Hello<|end|> world<|endoftext|>"]
        assert _drive(s, deltas) == "Hello world"

    def test_orphan_close_tags_split(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["Hello<|e", "nd|> world<|endof", "text|>"]
        assert _drive(s, deltas) == "Hello world"

    def test_partial_tag_flush(self) -> None:
        s = StreamingHarmonyScrubber()
        # Stream ends while holding back a partial tag prefix
        assert _drive(s, ["Hello <|chan"]) == "Hello <|chan"

    def test_case_insensitivity(self) -> None:
        s = StreamingHarmonyScrubber()
        deltas = ["<|CHANNEL|>FINAL<|MESSAGE|>Hello<|END|>"]
        assert _drive(s, deltas) == "Hello"

    def test_reset(self) -> None:
        s = StreamingHarmonyScrubber()
        s.feed("<|channel|>commentary<|message|>abc")
        assert s._state == "IN_CHANNEL_SUPPRESSED"
        s.reset()
        assert s._state == "SCANNING"
        assert _drive(s, ["hello"]) == "hello"


class TestAgentIntegration:
    class FakeAgent:
        def __init__(self):
            self._stream_harmony_scrubber = StreamingHarmonyScrubber()
            self._stream_think_scrubber = None
            self._stream_context_scrubber = None
            self._current_streamed_assistant_text = ""
            self.stream_delta_callback = self._cb
            self._stream_callback = None
            self.fired_deltas = []

        def _cb(self, text: str):
            self.fired_deltas.append(text)

        def _record_streamed_assistant_text(self, text: str):
            self._current_streamed_assistant_text += text

        def _strip_think_blocks(self, text: str) -> str:
            return text

        # Re-use the real _fire_stream_delta and _reset_stream_delivery_tracking
        from run_agent import AIAgent
        _fire_stream_delta = AIAgent._fire_stream_delta
        _reset_stream_delivery_tracking = AIAgent._reset_stream_delivery_tracking

    def test_agent_fire_stream_delta(self) -> None:
        agent = self.FakeAgent()
        agent._fire_stream_delta("<|channel|>commentary to=skill_view<|message|>{\"name\":\"hermes\"}<|end|>")
        agent._fire_stream_delta("<|channel|>final<|message|>Hello world!<|end|>")
        assert "".join(agent.fired_deltas) == "Hello world!"

    def test_agent_reset_stream_delivery_tracking(self) -> None:
        agent = self.FakeAgent()
        agent._fire_stream_delta("<|channel|>final<|message|>Hello ")
        # Stream finishes leaving "<|end" at the tail (partial end tag)
        agent._fire_stream_delta("world!<|end")
        assert "".join(agent.fired_deltas) == "Hello world!"
        
        # Now reset tracking which should flush the scrubber
        agent._reset_stream_delivery_tracking()
        # "<|end" is an end tag prefix. In flush, it is returned, routed and emitted.
        # So it should be present in fired_deltas now.
        assert "".join(agent.fired_deltas) == "Hello world!<|end"

