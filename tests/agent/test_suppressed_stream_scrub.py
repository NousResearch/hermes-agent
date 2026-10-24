"""Scrubbing on the tool-call suppressed stream path.

Tool-call turns do not stream their preamble as ordinary content, but the delta
still has to pass the stateful scrubbers before it reaches the display and the
streamed-text accumulator. Routing it around them delivered reasoning envelopes
live and let the finalizer adopt raw reasoning as the assistant reply (#95514).
"""

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from agent.chat_completion_helpers import _StreamingCall
from agent.memory_manager import StreamingContextScrubber
from agent.stream_delivery import StreamDeliveryMixin
from agent.think_scrubber import StreamingThinkScrubber


class _StubAgent(StreamDeliveryMixin):
    """Minimal surface used by ``_scrub_stream_delta`` / ``_route_suppressed_text``."""

    def __init__(self) -> None:
        self.delivered: list[str] = []
        self.reasoning: list[str] = []
        self.stream_delta_callback = self.delivered.append
        self.reasoning_callback = self.reasoning.append
        self._stream_callback = None
        self._stream_think_scrubber = StreamingThinkScrubber()
        self._stream_context_scrubber = StreamingContextScrubber()
        self._streamed_assistant_text_parts: list[str] = []
        self._native_reasoning_streamed = False
        self._stream_reasoning_hooks_enabled = False
        self._stream_needs_break = False

    def _strip_think_blocks(self, content: str) -> str:
        return strip_think_blocks(self, content)

    def _stream_hook_base_payload(self) -> dict:
        return {}


def _suppressed_call(agent):
    # _route_suppressed_text only touches self.agent and self._quiet; skip the
    # request registry the real constructor builds.
    call = object.__new__(_StreamingCall)
    call.agent = agent
    return call


def test_suppressed_delta_never_delivers_reasoning():
    agent = _StubAgent()
    call = _suppressed_call(agent)

    call._route_suppressed_text("<think>hidden plan")
    call._route_suppressed_text(" continues</think>Visible answer.")

    assert "".join(agent.delivered) == "Visible answer."
    assert agent._current_streamed_assistant_text == "Visible answer."
    # Stripped reasoning still reaches the reasoning pane (inline forwarding).
    assert "".join(agent.reasoning) == "hidden plan continues"


def test_suppressed_delta_keeps_display_only_contract():
    agent = _StubAgent()
    tts: list[str] = []
    agent._stream_callback = tts.append
    call = _suppressed_call(agent)

    call._route_suppressed_text("plain preamble")

    assert agent.delivered == ["plain preamble"]
    assert tts == []


def test_suppressed_path_shares_scrubber_state_with_live_path():
    agent = _StubAgent()
    call = _suppressed_call(agent)

    # The live path opens the block; the suppressed path must see that state
    # instead of treating its delta as a fresh, tag-free string.
    agent._fire_stream_delta("<think>start ")
    call._route_suppressed_text("still reasoning</think>done")

    assert "".join(agent.delivered) == "done"
    assert agent._current_streamed_assistant_text == "done"


def test_suppressed_text_without_display_callback_is_not_recorded():
    agent = _StubAgent()
    agent.stream_delta_callback = None
    call = _suppressed_call(agent)

    call._route_suppressed_text("<think>x</think>y")

    assert agent._current_streamed_assistant_text == ""


def test_recover_final_strips_accumulated_reasoning():
    from agent.turn_finalizer import _recover_final_from_stream

    agent = _StubAgent()
    agent._current_streamed_assistant_text = "<think>secret</think>Visible answer."

    assert _recover_final_from_stream(agent, "", False, False) == ("Visible answer.", True)


def test_recover_final_drops_pure_reasoning_accumulator():
    from agent.turn_finalizer import _recover_final_from_stream

    agent = _StubAgent()
    agent._current_streamed_assistant_text = "<think>secret</think>"

    assert _recover_final_from_stream(agent, "", False, False) == ("", False)


def test_recover_final_keeps_plain_streamed_text():
    from agent.turn_finalizer import _recover_final_from_stream

    agent = _StubAgent()
    agent._current_streamed_assistant_text = "Visible answer."

    assert _recover_final_from_stream(agent, "", False, False) == ("Visible answer.", True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
