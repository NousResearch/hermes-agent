"""Delivery gates must prevent pre-validation text exposure."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.stream_delivery import StreamDeliveryMixin


class Agent(StreamDeliveryMixin):
    stream_delta_callback: object
    _stream_callback: object
    interim_assistant_callback: object
    _stream_think_scrubber: object
    _stream_context_scrubber: object


def _agent():
    agent = Agent()
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.interim_assistant_callback = None
    agent._streamed_assistant_text_parts = []
    agent._delivered_interim_texts = set()
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    return agent


def test_pre_delivery_gate_buffers_text_deltas():
    delivered = []
    agent = _agent()
    agent.stream_delta_callback = delivered.append
    manager = SimpleNamespace(has_hook=lambda name: name == "pre_delivery")

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        agent._fire_stream_delta("clinical claim")

    assert delivered == []
    assert agent._current_streamed_assistant_text == ""


def test_without_pre_delivery_gate_streams_normally():
    delivered = []
    agent = _agent()
    agent.stream_delta_callback = delivered.append
    manager = SimpleNamespace(has_hook=lambda name: False)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        agent._fire_stream_delta("ordinary answer")

    assert delivered == ["ordinary answer"]


def test_pre_delivery_gate_suppresses_interim_messages():
    delivered = []
    agent = _agent()
    agent.interim_assistant_callback = lambda text, **kwargs: delivered.append(text)
    manager = SimpleNamespace(has_hook=lambda name: name == "pre_delivery")

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        agent._emit_interim_assistant_message({"content": "unvalidated commentary"})

    assert delivered == []


def test_protected_stream_lifecycle_ends_after_approved_delta():
    delivered = []
    events = []
    agent = _agent()
    agent.stream_delta_callback = delivered.append
    agent._pre_delivery_gate_active = True
    agent._enqueue_stream_hook = lambda event, **fields: events.append((event, fields))

    agent._emit_stream_start()
    agent._emit_stream_end(final_text="UNAPPROVED", finished=True, error=None)
    agent._release_pre_delivery_text("APPROVED")

    assert delivered == ["APPROVED"]
    assert [name for name, _ in events] == [
        "on_stream_start", "on_stream_delta", "on_stream_end",
    ]
    assert events[-1][1]["final_text"] == "APPROVED"
