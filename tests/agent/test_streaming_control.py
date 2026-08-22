"""Focused seam tests for streaming spinner cleanup."""

import pytest

from agent.streaming_control import _stop_spinner


class _Spinner:
    def __init__(self, events):
        self.events = events

    def stop(self, value):
        self.events.append(("spinner", value))


def test_stop_spinner_stops_and_clears_then_calls_callback_by_identity():
    events = []
    callback = lambda value: events.append(("callback", value))
    spinner = _Spinner(events)

    assert _stop_spinner(spinner, callback) is None
    assert events == [("spinner", ""), ("callback", "")]


def test_stop_spinner_calls_callback_without_spinner():
    events = []
    callback = lambda value: events.append(("callback", value))

    assert _stop_spinner(None, callback) is None
    assert events == [("callback", "")]


def test_stop_spinner_does_nothing_when_both_inputs_absent():
    assert _stop_spinner(None, None) is None


def test_stop_spinner_preserves_callback_identity_and_order():
    events = []
    callback = lambda value: events.append(("callback", value))
    spinner = _Spinner(events)

    returned = _stop_spinner(spinner, callback)
    assert returned is None
    assert callback is not None
    assert [kind for kind, _ in events] == ["spinner", "callback"]


def test_stop_spinner_propagates_spinner_exception_and_skips_callback():
    events = []

    class FailingSpinner:
        def stop(self, value):
            events.append(("spinner", value))
            raise RuntimeError("stop failed")

    callback = lambda value: events.append(("callback", value))
    with pytest.raises(RuntimeError, match="stop failed"):
        _stop_spinner(FailingSpinner(), callback)
    assert events == [("spinner", "")]


def test_streaming_adapter_stops_spinner_on_first_delta_at_runtime():
    events = []
    thinking_callback = lambda value: events.append(("thinking-callback", value))
    spinner = _Spinner(events)
    thinking_spinner = spinner

    def _on_first_delta():
        nonlocal thinking_spinner
        thinking_spinner = _stop_spinner(thinking_spinner, thinking_callback)

    def fake_streaming_api_call(*, on_first_delta):
        events.append(("stream-start", thinking_spinner))
        assert on_first_delta is _on_first_delta
        on_first_delta()
        events.append(("stream-after-callback", thinking_spinner))
        return "response"

    result = fake_streaming_api_call(on_first_delta=_on_first_delta)

    assert result == "response"
    assert events == [
        ("stream-start", spinner),
        ("spinner", ""),
        ("thinking-callback", ""),
        ("stream-after-callback", None),
    ]
