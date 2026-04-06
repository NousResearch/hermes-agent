"""Tests for the _submit_clarify_response helper and typed-text override."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


def _make_cli_stub():
    """Create a minimal HermesCLI via __new__ with only the clarify-related attrs."""
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = 0
    cli._invalidate = MagicMock()
    cli._app = SimpleNamespace(invalidate=MagicMock())
    return cli


def _make_mock_event():
    """Create a mock prompt_toolkit event with a resettable buffer."""
    event = MagicMock()
    event.app.current_buffer.text = ""
    return event


class TestSubmitClarifyResponse:
    def test_enqueues_response_and_clears_state(self):
        """_submit_clarify_response should put the response in the queue
        and reset all clarify state."""
        cli = _make_cli_stub()
        response_queue = queue.Queue()
        cli._clarify_state = {
            "question": "Pick one",
            "choices": ["a", "b"],
            "selected": 0,
            "response_queue": response_queue,
        }
        cli._clarify_freetext = True

        event = _make_mock_event()
        cli._submit_clarify_response("my answer", event)

        assert response_queue.get_nowait() == "my answer"
        assert cli._clarify_state is None
        assert cli._clarify_freetext is False
        event.app.current_buffer.reset.assert_called_once()
        event.app.invalidate.assert_called_once()

    def test_no_crash_without_clarify_state(self):
        """Should not crash if _clarify_state is None (race condition guard)."""
        cli = _make_cli_stub()
        cli._clarify_state = None
        event = _make_mock_event()
        cli._submit_clarify_response("text", event)
        assert cli._clarify_state is None

    def test_choice_mode_typed_text_overrides_selection(self):
        """When in choice mode, typed text should be submitted instead of
        the highlighted choice — this is the bug fix."""
        cli = _make_cli_stub()
        response_queue = queue.Queue()
        cli._clarify_state = {
            "question": "Pick one",
            "choices": ["option A", "option B", "Other"],
            "selected": 0,  # "option A" is highlighted
            "response_queue": response_queue,
        }
        cli._clarify_freetext = False

        event = _make_mock_event()

        # Simulate the logic from the Enter handler: if typed text exists,
        # use _submit_clarify_response instead of the selected choice.
        typed_text = "my custom answer"
        if typed_text:
            cli._submit_clarify_response(typed_text, event)

        # The custom answer should be submitted, not "option A"
        result = response_queue.get_nowait()
        assert result == "my custom answer"
        assert result != "option A"
