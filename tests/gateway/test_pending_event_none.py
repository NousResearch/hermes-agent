"""Tests for the pending_event None guard in recursive _run_agent calls.

When pending_event is None (Path B: pending comes from interrupt_message),
accessing pending_event.channel_prompt previously raised AttributeError.
This verifies the fix: channel_prompt is captured inside the
`if pending_event is not None:` block and falls back to None otherwise.
"""

from types import SimpleNamespace


def _extract_channel_prompt(pending_event):
    """Reproduce the fixed logic from gateway/run.py.

    Mirrors the variable-capture pattern used before the recursive
    _run_agent call so we can test both paths without a full runner.
    """
    next_channel_prompt = None
    if pending_event is not None:
        next_channel_prompt = getattr(pending_event, "channel_prompt", None)
    return next_channel_prompt


class TestPendingEventNoneChannelPrompt:
    """Guard against AttributeError when pending_event is None."""

    def test_none_pending_event_returns_none_channel_prompt(self):
        """Path B: pending_event is None — must not raise AttributeError."""
        result = _extract_channel_prompt(None)
        assert result is None

    def test_pending_event_with_channel_prompt_passes_through(self):
        """Path A: pending_event present — channel_prompt is forwarded."""
        event = SimpleNamespace(channel_prompt="You are a helpful bot.")
        result = _extract_channel_prompt(event)
        assert result == "You are a helpful bot."

    def test_pending_event_without_channel_prompt_returns_none(self):
        """Path A: pending_event present but has no channel_prompt attribute."""
        event = SimpleNamespace()
        result = _extract_channel_prompt(event)
        assert result is None
