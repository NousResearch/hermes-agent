"""Buzz progress/status thread-routing regressions."""

from gateway.run import _resolve_progress_thread_id


def test_buzz_progress_without_source_thread_uses_triggering_event():
    """Progress for a Buzz reply must inherit the triggering event anchor."""
    assert (
        _resolve_progress_thread_id(
            "buzz",
            source_thread_id=None,
            event_message_id="buzz-event-123",
        )
        == "buzz-event-123"
    )
