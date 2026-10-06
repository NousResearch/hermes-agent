"""Buzz progress/status thread-routing regressions."""

from gateway.run import _resolve_progress_thread_id


def test_buzz_progress_without_source_thread_defers_to_adapter_policy():
    """Buzz placement is resolved from source metadata by its adapter."""
    assert (
        _resolve_progress_thread_id(
            "buzz",
            source_thread_id=None,
            event_message_id="buzz-event-123",
        )
        is None
    )


def test_buzz_progress_preserves_explicit_source_thread():
    assert (
        _resolve_progress_thread_id(
            "buzz",
            source_thread_id="explicit-thread-root",
            event_message_id="buzz-event-123",
        )
        == "explicit-thread-root"
    )
