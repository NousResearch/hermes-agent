"""Platform thread anchors for progress replies."""

from gateway.config import Platform


class TestSlackReplyInThreadProgressRouting:
    """#18859: reply_in_thread=false must stop progress from creating threads."""

    def test_slack_reply_in_thread_false_drops_synthetic_thread(self):
        from gateway.run import _resolve_progress_thread_id

        # source.thread_id == event ts is the adapter's synthetic
        # session-keying thread for top-level messages — not a real thread.
        assert _resolve_progress_thread_id(
            Platform.SLACK,
            source_thread_id="1700000000.000100",
            event_message_id="1700000000.000100",
            reply_in_thread=False,
        ) is None

    def test_buzz_uses_event_message_id_as_progress_thread(self):
        """Buzz has no native thread_id; progress must reply-to the trigger."""
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz",
            source_thread_id=None,
            event_message_id="evt-trigger-001",
            reply_in_thread=True,
        ) == "evt-trigger-001"
