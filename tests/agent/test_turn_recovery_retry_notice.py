"""A provider retry backoff names itself on the live status line.

The buffered retry status replays only if every retry fails, so during the
backoff itself the user used to see an anonymous spinner — and right after a
tool that just finished (a connector sign-in landing, say) it read as the
agent going silent. The wait notice is transient (rewritten by the next
frame, cleared on recovery) and rides the frame long provider waits already
use, so it adds none of the transcript chatter the buffer exists to avoid."""

from unittest.mock import MagicMock

from agent.turn_recovery import compute_error_backoff


def test_retry_backoff_names_the_wait_on_the_live_status_line():
    agent = MagicMock()
    agent._client_log_context.return_value = ""

    wait = compute_error_backoff(
        agent, RuntimeError("502"), retry_count=1, max_retries=3,
        is_rate_limited=False, is_zai_coding_overload=False,
        base_url="https://example.test/v1", model="test/model",
    )

    assert wait > 0
    # Still buffered for the exhausted-retries replay …
    agent._buffer_status.assert_called_once()
    # … and named live while the backoff runs.
    agent._emit_wait_notice.assert_called_once()
    text = agent._emit_wait_notice.call_args.args[0]
    assert text.startswith("⏳ waiting on provider")
    assert "attempt 1/3" in text
