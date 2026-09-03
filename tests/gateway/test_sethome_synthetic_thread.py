"""/sethome must not persist Slack's synthetic per-message session thread.

Live repro (relay-fronted Slack staging, 2026-08-13): /sethome run as a
top-level DM message captured source.thread_id — which the relay adapter had
stamped with the /sethome message's OWN id for session keying — into the
persisted HomeChannel. Every bare-platform delivery (deliver="slack") then
resolved home chat + home thread and landed inside the ephemeral thread
spawned around the old /sethome message.

Same contract as cron origin capture: a Slack thread id equal to the
message's own id is a synthetic session key, never a durable location.
"""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.slash_commands import _home_thread_from_source


def _source(platform=Platform.SLACK, thread_id=None, message_id=None):
    return SimpleNamespace(
        platform=platform, thread_id=thread_id, message_id=message_id
    )


class TestHomeThreadFromSource:
    def test_synthetic_slack_thread_dropped(self):
        """Top-level /sethome: stamped thread == own message id -> None."""
        src = _source(thread_id="1755043010.123456", message_id="1755043010.123456")
        assert _home_thread_from_source(src) is None

    def test_genuine_slack_thread_kept(self):
        """/sethome inside a real thread keeps that thread as home target."""
        src = _source(thread_id="1755040000.000100", message_id="1755043010.123456")
        assert _home_thread_from_source(src) == "1755040000.000100"

    def test_no_thread_returns_none(self):
        assert _home_thread_from_source(_source()) is None

    def test_no_message_id_keeps_thread(self):
        """Without a message id to compare, never guess: keep the thread."""
        src = _source(thread_id="1755040000.000100", message_id=None)
        assert _home_thread_from_source(src) == "1755040000.000100"

    def test_synthetic_matrix_thread_dropped(self):
        """Top-level /sethome in Matrix under auto-thread: thread == message id -> None."""
        src = _source(
            platform=Platform.MATRIX,
            thread_id="$event123:matrix.org",
            message_id="$event123:matrix.org",
        )
        assert _home_thread_from_source(src) is None

    def test_genuine_matrix_thread_kept(self):
        """/sethome in an existing Matrix thread keeps that thread as home."""
        src = _source(
            platform=Platform.MATRIX,
            thread_id="$root_event:matrix.org",
            message_id="$reply_event:matrix.org",
        )
        assert _home_thread_from_source(src) == "$root_event:matrix.org"

    def test_non_slack_matrix_platform_untouched(self):
        """Telegram forum topics legitimately reuse ids; rule is Slack/Matrix-scoped."""
        src = _source(platform=Platform.TELEGRAM, thread_id="2203", message_id="2203")
        assert _home_thread_from_source(src) == "2203"
