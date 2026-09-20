"""Cron origin capture: Slack per-message session-key threads are not routing.

Bug report (relay-fronted Slack, thread-per-message mode): creating a cron job
from a top-level Slack DM message persisted the creation message's own id as
``origin.thread_id`` — the relay adapter stamps ``source.thread_id = message_id``
on every top-level Slack message purely for SESSION KEYING (native SlackAdapter
parity: ``thread_ts = event.thread_ts or ts``). Every subsequent cron delivery
then landed inside the ephemeral thread spawned around the creation message
instead of the top-level conversation / configured home.

The stamp is recognizable at capture time: a Slack session whose thread id
equals the triggering message's own id is a synthetic per-message key, not a
durable thread. A genuine in-thread creation has thread_id == the parent
thread's id != the triggering message's own id, and must keep its thread.

Near-horizon exception (#117306): with ``platforms.slack.extra.reply_in_thread``
at its default ``true`` the assistant answers a top-level message inside a thread
keyed on that same message id, so for a job whose first fire is within the
conversation's remaining lifetime the id is the conversation's location, not a
per-message key — the thread is kept for near one-shots and still dropped for
recurring jobs and one-shots beyond the horizon window.
"""

from datetime import timedelta
from unittest.mock import patch

import hermes_time
from tools.cronjob_tools import _origin_from_env


def _session_env(env: dict):
    """Patch gateway.session_context.get_session_env with a dict lookup."""
    return patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda name, default="": env.get(name, default),
    )


class TestSlackSyntheticThreadCapture:
    def test_synthetic_slack_thread_not_captured(self):
        """thread_id == message_id on Slack = per-message session key: drop it."""
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_THREAD_ID": "1755043010.123456",
            "HERMES_SESSION_MESSAGE_ID": "1755043010.123456",
        }
        with _session_env(env):
            origin = _origin_from_env()
        assert origin is not None
        assert origin["platform"] == "slack"
        assert origin["chat_id"] == "D0BJTDCSR7C"
        assert origin["thread_id"] is None

    def test_genuine_slack_thread_preserved(self):
        """A real in-thread creation (thread != own message id) keeps its thread."""
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "C0AGENERAL",
            "HERMES_SESSION_THREAD_ID": "1755040000.000100",
            "HERMES_SESSION_MESSAGE_ID": "1755043010.123456",
        }
        with _session_env(env):
            origin = _origin_from_env()
        assert origin is not None
        assert origin["thread_id"] == "1755040000.000100"

    def test_non_slack_platform_thread_untouched(self):
        """Telegram forum topics legitimately reuse ids; the rule is Slack-scoped."""
        env = {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "-1003941067111",
            "HERMES_SESSION_THREAD_ID": "2203",
            "HERMES_SESSION_MESSAGE_ID": "2203",
        }
        with _session_env(env):
            origin = _origin_from_env()
        assert origin is not None
        assert origin["thread_id"] == "2203"

    def test_slack_no_message_id_keeps_thread(self):
        """Without a message id to compare, never guess: keep the thread."""
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_THREAD_ID": "1755040000.000100",
        }
        with _session_env(env):
            origin = _origin_from_env()
        assert origin is not None
        assert origin["thread_id"] == "1755040000.000100"


_TOP_LEVEL_SLACK = {
    "HERMES_SESSION_PLATFORM": "slack",
    "HERMES_SESSION_CHAT_ID": "C0AGENERAL",
    # reply_in_thread default: the assistant's whole exchange lives in the thread keyed on the
    # asking message's own id (ts == thread_ts on the wire).
    "HERMES_SESSION_THREAD_ID": "1755043010.123456",
    "HERMES_SESSION_MESSAGE_ID": "1755043010.123456",
}


def _run_at_in(minutes: int) -> str:
    return (hermes_time.now() + timedelta(minutes=minutes)).isoformat()


class TestNearHorizonSlackThreadKept:
    """#117306 — a one-shot firing within the conversation's remaining lifetime delivers back
    into the thread the assistant opened under the asking message."""

    def test_oneshot_within_thread_horizon_keeps_thread(self):
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env({"kind": "once", "run_at": _run_at_in(1)})
        assert origin is not None
        assert origin["thread_id"] == "1755043010.123456"

    def test_oneshot_at_horizon_boundary_keeps_thread(self):
        """The horizon itself (60 minutes) is inside the conversation's lifetime."""
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env({"kind": "once", "run_at": _run_at_in(60)})
        assert origin is not None
        assert origin["thread_id"] == "1755043010.123456"

    def test_oneshot_beyond_thread_horizon_keeps_drop(self):
        """A one-shot months away belongs in the channel, not a thread from the day it was
        asked — the original rationale still protects this case."""
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env({"kind": "once", "run_at": _run_at_in(61)})
        assert origin is not None
        assert origin["thread_id"] is None

    def test_recurring_schedule_keeps_drop(self):
        """A recurring job outlives any conversation; the per-message key stays synthetic."""
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env({"kind": "interval", "minutes": 1})
        assert origin is not None
        assert origin["thread_id"] is None

    def test_unparseable_run_at_keeps_drop(self):
        """An unparseable horizon must not resurrect the thread (fail-closed to the historical
        behaviour)."""
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env({"kind": "once", "run_at": "not-a-timestamp"})
        assert origin is not None
        assert origin["thread_id"] is None

    def test_no_schedule_falls_back_to_today_drop(self):
        """Legacy callers (no schedule) keep the unconditional drop."""
        with _session_env(_TOP_LEVEL_SLACK):
            origin = _origin_from_env()
        assert origin is not None
        assert origin["thread_id"] is None
