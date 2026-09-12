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
"""

from types import SimpleNamespace
from unittest.mock import patch

from cron.scheduler_delivery import _live_route_metadata, _standalone_send
from gateway.config import Platform
from tools.send_message_senders import _telegram_thread_kwargs
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


def test_telegram_direct_topic_origin_keeps_native_cron_route():
    env = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "775566675",
        "HERMES_SESSION_THREAD_ID": "270453",
        "HERMES_SESSION_THREAD_ID_KIND": "direct_messages_topic",
    }
    with _session_env(env):
        origin = _origin_from_env()

    assert origin is not None
    assert origin["thread_id_kind"] == "direct_messages_topic"
    delivery = SimpleNamespace(
        job={"id": "job-1"},
        platform=Platform.TELEGRAM,
        platform_name="telegram",
        chat_id="775566675",
        thread_id="270453",
        runtime_adapter=SimpleNamespace(),
        loop=None,
        notify_delivery=True,
        origin=origin,
        origin_target=True,
    )

    route_thread_id, route_metadata, media_metadata = _live_route_metadata(delivery)

    assert route_thread_id is None
    assert route_metadata["direct_messages_topic_id"] == "270453"
    assert "thread_id" not in route_metadata
    assert media_metadata["direct_messages_topic_id"] == "270453"


def test_telegram_direct_topic_standalone_route_keeps_native_parameter(monkeypatch):
    calls = {}

    async def fake_send_to_platform(*args, **kwargs):
        calls.update(kwargs)
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fake_send_to_platform)
    target = SimpleNamespace(
        job={"id": "job-1"},
        where="telegram:775566675",
        platform=Platform.TELEGRAM,
        pconfig=SimpleNamespace(),
        chat_id="775566675",
        thread_id="270453",
        origin={"thread_id_kind": "direct_messages_topic"},
        origin_target=True,
        is_relay=False,
    )

    result, error = _standalone_send(target, "cron follow-up", [])

    assert result == {"success": True}
    assert error is None
    assert calls["thread_id_kind"] == "direct_messages_topic"
    assert _telegram_thread_kwargs("270453", "direct_messages_topic") == {
        "direct_messages_topic_id": 270453,
    }
