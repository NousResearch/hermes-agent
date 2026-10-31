"""A Slack thread source carries its channel as ``parent_chat_id``, like Discord/Matrix threads.

Without the anchor, every Kanban subscription auto-created from a Slack thread was thread-shaped
(``thread_id`` set) but anchorless, so once ``gateway.profile_routes`` held any channel-level entry the
notifier's fail-closed route gate skipped the row forever ("has no parent_chat_id anchor ... will
not be delivered").
"""
import asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from gateway.session import SessionContext
from plugins.platforms.slack.adapter import SlackAdapter


def make_adapter(extra=None):
    adapter = SlackAdapter(PlatformConfig(extra=extra or {}))
    adapter._bot_user_id = "UBOT"
    adapter._team_bot_user_ids["T1"] = "UBOT"
    adapter._has_active_session_for_thread = lambda **_: False

    async def empty(**_):
        return ""

    async def user_name(*_, **__):
        return "Sebastian"

    adapter._fetch_thread_context = empty
    adapter._fetch_thread_parent_text = empty
    adapter._resolve_user_name = user_name
    return adapter


def slack_event(text, ts="100.000", thread_ts=None):
    event = {"type": "message", "channel": "C123", "channel_type": "channel", "team": "T1",
             "user": "U123", "text": text, "ts": ts}
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def capture(adapter):
    handled = []

    async def _handle(event):
        handled.append(event)

    adapter.handle_message = _handle
    return handled


def test_thread_reply_source_is_anchored_to_its_channel():
    adapter = make_adapter({"allowed_channels": ["C123"], "require_mention": False})
    handled = capture(adapter)
    asyncio.run(adapter._handle_slack_message(slack_event("<@UBOT> status?", ts="100.500", thread_ts="100.000")))
    source = handled[0].source
    assert (source.chat_id, source.thread_id, source.parent_chat_id) == ("C123", "100.000", "C123")


def test_flat_channel_message_stays_anchorless():
    adapter = make_adapter({"allowed_channels": ["C123"], "require_mention": False, "reply_in_thread": False})
    handled = capture(adapter)
    asyncio.run(adapter._handle_slack_message(slack_event("<@UBOT> hi")))
    source = handled[0].source
    assert source.thread_id is None and source.parent_chat_id is None


def test_slash_command_in_thread_is_anchored():
    adapter = make_adapter()
    handled = capture(adapter)
    asyncio.run(adapter._handle_slash_command({"text": "hello", "user_id": "U123", "channel_id": "C123",
                                               "team_id": "T1", "thread_ts": "100.000"}))
    assert handled and handled[0].source.parent_chat_id == "C123"
    handled.clear()
    asyncio.run(adapter._handle_slash_command({"text": "hello", "user_id": "U123", "channel_id": "C123",
                                               "team_id": "T1"}))
    assert handled and handled[0].source.parent_chat_id is None


def test_thread_subscription_passes_the_channel_route_gate(tmp_path, monkeypatch):
    """End to end: Slack thread turn -> session vars -> ``kanban_create`` auto-subscription row ->
    the notifier's profile_routes gate delivers it although a channel-level route exists."""
    from gateway.kanban_watchers_notifier import _adapter_for_subscription
    from tools import kanban_tools as kt

    adapter = make_adapter({"allowed_channels": ["C123"], "require_mention": False})
    handled = capture(adapter)
    asyncio.run(adapter._handle_slack_message(slack_event("<@UBOT> go", ts="100.500", thread_ts="100.000")))
    source = handled[0].source

    runner = GatewayRunner.__new__(GatewayRunner)
    primary = object()
    runner.adapters = {Platform.SLACK: primary}
    runner._profile_adapters = {}
    runner._primary_profile_name = runner._kanban_notifier_profile = "default"
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda config: [("default", tmp_path)])
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=parse_profile_routes([
        dict(platform="slack", chat_id="C999", profile="other"),  # any channel-level route arms the gate
    ]))
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_key="k")
    tokens = runner._set_session_env(context)
    try:
        monkeypatch.setenv("HERMES_SESSION_PROFILE", "default")
        target = kt._resolve_notify_target()
    finally:
        runner._clear_session_env(tokens)
    assert target["delivery_metadata"]["parent_chat_id"] == "C123"
    sub = dict(task_id="t_1", platform="slack", chat_id=target["chat_id"], thread_id=target["thread_id"],
               user_id=target["user_id"], chat_type=target["chat_type"], notifier_profile="default",
               delivery_mode="notify+wake", delivery_metadata=target["delivery_metadata"])
    assert _adapter_for_subscription(runner, Platform.SLACK, sub, "default") is primary
    anchorless = dict(sub, delivery_metadata={k: v for k, v in sub["delivery_metadata"].items() if k != "parent_chat_id"})
    assert _adapter_for_subscription(runner, Platform.SLACK, anchorless, "default") is None
