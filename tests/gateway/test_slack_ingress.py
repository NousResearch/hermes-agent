from __future__ import annotations

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.slack.ingress import (
    FollowStore,
    SlackIngressPolicy,
    message_event_to_wire,
)


def test_follow_store_prunes_expired_and_oldest_rows(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=2,
        clock=lambda: 1_000.0,
    )

    store.follow("T1", "C1", "expired", seen_at=800.0)
    store.follow("T1", "C1", "older", seen_at=950.0)
    store.follow("T1", "C1", "newer", seen_at=975.0)
    store.follow("T1", "C1", "newest", seen_at=990.0)

    assert store.is_followed("T1", "C1", "expired") is False
    assert store.is_followed("T1", "C1", "older") is False
    assert store.is_followed("T1", "C1", "newer") is True
    assert store.is_followed("T1", "C1", "newest") is True
    assert store.count() == 2


def test_root_mention_starts_following_the_new_thread(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
        clock=lambda: 1_000.0,
    )
    policy = SlackIngressPolicy(store)

    admitted = policy.admit(
        {"channel": "C1", "ts": "100.1", "text": "  <@B1> hello"},
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )

    assert admitted is True
    assert store.is_followed("T1", "C1", "100.1") is True


def test_followed_thread_messages_refresh_the_idle_ttl(tmp_path):
    now = [1_000.0]
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
        clock=lambda: now[0],
    )
    store.follow("T1", "C1", "100.1", seen_at=950.0)
    policy = SlackIngressPolicy(store)

    now[0] = 1_040.0
    admitted = policy.admit(
        {
            "channel": "C1",
            "ts": "104.1",
            "thread_ts": "100.1",
            "text": "plain follow-up",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )
    now[0] = 1_120.0

    assert admitted is True
    assert store.is_followed("T1", "C1", "100.1") is True


def test_untracked_thread_mention_is_one_shot(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
        clock=lambda: 1_000.0,
    )
    policy = SlackIngressPolicy(store)

    mentioned = policy.admit(
        {
            "channel": "C1",
            "ts": "200.2",
            "thread_ts": "200.1",
            "text": "<@B1> answer this only",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )
    later_plain = policy.admit(
        {
            "channel": "C1",
            "ts": "200.3",
            "thread_ts": "200.1",
            "text": "humans keep talking",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )

    assert mentioned is True
    assert store.is_followed("T1", "C1", "200.1") is False
    assert later_plain is False


def test_message_event_to_wire_preserves_slack_context_and_media():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        chat_name="general",
        chat_type="group",
        user_id="U1",
        user_name="Alice",
        thread_id="100.1",
        scope_id="T1",
        message_id="100.2",
    )
    event = MessageEvent(
        text="summarize this",
        message_type=MessageType.DOCUMENT,
        source=source,
        message_id="100.2",
        media_urls=["/tmp/report.pdf"],
        media_types=["application/pdf"],
        reply_to_message_id="100.1",
        reply_to_text="parent",
        reply_to_author_id="U2",
        reply_to_author_name="Bob",
        channel_prompt="channel rules",
        auto_skill=["one", "two"],
        metadata={"slack_team_id": "T1", "slack_thread_ts": "100.1"},
    )

    wire = message_event_to_wire(event)

    assert wire["message_type"] == "document"
    assert wire["source"]["platform"] == "slack"
    assert wire["source"]["scope_id"] == "T1"
    assert wire["media_urls"] == ["/tmp/report.pdf"]
    assert wire["media_types"] == ["application/pdf"]
    assert wire["reply_to_text"] == "parent"
    assert wire["reply_to_author_id"] == "U2"
    assert wire["reply_to_author_name"] == "Bob"
    assert wire["channel_prompt"] == "channel rules"
    assert wire["auto_skill"] == ["one", "two"]
    assert wire["metadata"]["slack_thread_ts"] == "100.1"

    from gateway.relay.ws_transport import _event_from_wire

    rebuilt = _event_from_wire(wire, authorization_is_upstream=False)
    assert rebuilt.media_types == ["application/pdf"]
    assert rebuilt.reply_to_text == "parent"
    assert rebuilt.reply_to_author_id == "U2"
    assert rebuilt.reply_to_author_name == "Bob"
    assert rebuilt.channel_prompt == "channel rules"
    assert rebuilt.auto_skill == ["one", "two"]
    assert rebuilt.metadata["slack_thread_ts"] == "100.1"


def test_mute_removes_follow_before_any_gateway_delivery(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
        clock=lambda: 1_000.0,
    )
    store.follow("T1", "C1", "300.1")
    policy = SlackIngressPolicy(store)

    mute_admitted = policy.admit(
        {
            "channel": "C1",
            "ts": "300.2",
            "thread_ts": "300.1",
            "text": "<@B1> /mute",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )
    later_plain = policy.admit(
        {
            "channel": "C1",
            "ts": "300.3",
            "thread_ts": "300.1",
            "text": "this must stay outside Gateway",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )

    assert mute_admitted is False
    assert store.is_followed("T1", "C1", "300.1") is False
    assert later_plain is False


def test_control_commands_are_extensible_and_consumed_locally(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
    )
    policy = SlackIngressPolicy(store)
    seen = []

    def _pause(**context):
        seen.append(context)

    policy.register_control_command("/pause", _pause)
    admitted = policy.admit(
        {
            "channel": "C1",
            "ts": "400.2",
            "thread_ts": "400.1",
            "text": "<@B1> /pause later",
        },
        team_id="T1",
        bot_user_id="B1",
        is_one_to_one_dm=False,
    )

    assert admitted is False
    assert seen[0]["team_id"] == "T1"
    assert seen[0]["channel_id"] == "C1"
    assert seen[0]["thread_ts"] == "400.1"
    assert seen[0]["event"]["ts"] == "400.2"


def test_reaction_trigger_requires_owner_allowed_emoji_and_message_item(tmp_path):
    store = FollowStore(
        tmp_path / "follow.db",
        ttl_seconds=100,
        max_threads=10,
    )
    policy = SlackIngressPolicy(
        store,
        reaction_user_ids={"U_OWNER"},
        reaction_names={"eyes"},
    )
    base = {
        "type": "reaction_added",
        "user": "U_OWNER",
        "reaction": "eyes",
        "item": {"type": "message", "channel": "C1", "ts": "500.1"},
    }

    assert policy.admit_reaction(base) is True
    assert policy.admit_reaction({**base, "user": "U_OTHER"}) is False
    assert policy.admit_reaction({**base, "reaction": "thumbsup"}) is False
    assert (
        policy.admit_reaction({**base, "item": {"type": "file", "file": "F1"}}) is False
    )
    assert store.count() == 0
