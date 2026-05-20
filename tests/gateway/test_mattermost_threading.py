from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, _reply_anchor_for_event
from gateway.platforms.mattermost import MattermostAdapter
from gateway.session import SessionSource


def test_mattermost_defaults_to_threaded_replies(monkeypatch):
    monkeypatch.delenv("MATTERMOST_REPLY_MODE", raising=False)

    adapter = MattermostAdapter(
        PlatformConfig(
            token="token",
            extra={"url": "https://mattermost.example.test"},
        )
    )

    assert adapter._reply_mode == "thread"
    assert adapter._should_thread_reply("channel-id") is True


def test_mattermost_existing_thread_replies_anchor_to_root_post():
    source = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="channel-id",
        chat_type="channel",
        thread_id="root-post-id",
    )
    event = MessageEvent(
        text="reply in existing thread",
        message_type=MessageType.TEXT,
        source=source,
        message_id="child-post-id",
    )

    assert _reply_anchor_for_event(event) == "root-post-id"


def test_mattermost_top_level_post_replies_anchor_to_triggering_post():
    source = SessionSource(
        platform=Platform.MATTERMOST,
        chat_id="channel-id",
        chat_type="channel",
        thread_id=None,
    )
    event = MessageEvent(
        text="top-level mention",
        message_type=MessageType.TEXT,
        source=source,
        message_id="top-level-post-id",
    )

    assert _reply_anchor_for_event(event) == "top-level-post-id"
