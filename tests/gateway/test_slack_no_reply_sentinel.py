import asyncio

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter, _is_no_reply_sentinel


class _FakeSlackClient:
    def __init__(self):
        self.posts = []

    async def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)
        return {"ts": "123.456"}


class _FakeSlackApp:
    def __init__(self, client):
        self.client = client


def _connected_adapter():
    client = _FakeSlackClient()
    adapter = SlackAdapter(PlatformConfig(enabled=True))
    adapter._app = _FakeSlackApp(client)
    return adapter, client


def test_no_reply_sentinel_matches_exact_trimmed_text_only():
    assert _is_no_reply_sentinel("NO_REPLY")
    assert _is_no_reply_sentinel("  NO_REPLY\n")
    assert not _is_no_reply_sentinel("Please do not use NO_REPLY here")
    assert not _is_no_reply_sentinel("")
    assert not _is_no_reply_sentinel(None)


def test_slack_send_suppresses_no_reply_sentinel_without_posting():
    adapter, client = _connected_adapter()

    result = asyncio.run(adapter.send("C123", "  NO_REPLY\n"))

    assert result.success is True
    assert result.message_id is None
    assert result.raw_response == {"suppressed": "NO_REPLY"}
    assert client.posts == []


def test_slack_send_posts_regular_messages_containing_no_reply_text():
    adapter, client = _connected_adapter()

    result = asyncio.run(adapter.send("C123", "I would not use NO_REPLY here"))

    assert result.success is True
    assert result.message_id == "123.456"
    assert len(client.posts) == 1
    assert client.posts[0]["channel"] == "C123"
    assert client.posts[0]["text"] == "I would not use NO_REPLY here"
