"""MattermostAdapter wiring for opt-in native rich posts."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import (
    MattermostAdapter,
    _apply_yaml_config,
    _standalone_send,
)


def _adapter(*, rich_posts=False, feedback_buttons=False):
    config = PlatformConfig(
        enabled=True,
        token="bot-token",
        extra={
            "url": "https://mm.example.com",
            "rich_posts": rich_posts,
            "feedback_buttons": feedback_buttons,
            "interaction_url": "https://hermes.example.com/mattermost/actions",
            "interaction_allowed_cidrs": ["127.0.0.1/32"],
        },
    )
    adapter = MattermostAdapter(config)
    adapter._thread_root_for_send = AsyncMock(return_value=None)
    adapter._post_preserving_thread = AsyncMock(return_value={"id": "post-1"})
    adapter._api_put = AsyncMock(return_value={"id": "post-1"})
    return adapter


@pytest.mark.asyncio
async def test_delete_message_removes_obsolete_stream_chunk():
    adapter = _adapter()
    adapter._api_delete = AsyncMock(return_value=True)

    assert await adapter.delete_message("channel-1", "post123") is True
    adapter._api_delete.assert_awaited_once_with("posts/post123")


@pytest.mark.asyncio
async def test_delete_message_rejects_malformed_post_id():
    adapter = _adapter()
    adapter._api_delete = AsyncMock(return_value=True)

    assert await adapter.delete_message("channel-1", "../post") is False
    adapter._api_delete.assert_not_awaited()


def test_yaml_bridge_seeds_rich_post_extras():
    extras = _apply_yaml_config(
        {},
        {
            "rich_posts": True,
            "feedback_buttons": True,
            "interaction_url": "https://hermes.example.com/mattermost/actions",
            "interaction_host": "0.0.0.0",
            "interaction_port": 9876,
            "interaction_timeout_seconds": 600,
            "interaction_allowed_cidrs": ["10.0.0.12/32"],
            "observe_unmentioned_channel_messages": True,
        },
    )

    assert extras == {
        "rich_posts": True,
        "feedback_buttons": True,
        "interaction_url": "https://hermes.example.com/mattermost/actions",
        "interaction_host": "0.0.0.0",
        "interaction_port": 9876,
        "interaction_timeout_seconds": 600,
        "interaction_allowed_cidrs": ["10.0.0.12/32"],
        "observe_unmentioned_channel_messages": True,
    }


@pytest.mark.asyncio
async def test_standalone_sender_uses_rich_posts_and_plain_retry(monkeypatch):
    posted = []
    responses = [(400, {"message": "attachments disabled"}), (201, {"id": "p-1"})]

    class Response:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return self._body

        async def text(self):
            return str(self._body)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, _url, **kwargs):
            posted.append(kwargs["json"])
            status, body = responses.pop(0)
            return Response(status, body)

    monkeypatch.setattr("aiohttp.ClientSession", Session)
    config = PlatformConfig(
        enabled=True,
        token="token",
        extra={"url": "https://mm.example.com", "rich_posts": True},
    )

    result = await _standalone_send(config, "channel-1", "# Status\n\nReady")

    assert result["success"] is True
    assert result["message_id"] == "p-1"
    assert posted[0]["message"] == "# Status\n\nReady"
    assert posted[0]["props"]["attachments"][0]["fallback"] == "# Status\n\nReady"
    assert posted[1] == {"channel_id": "channel-1", "message": "# Status\n\nReady"}

    responses.extend([(201, {"id": "p-2"}), (201, {"id": "p-3"})])
    multi_result = await _standalone_send(
        config,
        "channel-1",
        "part 1",
        multi_chunk=True,
    )
    long_message = "x" * 5000
    long_result = await _standalone_send(config, "channel-1", long_message)

    assert multi_result["success"] is True
    assert long_result["success"] is True
    assert posted[2] == {"channel_id": "channel-1", "message": "part 1"}
    assert posted[3] == {"channel_id": "channel-1", "message": long_message}


@pytest.mark.asyncio
async def test_rich_posts_are_off_by_default():
    adapter = _adapter()
    assert adapter.REQUIRES_EDIT_FINALIZE is False

    result = await adapter.send("channel-1", "# Status\n\nReady")

    assert result.success is True
    payload = adapter._post_preserving_thread.await_args.args[1]
    assert payload == {
        "channel_id": "channel-1",
        "message": "# Status\n\nReady",
        "props": {"disable_mentions": True},
    }


@pytest.mark.asyncio
async def test_single_chunk_rich_post_uses_attachment_and_fallback():
    adapter = _adapter(rich_posts=True)
    assert adapter.REQUIRES_EDIT_FINALIZE is True

    result = await adapter.send("channel-1", "# Status\n\nReady")

    assert result.success is True
    payload = adapter._post_preserving_thread.await_args.args[1]
    assert payload["message"] == "# Status\n\nReady"
    attachment = payload["props"]["attachments"][0]
    assert attachment["fallback"] == "# Status\n\nReady"
    assert attachment["title"] == "Status"
    assert attachment["text"] == "\nReady"


@pytest.mark.asyncio
async def test_multi_chunk_response_stays_plain_markdown():
    adapter = _adapter(rich_posts=True)

    result = await adapter.send("channel-1", "# Status\n\n" + ("x" * 5000))

    assert result.success is True
    assert adapter._post_preserving_thread.await_count > 1
    for call in adapter._post_preserving_thread.await_args_list:
        payload = call.args[1]
        assert payload["message"]
        assert payload["props"] == {"disable_mentions": True}


@pytest.mark.asyncio
async def test_rich_post_failure_retries_plain_markdown():
    adapter = _adapter(rich_posts=True)
    adapter._post_preserving_thread = AsyncMock(side_effect=[{}, {"id": "plain-post"}])

    result = await adapter.send("channel-1", "# Status\n\nReady")

    assert result.success is True
    assert result.message_id == "plain-post"
    first = adapter._post_preserving_thread.await_args_list[0].args[1]
    second = adapter._post_preserving_thread.await_args_list[1].args[1]
    assert "props" in first
    assert second == {
        "channel_id": "channel-1",
        "message": "# Status\n\nReady",
        "props": {"disable_mentions": True},
    }


@pytest.mark.asyncio
async def test_streaming_edit_only_renders_rich_post_on_finalize():
    adapter = _adapter(rich_posts=True)

    interim = await adapter.edit_message(
        "channel-1", "post-1", "# Status\n\nWor", finalize=False
    )
    final = await adapter.edit_message(
        "channel-1", "post-1", "# Status\n\nWorking", finalize=True
    )

    assert interim.success is True and final.success is True
    interim_payload = adapter._api_put.await_args_list[0].args[1]
    final_payload = adapter._api_put.await_args_list[1].args[1]
    assert interim_payload == {
        "message": "# Status\n\nWor",
        "props": {"disable_mentions": True},
    }
    assert final_payload["message"] == "# Status\n\nWorking"
    assert final_payload["props"]["attachments"][0]["title"] == "Status"


@pytest.mark.asyncio
async def test_finalized_stream_overflow_edit_stays_plain():
    adapter = _adapter(rich_posts=True)

    result = await adapter.edit_message(
        "channel-1",
        "post-1",
        "# Status\n\nFirst overflow chunk",
        finalize=True,
        metadata={"multi_chunk": True},
    )

    assert result.success is True
    payload = adapter._api_put.await_args.args[1]
    assert payload == {
        "message": "# Status\n\nFirst overflow chunk",
        "props": {"disable_mentions": True},
    }


@pytest.mark.asyncio
async def test_streaming_preview_send_stays_plain_until_final_edit():
    adapter = _adapter(rich_posts=True)

    await adapter.send(
        "channel-1",
        "# Status\n\nStarting",
        metadata={"expect_edits": True},
    )

    payload = adapter._post_preserving_thread.await_args.args[1]
    assert payload == {
        "channel_id": "channel-1",
        "message": "# Status\n\nStarting",
        "props": {"disable_mentions": True},
    }


@pytest.mark.asyncio
async def test_prechunked_send_message_piece_stays_plain():
    adapter = _adapter(rich_posts=True)

    await adapter.send(
        "channel-1",
        "part 1",
        metadata={"multi_chunk": True},
    )

    payload = adapter._post_preserving_thread.await_args.args[1]
    assert payload == {
        "channel_id": "channel-1",
        "message": "part 1",
        "props": {"disable_mentions": True},
    }


@pytest.mark.asyncio
async def test_single_chunk_rich_post_adds_signed_feedback_buttons(monkeypatch):
    monkeypatch.setenv("MATTERMOST_INTERACTION_SECRET", "feedback-secret")
    adapter = _adapter(rich_posts=True, feedback_buttons=True)

    result = await adapter.send("channel-1", "# Result\n\nComplete")

    assert result.success is True
    payload = adapter._post_preserving_thread.await_args.args[1]
    actions = payload["props"]["attachments"][0]["actions"]
    assert [action["name"] for action in actions] == ["Helpful", "Not helpful"]
    assert "feedback-secret" not in repr(payload)

    callback = {
        "user_id": "user-1",
        "user_name": "Nawaf",
        "channel_id": "channel-1",
        "post_id": "post-1",
        "context": actions[0]["integration"]["context"],
    }
    adapter._is_interactive_user_authorized = lambda *args, **kwargs: True
    body, status = await adapter._dispatch_interaction(callback)

    assert status == 200
    assert "Thanks" in body["ephemeral_text"]
    assert body["update"]["message"] == "# Result\n\nComplete"
    assert "actions" not in body["update"]["props"]["attachments"][0]
