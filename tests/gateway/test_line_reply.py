"""Tests for LineReplyClient — reply, postback button, and loading indicator calls."""
import json

import pytest
import respx
from httpx import Response

from gateway.platforms.line import (
    LineReplyClient,
    _strip_markdown_for_line,
    build_postback_button_message,
    DEFAULT_PENDING_REPLY_TEXT as PENDING_REPLY_TEXT,
    DEFAULT_EXPIRED_REPLY_TEXT as EXPIRED_REPLY_TEXT,
    DEFAULT_ALREADY_DELIVERED_TEXT as ALREADY_DELIVERED_TEXT,
)


def test_strip_markdown_for_line_preserves_link_urls():
    """LINE auto-linkifies bare URLs, so [label](url) must rewrite to
    'label (url)' before the shared strip_markdown drops the URL.
    Without this, citations would lose their tappable links."""
    out = _strip_markdown_for_line(
        "See **the docs** at [LINE Developers](https://developers.line.biz)."
    )
    # Bold stripped, link rewritten with URL preserved as bare text.
    assert "**" not in out
    assert "[LINE Developers]" not in out
    assert "LINE Developers" in out
    assert "https://developers.line.biz" in out


def test_strip_markdown_for_line_strips_other_markdown():
    """Headings, bold, italic, code spans all collapse to plain text."""
    out = _strip_markdown_for_line(
        "# Heading\n\n**bold** and *italic* and `code`"
    )
    for marker in ("#", "**", "*", "`"):
        assert marker not in out
    for word in ("Heading", "bold", "italic", "code"):
        assert word in out


def test_build_postback_button_message_carries_request_id():
    msg = build_postback_button_message(
        text=PENDING_REPLY_TEXT,
        button_label="📋 Show response",
        request_id="rid-123",
    )
    # Template Buttons message — persistent (Quick Reply chips would be
    # transient; see build_postback_button_message docstring).
    assert msg["type"] == "template"
    assert msg["template"]["type"] == "buttons"
    assert msg["altText"] == PENDING_REPLY_TEXT
    actions = msg["template"]["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["type"] == "postback"
    payload = json.loads(action["data"])
    assert payload["action"] == "show_response"
    assert payload["request_id"] == "rid-123"


@pytest.mark.asyncio
@respx.mock
async def test_reply_sends_post_to_line():
    route = respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(200, json={})
    )
    client = LineReplyClient(channel_access_token="test-token")
    await client.reply("rt-1", [{"type": "text", "text": "hello"}])
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    assert sent["replyToken"] == "rt-1"
    assert sent["messages"][0]["text"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_show_loading_suppressed_for_group_id():
    """show_loading() must skip the HTTP call for group/room IDs (LINE limitation)."""
    route = respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )
    client = LineReplyClient(channel_access_token="t")
    await client.show_loading("C_group_id_starts_with_C")
    await client.show_loading("R_room_id_starts_with_R")
    await client.show_loading("")
    assert not route.called


@pytest.mark.asyncio
@respx.mock
async def test_show_loading_fires_for_user_id():
    """show_loading() must POST for U-prefixed IDs (DM users) — guards against
    inverted prefix check that would silently disable the typing indicator."""
    route = respx.post("https://api.line.me/v2/bot/chat/loading/start").mock(
        return_value=Response(200, json={})
    )
    client = LineReplyClient(channel_access_token="t")
    await client.show_loading("U_user_id")
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["chatId"] == "U_user_id"


@pytest.mark.asyncio
@respx.mock
async def test_reply_raises_on_non_2xx():
    """LineReplyClient.reply() must surface LINE API errors via raise_for_status —
    documented contract; silently swallowing non-2xx would mask expired reply tokens."""
    import httpx
    respx.post("https://api.line.me/v2/bot/message/reply").mock(
        return_value=Response(400, json={"message": "Invalid reply token"})
    )
    client = LineReplyClient(channel_access_token="t")
    with pytest.raises(httpx.HTTPStatusError):
        await client.reply("expired-token", [{"type": "text", "text": "hi"}])
