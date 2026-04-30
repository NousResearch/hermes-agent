"""Tests for LINE send_message tool routing."""
from tests.gateway.conftest import make_line_platform_config
import json

import pytest
import respx
from httpx import Response


@pytest.mark.asyncio
@respx.mock
async def test_send_line_posts_to_push_api():
    """_send_line() must POST to LINE Push API with correct headers."""
    from tools.send_message_tool import _send_line

    route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )

    class _FakePConfig:
        token = "test-token"

    result = await _send_line(_FakePConfig(), "U123456789", "hello from cron")
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["to"] == "U123456789"
    assert body["messages"][0]["text"] == "hello from cron"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-token"
    assert result.get("success") is True


@pytest.mark.asyncio
@respx.mock
async def test_send_line_returns_error_without_token():
    """_send_line() must return an error dict when no token is configured."""
    from tools.send_message_tool import _send_line

    class _NullConfig:
        token = ""

    result = await _send_line(_NullConfig(), "U123", "hi")
    assert "error" in result
    assert "LINE" in result["error"]


@pytest.mark.asyncio
@respx.mock
async def test_adapter_send_image_https_url():
    """send_image() with HTTPS URL sends an image message via Push API."""
    from gateway.platforms.line import LineAdapter
    from gateway.platforms.base import SendResult

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "tok"
    os.environ["LINE_CHANNEL_SECRET"] = "sec"
    os.environ["LINE_ALLOWED_USERS"] = "U1"
    adapter = LineAdapter(make_line_platform_config(token="tok"))

    route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )

    result = await adapter.send_image("U1", "https://example.com/img.png", caption="hi")
    assert isinstance(result, SendResult)
    assert result.success
    body = json.loads(route.calls.last.request.content)
    assert body["to"] == "U1"
    assert body["messages"][0]["type"] == "image"
    assert body["messages"][0]["originalContentUrl"] == "https://example.com/img.png"
    # Caption arrives as a second text message
    assert body["messages"][1]["type"] == "text"
    assert body["messages"][1]["text"] == "hi"


@pytest.mark.asyncio
@respx.mock
async def test_adapter_send_image_http_url_falls_back_to_text():
    """send_image() with non-HTTPS URL degrades to a Push API text message."""
    from gateway.platforms.line import LineAdapter
    from gateway.platforms.base import SendResult

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "tok"
    os.environ["LINE_CHANNEL_SECRET"] = "sec"
    os.environ["LINE_ALLOWED_USERS"] = "U1"
    adapter = LineAdapter(make_line_platform_config(token="tok"))

    route = respx.post("https://api.line.me/v2/bot/message/push").mock(
        return_value=Response(200, json={})
    )

    result = await adapter.send_image("U1", "http://example.com/img.png")
    assert isinstance(result, SendResult)
    assert result.success
    body = json.loads(route.calls.last.request.content)
    # Non-HTTPS → text fallback, not image message
    assert body["messages"][0]["type"] == "text"
    assert "http://example.com/img.png" in body["messages"][0]["text"]


@pytest.mark.asyncio
async def test_adapter_send_image_file_returns_unsupported():
    """send_image_file() returns a non-fatal error — LINE has no local file upload."""
    from gateway.platforms.line import LineAdapter
    from gateway.platforms.base import SendResult

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "tok"
    os.environ["LINE_CHANNEL_SECRET"] = "sec"
    os.environ["LINE_ALLOWED_USERS"] = ""
    adapter = LineAdapter(make_line_platform_config(token="tok"))
    result = await adapter.send_image_file("U1", "/tmp/img.png", caption="cap")
    assert isinstance(result, SendResult)
    assert not result.success
    assert result.error and "HTTPS" in result.error


def test_build_reply_messages_extracts_markdown_image():
    """Agent output with ![alt](url) must yield a native LINE image message,
    not raw text. Regression guard: previously
    self._chunk_text(answer) sent the raw markdown to LINE which displayed
    it as text instead of an image bubble."""
    from gateway.platforms.line import LineAdapter

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "t"
    os.environ["LINE_CHANNEL_SECRET"] = "s"
    adapter = LineAdapter(make_line_platform_config(token="t"))
    msgs = adapter._build_reply_messages(
        "Here is the chart:\n![chart](https://example.com/chart.png)\nDone."
    )
    image_msgs = [m for m in msgs if m.get("type") == "image"]
    text_msgs = [m for m in msgs if m.get("type") == "text"]
    assert len(image_msgs) == 1
    assert image_msgs[0]["originalContentUrl"] == "https://example.com/chart.png"
    assert any("chart" in m.get("text", "") and "https://example.com" not in m.get("text", "")
               for m in text_msgs)


def test_build_reply_messages_caps_at_5_total():
    """LINE rejects > 5 messages per Reply/Push call. The builder must
    keep the total at or below 5 even when an agent dumps many images."""
    from gateway.platforms.line import LineAdapter

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "t"
    os.environ["LINE_CHANNEL_SECRET"] = "s"
    adapter = LineAdapter(make_line_platform_config(token="t"))
    text = "\n".join(f"![img{i}](https://example.com/img{i}.png)" for i in range(10))
    text += "\nFinal text after all images."
    msgs = adapter._build_reply_messages(text)
    assert len(msgs) <= 5


def test_build_reply_messages_falls_back_to_text_for_non_https():
    """LINE rejects non-HTTPS image URLs — those should stay in the cleaned
    text rather than producing an image message LINE will reject."""
    from gateway.platforms.line import LineAdapter

    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "t"
    os.environ["LINE_CHANNEL_SECRET"] = "s"
    adapter = LineAdapter(make_line_platform_config(token="t"))
    msgs = adapter._build_reply_messages(
        "![local](http://localhost/img.png)\nResponse text."
    )
    image_msgs = [m for m in msgs if m.get("type") == "image"]
    assert len(image_msgs) == 0
    # Cleaned text still goes through (markdown extracted but no image
    # message emitted because URL was non-HTTPS).
    assert any(m.get("type") == "text" for m in msgs)


def test_parse_target_ref_recognizes_line_chat_id_prefixes():
    """`send_message(target="line:U1234...")` etc. must be parsed as an
    explicit chat_id, not handed to the channel-name resolver."""
    from tools.send_message_tool import _parse_target_ref

    chat_id, thread_id, is_explicit = _parse_target_ref("line", "U1234567890abcdef1234567890abcdef")
    assert chat_id == "U1234567890abcdef1234567890abcdef"
    assert thread_id is None
    assert is_explicit is True

    chat_id, _, is_explicit = _parse_target_ref("line", "Cabcdef0123456789abcdef0123456789")
    assert chat_id == "Cabcdef0123456789abcdef0123456789"
    assert is_explicit is True

    chat_id, _, is_explicit = _parse_target_ref("line", "Rabcdef0123456789abcdef0123456789")
    assert chat_id == "Rabcdef0123456789abcdef0123456789"
    assert is_explicit is True

    # Non-LINE-prefixed strings fall through to channel-name resolution
    chat_id, _, is_explicit = _parse_target_ref("line", "team-channel-name")
    assert is_explicit is False
