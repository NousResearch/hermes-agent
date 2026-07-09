"""Exercise the profile's document cap from config.yaml to the delivered Slack event."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,size,text,accepted,reads", [
    ("4", 1, "summarize", False, 2), ("4", 5, "summarize", False, 0),
    ("4", 1, "/help", False, 2), ("5", None, "summarize", True, 2),
    ("0", 20971521, "summarize", True, 2), ("-4", 20971521, "summarize", True, 2),
    ("invalid", 20971521, "summarize", False, 0), (None, 20971521, "summarize", False, 0),
])
async def test_configured_document_acceptance(tmp_path, monkeypatch, public_dns,
                                              setting, size, text, accepted, reads):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if setting is not None:
        (tmp_path / "config.yaml").write_text(f"gateway:\n  max_inbound_document_bytes: {setting}\n")
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._app = MagicMock()
    adapter._app.client.users_info = AsyncMock(return_value={"user": {"profile": {"display_name": "Alice"}}})
    adapter._bot_user_id = "U_BOT"
    adapter.handle_message = AsyncMock()
    seen = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in (b"123", b"45"):
                seen.append(chunk)
                yield chunk
            if setting == "4":
                pytest.fail("read past the first over-limit chunk")

    def serve(req):
        assert reads, "metadata should reject before requesting the file"
        return httpx.Response(200, headers={"content-length": "1"}, stream=Body())

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", lambda **kw:
                        httpx.AsyncClient(transport=httpx.MockTransport(serve), **kw))
    attachment = {"name": "doc.pdf" if accepted else "<@U1>`\nonce.pdf", "mimetype": "application/pdf",
                  "url_private": "https://files.slack.com/doc"}
    if size is not None:
        attachment["size"] = size
    await adapter._handle_slack_message({
        "files": [attachment], "text": text, "user": "U_USER", "channel": "D123",
        "channel_type": "im", "ts": "1234567890.000001"})
    msg = adapter.handle_message.await_args.args[0]
    assert len(seen) == reads
    assert bool(msg.media_urls) is accepted
    if accepted:
        assert Path(msg.media_urls[0]).read_bytes() == b"12345"
    else:
        assert not list((tmp_path / "doc_cache").glob("*"))
        if text.startswith("/"):
            assert msg.text == text and msg.message_type == MessageType.COMMAND
        else:
            assert "once.pdf" in msg.text and "split" in msg.text and text in msg.text
            assert "<@U1>" not in msg.text and "\nonce.pdf" not in msg.text
