import asyncio
import sys

def test_send_weixin_direct_does_not_reuse_live_session_across_loops(monkeypatch):
    if sys.version_info < (3, 10):
        # Hermes targets Python 3.10+. Some local runners still use 3.8/3.9.
        return
    from gateway.platforms import weixin as wx

    class DummySession:
        def __init__(self):
            self.closed = False
            self._loop = object()

    class DummyLiveAdapter:
        def __init__(self):
            self._send_session = DummySession()

        def format_message(self, content):
            return content or ""

        async def send(self, *_args, **_kwargs):
            raise AssertionError("live adapter should not be used across loops")

        async def send_image_file(self, *_args, **_kwargs):
            raise AssertionError("live adapter should not be used across loops")

        async def send_document(self, *_args, **_kwargs):
            raise AssertionError("live adapter should not be used across loops")

    called = {"fallback_send": 0}

    async def fake_send(self, *_args, **_kwargs):
        called["fallback_send"] += 1
        return wx.SendResult(success=True, message_id="mid")

    monkeypatch.setattr(wx.WeixinAdapter, "send", fake_send, raising=True)
    monkeypatch.setattr(wx, "_LIVE_ADAPTERS", {"tok": DummyLiveAdapter()}, raising=True)

    # Avoid touching real config or network in this unit test.
    monkeypatch.setattr(wx, "_make_ssl_connector", lambda: None, raising=True)

    async def run():
        return await wx.send_weixin_direct(
            extra={
                "account_id": "acc",
                "base_url": "https://example.invalid",
                "cdn_base_url": "https://cdn.example.invalid",
            },
            token="tok",
            chat_id="chat",
            message="hi",
            media_files=None,
        )

    result = asyncio.run(run())

    assert result.get("success") is True
    assert called["fallback_send"] == 1

