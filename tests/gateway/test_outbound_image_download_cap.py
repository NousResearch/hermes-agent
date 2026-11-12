"""Image URLs in an agent reply are downloaded by the adapter before upload. That download is
held in memory, so it must honour ``gateway.max_inbound_media_bytes`` like the inbound path:
an over-cap body (chunked, no Content-Length) is refused whole, and never uploaded truncated."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import SendResult

CAP = 64 * 1024
HUGE = CAP * 512  # 32 MiB: far more than socket buffers absorb once the client stops reading
PNG = b"\x89PNG\r\n\x1a\n"


class _ChunkedImage(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    served: dict[str, int] = {}  # request path -> bytes the server managed to write
    finished: dict[str, threading.Event] = {}  # request path -> set once its handler returns

    @classmethod
    def expect(cls, path: str) -> threading.Event:
        """Fresh completion event for ``path``; ``served[path]`` is final once it is set."""
        cls.served.pop(path, None)
        cls.finished[path] = threading.Event()
        return cls.finished[path]

    def do_GET(self):
        size = int(parse_qs(urlparse(self.path).query)["size"][0])
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        written = 0
        try:
            for i in range(0, size, 16 * 1024):
                piece = (PNG if i == 0 else b"") + b"\0" * (min(16 * 1024, size - i) - (len(PNG) if i == 0 else 0))
                self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                written += len(piece)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.served[self.path] = written
            if (event := self.finished.get(self.path)) is not None:
                event.set()

    def log_message(self, *_args):
        pass


@pytest.fixture
def image_host():
    from hermes_agent.hermes_constants import get_hermes_home
    from hermes_agent.tools import url_safety

    (get_hermes_home() / "config.yaml").write_text(
        f"gateway:\n  max_inbound_media_bytes: {CAP}\nsecurity:\n  allow_private_urls: true\n")
    url_safety._reset_allow_private_cache()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedImage)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    url_safety._reset_allow_private_cache()


TEXTS: list[str] = []  # what each adapter's ``send`` (text fallback) carried


async def _ok(*args, **kwargs):
    TEXTS.append(str(kwargs.get("content", args[1] if len(args) > 1 else "")))
    return SendResult(success=True, message_id="1")


def _slack(uploaded):
    from plugins.platforms.slack.adapter import SlackAdapter
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    a._app = MagicMock()
    a.send = _ok

    async def _dm_target(chat_id, _metadata):
        return chat_id

    async def _upload_with_retry(*_args, content=None, **_kwargs):
        uploaded.append(len(content))
        return SendResult(success=True, message_id="1")

    async def _files_upload_v2(file_uploads, **_kwargs):
        uploaded.extend(len(u["content"]) for u in file_uploads)
        return {"ok": True}

    a._dm_target, a._upload_with_retry = _dm_target, _upload_with_retry
    a._client_for = lambda *_a: SimpleNamespace(files_upload_v2=_files_upload_v2)
    a._record_uploaded_file_thread = lambda *_a: None
    return a


def _telegram(uploaded):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    a = TelegramAdapter(PlatformConfig(enabled=True, token="123:fake"))
    a.send = _ok

    async def send_photo(**kw):
        if isinstance(kw["photo"], str):  # Telegram could not fetch the URL itself
            raise RuntimeError("Bad Request: failed to get HTTP URL content")
        uploaded.append(len(kw["photo"]))
        return SimpleNamespace(message_id=1)

    async def _send_media(send_fn, *_args, **kw):
        return await send_fn(**kw)

    a._bot, a._send_media = SimpleNamespace(send_photo=send_photo), _send_media
    return a


def _feishu(uploaded):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    a = FeishuAdapter(PlatformConfig())
    a.send = _ok

    async def _send_file(chat_id, file_path=None, image_path=None, **_kwargs):
        uploaded.append(Path(file_path or image_path).stat().st_size)
        return SendResult(success=True, message_id="1")

    a.send_document = a.send_image_file = _send_file
    return a


def _mattermost(uploaded):
    import aiohttp
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    a = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": "https://mm.example.com"}))
    a._session = aiohttp.ClientSession()
    a.send = _ok

    async def _upload_file(_chat_id, data, *_args):
        uploaded.append(len(data))
        return "fid"

    async def _post(*_args, **_kwargs):
        return {"id": "post"}

    a._upload_file, a._post_message, a._post_with_file = _upload_file, _post, _ok
    return a


def _weixin(uploaded):
    import aiohttp
    from plugins.platforms.weixin.adapter import WeixinAdapter
    a = WeixinAdapter(PlatformConfig(enabled=True, token="t", extra={"account_id": "acct"}))
    a._send_session = aiohttp.ClientSession()
    a.send = _ok

    async def send_document(_chat_id, file_path, **_kwargs):
        uploaded.append(Path(file_path).stat().st_size)
        return SendResult(success=True, message_id="1")

    a.send_document = send_document
    return a


# path id -> (adapter factory, how the reply's image URL reaches the adapter)
SENDS = {
    "slack-send_image": (_slack, lambda a, url: a.send_image("C1", url)),
    "slack-send_multiple_images": (_slack, lambda a, url: a.send_multiple_images("C1", [(url, "")])),
    "telegram-send_image": (_telegram, lambda a, url: a.send_image("42", url)),
    "feishu-send_animation": (_feishu, lambda a, url: a.send_animation("oc_1", url)),
    "mattermost-send_image": (_mattermost, lambda a, url: a.send_image("ch", url)),
    "mattermost-send_multiple_images": (
        _mattermost, lambda a, url: a.send_multiple_images("ch", [(url, "")])),
    "weixin-send_multiple_images": (_weixin, lambda a, url: a.send_multiple_images("wx", [(url, "")])),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(SENDS))
async def test_reply_image_url_is_uploaded_whole_under_the_cap_and_refused_over_it(image_host, path):
    make, send = SENDS[path]
    uploaded: list[int] = []
    adapter = make(uploaded)
    try:
        await send(adapter, f"{image_host}/ok.png?size={CAP // 2}")
        assert uploaded == [CAP // 2], "an under-cap image must still be uploaded whole"

        uploaded.clear()
        TEXTS.clear()
        huge_path = f"/huge.png?size={HUGE}"
        huge_done = _ChunkedImage.expect(huge_path)
        await send(adapter, f"{image_host}{huge_path}")
        assert uploaded == [], f"an over-cap image reached the upload ({uploaded} bytes)"
        # The refused image already left the reply text, so its link survives as text.
        assert any("huge.png" in text for text in TEXTS), f"the refused image vanished: {TEXTS}"
    finally:
        for session in (getattr(adapter, "_session", None), getattr(adapter, "_send_session", None)):
            if session is not None:
                await session.close()
    # The download stops near the cap: buffering the whole body and rejecting it after would let
    # the server finish writing every byte (an aiohttp session holds the socket until it closes).
    assert huge_done.wait(10)
    assert _ChunkedImage.served[huge_path] < HUGE // 2, "the over-cap body was read to the end"


def _failing_batch(make):
    """Adapter whose one-message batch post raises, forcing the per-image fallback."""
    def factory(uploaded):
        adapter = make(uploaded)

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("batch post failed")

        if hasattr(adapter, "_post_message"):  # Mattermost
            adapter._post_message = _boom
        else:  # Slack
            adapter._client_for = lambda *_a: SimpleNamespace(files_upload_v2=_boom)
        return adapter
    return factory


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [_slack, _mattermost], ids=["slack", "mattermost"])
async def test_batch_fallback_does_not_resend_links_already_sent_as_text(image_host, make):
    """A refused image's link goes out as text once; when the batch post then fails, the
    per-image fallback retries only the images that are still owed."""
    uploaded: list[int] = []
    adapter = _failing_batch(make)(uploaded)
    TEXTS.clear()
    try:
        result = await adapter.send_multiple_images("C1", [
            (f"{image_host}/huge.png?size={HUGE}", "too big"),
            (f"{image_host}/ok.png?size={CAP // 2}", "fine"),
        ])
    finally:
        if getattr(adapter, "_session", None) is not None:
            await adapter._session.close()
    assert result.success
    assert sum("huge.png" in text for text in TEXTS) == 1, f"refused link sent more than once: {TEXTS}"
    assert CAP // 2 in uploaded, "the remaining image was not retried"
