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

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

CAP = 64 * 1024
PNG = b"\x89PNG\r\n\x1a\n"


class _ChunkedImage(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        size = int(parse_qs(urlparse(self.path).query)["size"][0])
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        body = PNG + b"\0" * (size - len(PNG))
        try:
            for i in range(0, size, 16 * 1024):
                piece = body[i:i + 16 * 1024]
                self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args):
        pass


@pytest.fixture
def image_host():
    from hermes_constants import get_hermes_home
    import tools.url_safety as url_safety

    (get_hermes_home() / "config.yaml").write_text(
        f"gateway:\n  max_inbound_media_bytes: {CAP}\nsecurity:\n  allow_private_urls: true\n")
    url_safety._reset_allow_private_cache()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedImage)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    url_safety._reset_allow_private_cache()


async def _ok(*_args, **_kwargs):
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
    from gateway.platforms.weixin import WeixinAdapter
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
        await send(adapter, f"{image_host}/huge.png?size={CAP * 8}")
        assert uploaded == [], f"an over-cap image reached the upload ({uploaded} bytes)"
    finally:
        for session in (getattr(adapter, "_session", None), getattr(adapter, "_send_session", None)):
            if session is not None:
                await session.close()
