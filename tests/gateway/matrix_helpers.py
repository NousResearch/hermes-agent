"""Test doubles for the Matrix adapter's media download path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock


class FakeMediaDownload:
    """A homeserver media download endpoint behind a mocked mautrix client.

    ``install`` wires the client's ``versions`` call and HTTP session to this object. Each
    download serves ``body`` in ``chunk_size`` pieces. The object records the requested MXC
    URIs and counts the chunks that the adapter reads. With ``fail`` set, every download
    raises as a failed HTTP request does.
    """

    def __init__(
        self, body: bytes = b"media", *, chunk_size: int = 65536, send_content_length: bool = False,
        fail: bool = False,
    ) -> None:
        self.body = body
        self.chunk_size = chunk_size
        self.send_content_length = send_content_length
        self.fail = fail
        self.requested: list[str] = []
        self.chunks_read = 0

    def install(self, client) -> "FakeMediaDownload":
        client.versions = AsyncMock(return_value=SimpleNamespace(supports=lambda _version: True))
        client.api.token = "syt_test_token"
        client.api.get_download_url = lambda mxc, authenticated=False: mxc
        client.api.session.get = self._get
        return self

    def _get(self, url, **_kwargs) -> "_FakeMediaResponse":
        self.requested.append(url)
        return _FakeMediaResponse(self)


class _FakeMediaResponse:
    def __init__(self, download: FakeMediaDownload) -> None:
        self._download = download
        self.content = self
        self.content_length = len(download.body) if download.send_content_length else None

    async def __aenter__(self) -> "_FakeMediaResponse":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    def raise_for_status(self) -> None:
        if self._download.fail:
            raise RuntimeError("media download failed")

    async def iter_chunked(self, _size: int):
        body, step = self._download.body, self._download.chunk_size
        for offset in range(0, len(body), step):
            self._download.chunks_read += 1
            yield body[offset:offset + step]
