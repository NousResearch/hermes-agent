"""Test doubles for the Matrix adapter's media download path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock


class FakeMediaDownload:
    """A homeserver media download endpoint behind a mocked mautrix client.

    ``install`` wires the client's ``versions`` call and HTTP session to this object. The
    homeserver reports Matrix v1.11 support, and therefore the authenticated media endpoint,
    only when ``authenticated_media`` is set. Each download serves ``body`` in ``chunk_size``
    pieces. The object records the requested MXC URIs, whether each request used the
    authenticated endpoint together with its headers, and the number of chunks that the
    adapter reads. With ``fail`` set, every download raises as a failed HTTP request does.
    """

    def __init__(
        self, body: bytes = b"media", *, chunk_size: int = 65536, send_content_length: bool = False,
        fail: bool = False, authenticated_media: bool = True,
    ) -> None:
        self.body = body
        self.chunk_size = chunk_size
        self.send_content_length = send_content_length
        self.fail = fail
        self.authenticated_media = authenticated_media
        self.requested: list[str] = []
        self.request_auth: list[tuple[bool, dict[str, str]]] = []
        self.chunks_read = 0

    def install(self, client) -> "FakeMediaDownload":
        client.versions = AsyncMock(
            return_value=SimpleNamespace(supports=lambda _version: self.authenticated_media))
        client.api.token = "syt_test_token"
        client.api.get_download_url = lambda mxc, authenticated=False: (mxc, authenticated)
        client.api.session.get = self._get
        return self

    def _get(self, url: tuple[str, bool], *, headers=None, **_kwargs) -> "_FakeMediaResponse":
        mxc, authenticated = url
        self.requested.append(mxc)
        self.request_auth.append((authenticated, dict(headers or {})))
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
