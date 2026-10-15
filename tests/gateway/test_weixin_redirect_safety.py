"""Weixin media downloads must re-validate every redirect hop (#138362).

``_download_bytes`` used to let aiohttp follow redirects silently, so an
SSRF-checked / CDN-allowlisted first URL could 302 to an intranet or
cloud-metadata address. Redirects are now followed manually and every target
re-passes ``is_safe_url`` — fail-closed, mirroring the per-hop guard in
``gateway/platforms/base.py``.
"""

import asyncio

import pytest

from gateway.platforms import weixin


class _MediaResponse:
    """Minimal aiohttp response double: async CM with status/headers/read."""

    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def read(self):
        return self._body


def _redirect(location):
    return _MediaResponse(302, headers={"Location": location})


def _ok(body=b"payload"):
    return _MediaResponse(200, body=body)


class _RoutingSession:
    """Returns one queued response per GET, recording (url, kwargs)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._responses.pop(0)


_CDN_URL = "https://novac2c.cdn.weixin.qq.com/download?encrypted_query_param=abc"


class TestDownloadBytesRedirectSafety:
    def test_redirect_to_cloud_metadata_is_blocked(self):
        session = _RoutingSession([_redirect("http://169.254.169.254/latest/meta-data")])

        with pytest.raises(ValueError, match="Blocked redirect to unsafe media host"):
            asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))

    def test_redirect_to_loopback_is_blocked(self):
        session = _RoutingSession([_redirect("http://127.0.0.1/private.png")])

        with pytest.raises(ValueError, match="Blocked redirect to unsafe media host"):
            asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))

    def test_redirect_to_public_host_is_followed(self):
        session = _RoutingSession([_redirect("https://mmbiz.qpic.cn/photo.png"), _ok(b"payload")])

        result = asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))

        assert result == b"payload"
        assert [url for url, _ in session.get_calls] == [_CDN_URL, "https://mmbiz.qpic.cn/photo.png"]
        assert all(kwargs["allow_redirects"] is False for _, kwargs in session.get_calls)

    def test_relative_redirect_location_resolves_against_current_url(self):
        session = _RoutingSession([_redirect("/cdn-hop/photo.png"), _ok(b"payload")])

        result = asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))

        assert result == b"payload"
        assert session.get_calls[1][0] == "https://novac2c.cdn.weixin.qq.com/cdn-hop/photo.png"

    def test_redirect_chain_is_capped(self):
        responses = [_redirect(f"https://mmbiz.qpic.cn/hop-{i}.png") for i in range(12)]
        session = _RoutingSession(responses)

        with pytest.raises(ValueError, match="exceeded 10 redirects"):
            asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))

    def test_redirect_without_location_fails_closed(self):
        session = _RoutingSession([_MediaResponse(302)])

        with pytest.raises(ValueError, match="carried no Location header"):
            asyncio.run(weixin._download_bytes(session, url=_CDN_URL, timeout_seconds=5))


class TestInboundMediaRedirectSafety:
    def test_allowlisted_full_url_redirecting_to_private_is_blocked(self):
        session = _RoutingSession([_redirect("http://10.0.0.5/steal")])

        with pytest.raises(ValueError, match="Blocked redirect to unsafe media host"):
            asyncio.run(
                weixin._download_and_decrypt_media(
                    session,
                    cdn_base_url="https://novac2c.cdn.weixin.qq.com",
                    encrypted_query_param=None,
                    aes_key_b64=None,
                    full_url=_CDN_URL,
                    timeout_seconds=5,
                )
            )
