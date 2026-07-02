"""SSRF protection tests for yuanbao_media.download_url().

download_url() fetches both model-supplied (outbound) and inbound image/file
URLs server-side via httpx. Without an is_safe_url() pre-flight, a model
response (or inbound message) containing http://169.254.169.254/... would make
the gateway fetch cloud-metadata endpoints. These tests pin the guard.
"""

import pytest

from gateway.platforms.yuanbao_media import download_url


class TestDownloadUrlSSRF:
    @pytest.mark.asyncio
    async def test_metadata_endpoint_blocked(self):
        with pytest.raises(ValueError, match="SSRF protection"):
            await download_url("http://169.254.169.254/latest/meta-data/")

    @pytest.mark.asyncio
    async def test_loopback_blocked(self):
        with pytest.raises(ValueError, match="SSRF protection"):
            await download_url("http://127.0.0.1:8080/secret")

    @pytest.mark.asyncio
    async def test_private_range_blocked(self):
        with pytest.raises(ValueError, match="SSRF protection"):
            await download_url("http://192.168.1.1/admin/logo.png")

    @pytest.mark.asyncio
    async def test_non_http_scheme_blocked(self):
        with pytest.raises(ValueError, match="SSRF protection"):
            await download_url("file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_public_url_passes_guard_then_fetches(self, monkeypatch):
        """A public URL clears the SSRF guard and reaches the HTTP client.

        We stub is_safe_url True and the httpx client so no real network call
        happens — the assertion is that the guard does not reject a public URL.
        """
        import gateway.platforms.yuanbao_media as ym

        fetched = {}

        class _FakeResp:
            headers = {"content-type": "image/png", "content-length": "3"}
            is_redirect = False
            next_request = None

            def raise_for_status(self):
                pass

            async def aiter_bytes(self, _n):
                yield b"png"

        class _FakeStream:
            async def __aenter__(self):
                return _FakeResp()

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **kw):
                fetched["hooks"] = kw.get("event_hooks")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def head(self, url):
                return _FakeResp()

            def stream(self, method, url, **kw):
                fetched["url"] = url
                return _FakeStream()

        monkeypatch.setattr(ym, "is_safe_url", lambda u: True, raising=False)
        # is_safe_url is imported inside the function, so patch the source too
        from tools import url_safety
        monkeypatch.setattr(url_safety, "is_safe_url", lambda u: True)
        monkeypatch.setattr(ym.httpx, "AsyncClient", _FakeClient)

        data, ct = await download_url("https://example.com/image.png")
        assert data == b"png"
        assert ct == "image/png"
        # The guarded client must register a redirect event hook.
        assert fetched["hooks"] is not None
        assert "response" in fetched["hooks"]

    @pytest.mark.asyncio
    async def test_redirect_to_metadata_blocked(self, monkeypatch):
        """A 302 whose Location points at a metadata address is rejected even
        though httpx leaves response.next_request unset inside the hook.

        Regression: the guard used to key on response.next_request, which is
        None inside an httpx response event hook, so redirect-based SSRF (a
        public URL 302-ing to 169.254.169.254) was never blocked.
        """
        import gateway.platforms.yuanbao_media as ym
        from tools import url_safety

        # Public pre-flight passes; the redirect target does not.
        monkeypatch.setattr(
            url_safety, "is_safe_url", lambda u: str(u).startswith("https://example.com")
        )

        captured = {}

        class _FakeResp:
            headers = {"content-type": "image/png", "content-length": "3"}
            is_redirect = False
            next_request = None

            def raise_for_status(self):
                pass

            async def aiter_bytes(self, _n):
                yield b"png"

        class _FakeStream:
            async def __aenter__(self):
                return _FakeResp()

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **kw):
                captured["guard"] = kw["event_hooks"]["response"][0]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def head(self, url):
                return _FakeResp()

            def stream(self, method, url, **kw):
                return _FakeStream()

        monkeypatch.setattr(ym.httpx, "AsyncClient", _FakeClient)

        await download_url("https://example.com/image.png")
        guard = captured["guard"]

        class _RedirectResp:
            # A genuine 302 as seen inside the hook: Location set, next_request None.
            is_redirect = True
            url = "https://example.com/image.png"
            headers = {"location": "http://169.254.169.254/latest/meta-data/"}
            next_request = None

        with pytest.raises(ValueError, match="Blocked redirect"):
            await guard(_RedirectResp())
