"""SSRF protection tests for gateway.platforms.yuanbao_media.download_url.

The download URL comes from model/agent output (e.g. an ``![alt](url)`` image
link in a reply that the gateway resolves and sends back). Without a guard, an
agent could make the gateway fetch internal services or the cloud metadata
endpoint, so ``download_url`` must reject private/internal targets and must
re-validate redirect hops.
"""

import pytest

from gateway.platforms import yuanbao_media


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://127.0.0.1:8080/admin",
        "http://localhost/secret",
        "http://[::1]/secret",
        "http://10.0.0.5/internal",
    ],
)
async def test_download_url_blocks_private_targets(url):
    with pytest.raises(ValueError, match="SSRF"):
        await yuanbao_media.download_url(url)


class _Resp:
    def __init__(self, redirect, next_url):
        self.is_redirect = redirect
        if next_url is None:
            self.next_request = None
        else:
            self.next_request = type("R", (), {"url": next_url})()


@pytest.mark.asyncio
async def test_redirect_guard_blocks_unsafe_redirect(monkeypatch):
    """A 30x redirect whose target fails is_safe_url must be rejected."""
    # Deterministic stub so the test does not depend on live DNS resolution.
    monkeypatch.setattr(
        "tools.url_safety.is_safe_url",
        lambda u: "evil-internal" not in u,
    )

    with pytest.raises(ValueError, match="private/internal"):
        await yuanbao_media._ssrf_redirect_guard(
            _Resp(True, "http://evil-internal/latest/meta-data/")
        )


@pytest.mark.asyncio
async def test_redirect_guard_allows_safe_redirect(monkeypatch):
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: True)
    # Safe redirect -> no raise.
    await yuanbao_media._ssrf_redirect_guard(
        _Resp(True, "https://cdn.example.com/image.png")
    )


@pytest.mark.asyncio
async def test_redirect_guard_noop_on_non_redirect():
    # A non-redirect response is a no-op even without next_request.
    await yuanbao_media._ssrf_redirect_guard(_Resp(False, None))
