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


