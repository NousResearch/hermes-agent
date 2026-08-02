"""WhatsApp Cloud Graph media URL SSRF invariant."""

import asyncio
from unittest.mock import AsyncMock, MagicMock


def test_download_media_rejects_unsafe_temp_url(monkeypatch):
    from gateway.platforms import whatsapp_cloud as mod

    adapter = object.__new__(mod.WhatsAppCloudAdapter)
    adapter._http_client = MagicMock()
    adapter._access_token = "token"
    adapter._api_version = "v20.0"

    class _MetaResp:
        status_code = 200

        def json(self):
            return {"url": "http://169.254.169.254/latest/meta-data/", "mime_type": "image/jpeg"}

    async def fake_get(url, headers=None, follow_redirects=None):
        assert "169.254" not in url  # must not fetch the unsafe temp URL
        return _MetaResp()

    adapter._http_client.get = AsyncMock(side_effect=fake_get)
    monkeypatch.setattr(mod, "GRAPH_API_BASE", "https://graph.facebook.com")

    path, mime = asyncio.run(adapter._download_media_to_cache("MEDIA123"))
    assert path is None
    assert mime is None
    # Only the Graph metadata lookup should have been attempted.
    assert adapter._http_client.get.await_count == 1
