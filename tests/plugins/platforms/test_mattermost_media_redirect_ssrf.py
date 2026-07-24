"""Mattermost media download redirect SSRF invariants (salvage of #24831)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_download_media_url_blocks_redirect_to_metadata():
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = object.__new__(MattermostAdapter)

    public = "https://cdn.example.com/a.png"
    evil = "http://169.254.169.254/latest/meta-data/"

    redirect_resp = MagicMock()
    redirect_resp.status = 302
    redirect_resp.headers = {"Location": evil}
    redirect_resp.url = public
    redirect_resp.__aenter__ = AsyncMock(return_value=redirect_resp)
    redirect_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=redirect_resp)
    adapter._session = session

    with patch("tools.url_safety.is_safe_url", side_effect=lambda u: u == public):
        with pytest.raises(ValueError, match="blocked unsafe"):
            asyncio.run(adapter._download_media_url(public))
