"""Mattermost media download redirect SSRF invariants (salvage of #24831)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _redirecting_adapter():
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
    adapter.send = AsyncMock()
    adapter._upload_file = AsyncMock()
    return adapter, session, public, evil


def test_send_image_blocks_redirect_to_metadata():
    adapter, session, public, _evil = _redirecting_adapter()

    with patch(
        "tools.url_safety.async_is_safe_url",
        new=AsyncMock(side_effect=lambda url: url == public),
    ):
        asyncio.run(adapter.send_image("channel-1", public))

    assert session.get.call_args.kwargs["allow_redirects"] is False
    adapter.send.assert_awaited_once()


def test_send_multiple_images_blocks_redirect_to_metadata():
    adapter, session, public, _evil = _redirecting_adapter()

    with patch(
        "tools.url_safety.async_is_safe_url",
        new=AsyncMock(side_effect=lambda url: url == public),
    ):
        asyncio.run(adapter.send_multiple_images("channel-1", [(public, "alt")]))

    assert session.get.call_args.kwargs["allow_redirects"] is False
    adapter._upload_file.assert_not_awaited()
