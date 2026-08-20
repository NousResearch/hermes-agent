"""Regression: Mattermost's aiohttp session must use a keepalive-tuned
connector, not aiohttp's bare default.

Same class of issue as #18451 (CLOSE_WAIT accumulation from idle pooled
connections whose peer already closed them). That fix landed for the
httpx-based platform adapters (``gateway/platforms/_http_client_limits.py``)
and for Weixin's aiohttp connector (``gateway/platforms/weixin.py::_make_ssl_connector``,
#69089), but Mattermost was building a bare ``aiohttp.ClientSession()`` with
no connector tuning at all.
"""
import aiohttp
import pytest

from plugins.platforms.mattermost.adapter import _make_connector


@pytest.mark.asyncio
async def test_make_connector_tunes_keepalive_timeout():
    connector = _make_connector()
    try:
        assert isinstance(connector, aiohttp.TCPConnector)
        # aiohttp's own default is 15s -- too long for peers with a shorter
        # idle timeout. Mirrors the Weixin fix's keepalive_timeout=2.
        assert connector._keepalive_timeout == 2
        # enable_cleanup_closed=True -- default is disabled.
        assert connector._cleanup_closed_disabled is False
    finally:
        await connector.close()
