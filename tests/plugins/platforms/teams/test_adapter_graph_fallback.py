"""Tests for inbound Graph hostedContents fallback (Task 8).

When a user shares an inline image (or document) in a Teams channel, the
direct download URL embedded in the activity is short-lived. By the time
the gateway issues a GET, the tempauth/bearer can already be rejected
(403 / 410 / 401) — even though the underlying ``hostedContents`` blob is
still addressable through Microsoft Graph.

These tests cover:

* :func:`_parse_hosted_content_id` — pure URL parser.
* :meth:`TeamsAdapter._try_graph_hosted_bytes` — Graph fallback that
  returns raw bytes (used by the file.download.info path).
* :meth:`TeamsAdapter._try_graph_hosted_fallback` — image-path entry
  point that runs the bytes fallback then caches the result.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.teams.adapter import (
    TeamsAdapter,
    _parse_hosted_content_id,
)


# ---------------------------------------------------------------------------
# Pure unit tests — _parse_hosted_content_id
# ---------------------------------------------------------------------------


def test_parse_hosted_content_id_extracts_id():
    url = (
        "https://graph.microsoft.com/v1.0/teams/T/channels/C/messages/"
        "123/hostedContents/abc-xyz/$value?token=foo"
    )
    assert _parse_hosted_content_id(url) == "abc-xyz"


def test_parse_hosted_content_id_returns_none_for_sharepoint_url():
    url = "https://contoso.sharepoint.com/sites/X/Documents/file.pdf?tempauth=Y..."
    assert _parse_hosted_content_id(url) is None


def test_parse_hosted_content_id_handles_empty_or_none():
    assert _parse_hosted_content_id("") is None
    assert _parse_hosted_content_id(None) is None  # type: ignore[arg-type]


def test_parse_hosted_content_id_with_terminating_slash_then_value():
    base = "https://graph.microsoft.com/v1.0/teams/T/channels/C/messages/M/hostedContents"
    assert _parse_hosted_content_id(f"{base}/abc/$value") == "abc"
    assert _parse_hosted_content_id(f"{base}/abc/$value?x=y") == "abc"
    assert _parse_hosted_content_id(f"{base}/abc?x=y") == "abc"


# ---------------------------------------------------------------------------
# Adapter fixture — minimal, no connect()
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter(monkeypatch) -> TeamsAdapter:
    for var in (
        "TEAMS_CLIENT_ID",
        "TEAMS_CLIENT_SECRET",
        "TEAMS_TENANT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "fake-client",
            "client_secret": "fake-secret",
            "tenant_id": "fake-tenant",
        },
    )
    return TeamsAdapter(cfg)


_HC_URL = (
    "https://graph.microsoft.com/v1.0/teams/T/channels/C/messages/M/"
    "hostedContents/abc-xyz/$value"
)


# ---------------------------------------------------------------------------
# _try_graph_hosted_bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_missing_context_returns_none(adapter):
    # _ensure_graph would raise if it were called (no real creds wired)
    ensure_graph = AsyncMock(side_effect=AssertionError("should not be called"))
    adapter._ensure_graph = ensure_graph  # type: ignore[method-assign]

    out = await adapter._try_graph_hosted_bytes(
        url=_HC_URL,
        team_id=None,
        channel_id="C",
        activity_id="A1",
    )
    assert out is None
    ensure_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_no_hosted_id_in_url_returns_none(adapter):
    ensure_graph = AsyncMock(side_effect=AssertionError("should not be called"))
    adapter._ensure_graph = ensure_graph  # type: ignore[method-assign]

    out = await adapter._try_graph_hosted_bytes(
        url="https://contoso.sharepoint.com/sites/X/Documents/file.pdf?tempauth=Y",
        team_id="T",
        channel_id="C",
        activity_id="A1",
    )
    assert out is None
    ensure_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_graph_unavailable_returns_none(adapter, caplog):
    adapter._ensure_graph = AsyncMock(  # type: ignore[method-assign]
        side_effect=ImportError("msgraph")
    )
    with caplog.at_level(logging.WARNING):
        out = await adapter._try_graph_hosted_bytes(
            url=_HC_URL,
            team_id="T",
            channel_id="C",
            activity_id="A1",
        )
    assert out is None
    assert "Graph not available" in caplog.text


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_graph_returns_none_returns_none(adapter, caplog):
    fake_graph = MagicMock()
    fake_graph.download_hosted_content = AsyncMock(return_value=None)
    adapter._ensure_graph = AsyncMock(return_value=fake_graph)  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        out = await adapter._try_graph_hosted_bytes(
            url=_HC_URL,
            team_id="T",
            channel_id="C",
            activity_id="A1",
        )
    assert out is None
    assert "recovered" not in caplog.text


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_graph_raises_returns_none_logs_exception(
    adapter, caplog
):
    fake_graph = MagicMock()
    fake_graph.download_hosted_content = AsyncMock(side_effect=ValueError("boom"))
    adapter._ensure_graph = AsyncMock(return_value=fake_graph)  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        out = await adapter._try_graph_hosted_bytes(
            url=_HC_URL,
            team_id="T",
            channel_id="C",
            activity_id="A1",
        )
    assert out is None
    assert "download_hosted_content raised" in caplog.text


@pytest.mark.asyncio
async def test_try_graph_hosted_bytes_happy_path(adapter, caplog):
    fake_graph = MagicMock()
    fake_graph.download_hosted_content = AsyncMock(return_value=b"abc123")
    adapter._ensure_graph = AsyncMock(return_value=fake_graph)  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        out = await adapter._try_graph_hosted_bytes(
            url=_HC_URL,
            team_id="T",
            channel_id="C",
            activity_id="A1",
        )
    assert out == b"abc123"
    fake_graph.download_hosted_content.assert_awaited_once_with(
        team_id="T",
        channel_id="C",
        message_id="A1",
        hosted_content_id="abc-xyz",
    )
    assert "recovered 6 bytes" in caplog.text


# ---------------------------------------------------------------------------
# _try_graph_hosted_fallback (image path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_try_graph_hosted_fallback_image_caches_and_returns_path(
    adapter, monkeypatch
):
    fake_graph = MagicMock()
    fake_graph.download_hosted_content = AsyncMock(return_value=b"PNGDATA")
    adapter._ensure_graph = AsyncMock(return_value=fake_graph)  # type: ignore[method-assign]

    captured: dict = {}

    def fake_cache(data: bytes, ext: str) -> str:
        captured["data"] = data
        captured["ext"] = ext
        return "/cache/abc.jpg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", fake_cache)

    out = await adapter._try_graph_hosted_fallback(
        idx=0,
        url=_HC_URL,
        team_id="T",
        channel_id="C",
        activity_id="A1",
        kind="image",
        ext=".jpg",
        filename="image.jpg",
    )
    assert out == "/cache/abc.jpg"
    assert captured == {"data": b"PNGDATA", "ext": ".jpg"}


@pytest.mark.asyncio
async def test_try_graph_hosted_fallback_image_cache_failure_returns_none(
    adapter, monkeypatch, caplog
):
    fake_graph = MagicMock()
    fake_graph.download_hosted_content = AsyncMock(return_value=b"PNGDATA")
    adapter._ensure_graph = AsyncMock(return_value=fake_graph)  # type: ignore[method-assign]

    def boom(data: bytes, ext: str) -> str:
        raise ValueError("cache pop")

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", boom)

    with caplog.at_level(logging.ERROR):
        out = await adapter._try_graph_hosted_fallback(
            idx=3,
            url=_HC_URL,
            team_id="T",
            channel_id="C",
            activity_id="A1",
            kind="image",
            ext=".jpg",
            filename="bad.jpg",
        )
    assert out is None
    assert "cache failed" in caplog.text
