"""Tests for the narrow Microsoft Graph client wrapper."""
from __future__ import annotations

import sys
import time
import types
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


# ---------------------------------------------------------------------------
# Stub azure.core.credentials.AccessToken before importing graph.py.
#
# The Graph client wrapper lazy-imports ``from azure.core.credentials import
# AccessToken`` inside ``_HermesTokenCredential.get_token``. The azure-core
# package is in the optional ``teams-files`` extra and is not installed in
# the default dev venv, so we provide a minimal stand-in. This keeps the
# tests honest about the boundary (``AccessToken`` is just a named tuple of
# ``(token, expires_on)``) without forcing every developer to install the
# Azure SDK to run the unit suite.
# ---------------------------------------------------------------------------
if "azure.core.credentials" not in sys.modules:
    _FakeAccessToken = namedtuple("AccessToken", ["token", "expires_on"])
    azure_pkg = sys.modules.setdefault("azure", types.ModuleType("azure"))
    azure_pkg.__path__ = []  # mark as package
    azure_core_pkg = sys.modules.setdefault("azure.core", types.ModuleType("azure.core"))
    azure_core_pkg.__path__ = []
    azure_creds_mod = types.ModuleType("azure.core.credentials")
    azure_creds_mod.AccessToken = _FakeAccessToken
    sys.modules["azure.core.credentials"] = azure_creds_mod


from plugins.platforms.teams.auth_graph import AuthError  # noqa: E402
from plugins.platforms.teams.graph import (  # noqa: E402
    GRAPH_SCOPE,
    GraphClient,
    _HermesTokenCredential,
    _attr,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

def test_graph_scope_constant():
    assert GRAPH_SCOPE == "https://graph.microsoft.com/.default"


# ---------------------------------------------------------------------------
# _HermesTokenCredential
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hermes_token_credential_returns_access_token():
    provider = Mock()
    provider.get_token = AsyncMock(return_value="abc")

    cred = _HermesTokenCredential(provider)
    before = int(time.time())
    tok = await cred.get_token("scope1")

    assert tok.token == "abc"
    # expires_on should be roughly +3000 seconds from now
    assert tok.expires_on >= before + 2900
    assert tok.expires_on <= before + 3100
    provider.get_token.assert_awaited_once_with("scope1")


@pytest.mark.asyncio
async def test_hermes_token_credential_no_scopes_raises():
    provider = Mock()
    provider.get_token = AsyncMock(return_value="abc")

    cred = _HermesTokenCredential(provider)

    with pytest.raises(AuthError):
        await cred.get_token()


@pytest.mark.asyncio
async def test_hermes_token_credential_close_is_noop():
    cred = _HermesTokenCredential(Mock())
    assert await cred.close() is None


@pytest.mark.asyncio
async def test_hermes_token_credential_uses_first_scope_only():
    provider = Mock()
    provider.get_token = AsyncMock(return_value="multi")

    cred = _HermesTokenCredential(provider)
    tok = await cred.get_token("first-scope", "second-scope")

    assert tok.token == "multi"
    provider.get_token.assert_awaited_once_with("first-scope")


# ---------------------------------------------------------------------------
# _attr helper
# ---------------------------------------------------------------------------

def test_attr_tolerates_dict_and_object_and_none():
    # dict
    assert _attr({"a": 1}, "a") == 1
    assert _attr({"a": 1}, "missing", default="x") == "x"
    # object
    obj = Mock(spec=["foo"])
    obj.foo = "bar"
    assert _attr(obj, "foo") == "bar"
    assert _attr(obj, "missing", default="dflt") == "dflt"
    # None
    assert _attr(None, "anything") is None
    assert _attr(None, "anything", default="z") == "z"


# ---------------------------------------------------------------------------
# GraphClient — lazy SDK import
# ---------------------------------------------------------------------------

def test_graph_client_lazy_builds_msgraph():
    # If msgraph was previously imported (by other tests), pop it so we can
    # confirm GraphClient construction itself does not import it.
    sys.modules.pop("msgraph", None)

    g = GraphClient(provider=Mock())
    assert g._client is None
    # Constructing GraphClient must NOT have triggered the heavy import.
    assert "msgraph" not in sys.modules


# ---------------------------------------------------------------------------
# download_hosted_content
# ---------------------------------------------------------------------------

def _build_hosted_chain(return_value=b"PNG-bytes", side_effect=None):
    """Build a fluent fake chain for download_hosted_content."""
    fake_client = MagicMock()
    get = AsyncMock(return_value=return_value)
    if side_effect is not None:
        get.side_effect = side_effect
        get.return_value = None
    (
        fake_client.teams.by_team_id.return_value
        .channels.by_channel_id.return_value
        .messages.by_chat_message_id.return_value
        .hosted_contents.by_chat_message_hosted_content_id.return_value
        .content.get
    ) = get
    return fake_client, get


@pytest.mark.asyncio
async def test_download_hosted_content_returns_bytes():
    fake_client, _get = _build_hosted_chain(return_value=b"PNG-bytes")
    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    result = await g.download_hosted_content("t", "c", "m", "h")

    assert result == b"PNG-bytes"
    fake_client.teams.by_team_id.assert_called_once_with("t")
    fake_client.teams.by_team_id.return_value.channels.by_channel_id.assert_called_once_with("c")
    (
        fake_client.teams.by_team_id.return_value
        .channels.by_channel_id.return_value
        .messages.by_chat_message_id.assert_called_once_with("m")
    )
    (
        fake_client.teams.by_team_id.return_value
        .channels.by_channel_id.return_value
        .messages.by_chat_message_id.return_value
        .hosted_contents.by_chat_message_hosted_content_id.assert_called_once_with("h")
    )


@pytest.mark.asyncio
async def test_download_hosted_content_returns_none_on_error(caplog):
    fake_client, _get = _build_hosted_chain(side_effect=Exception("graph 404"))
    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        result = await g.download_hosted_content("t", "c", "m", "h")

    assert result is None
    assert any("download_hosted_content" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# upload_to_sharepoint
# ---------------------------------------------------------------------------

def _build_upload_chain(put_return=None, put_side_effect=None):
    """Build a fluent fake chain for upload_to_sharepoint."""
    fake_client = MagicMock()
    put = AsyncMock(return_value=put_return)
    if put_side_effect is not None:
        put.side_effect = put_side_effect
    (
        fake_client.sites.by_site_id.return_value
        .drive.items.by_drive_item_id.return_value
        .content.put
    ) = put
    return fake_client, put


@pytest.mark.asyncio
async def test_upload_to_sharepoint_returns_web_url():
    result_obj = Mock()
    result_obj.web_url = "https://example.com/x.png"
    result_obj._raw_url = None
    fake_client, _put = _build_upload_chain(put_return=result_obj)

    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    url = await g.upload_to_sharepoint("site", "folder", "x.png", b"data")

    assert url == "https://example.com/x.png"
    fake_client.sites.by_site_id.assert_called_once_with("site")
    fake_client.sites.by_site_id.return_value.drive.items.by_drive_item_id.assert_called_once_with(
        "root:/folder/x.png:"
    )


@pytest.mark.asyncio
async def test_upload_to_sharepoint_strips_leading_trailing_slashes_in_folder():
    result_obj = Mock()
    result_obj.web_url = "https://example.com/file.png"
    result_obj._raw_url = None
    fake_client, _put = _build_upload_chain(put_return=result_obj)

    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    await g.upload_to_sharepoint("site", "/sub/dir/", "file.png", b"data")

    fake_client.sites.by_site_id.return_value.drive.items.by_drive_item_id.assert_called_once_with(
        "root:/sub/dir/file.png:"
    )


@pytest.mark.asyncio
async def test_upload_to_sharepoint_empty_folder_uses_root():
    result_obj = Mock()
    result_obj.web_url = "https://example.com/file.png"
    result_obj._raw_url = None

    for folder in ("", "/"):
        fake_client, _put = _build_upload_chain(put_return=result_obj)
        g = GraphClient(provider=Mock())
        g._build_client = lambda: fake_client  # type: ignore[method-assign]

        await g.upload_to_sharepoint("site", folder, "file.png", b"data")

        fake_client.sites.by_site_id.return_value.drive.items.by_drive_item_id.assert_called_once_with(
            "root:/file.png:"
        )


@pytest.mark.asyncio
async def test_upload_to_sharepoint_falls_back_to_raw_url():
    result_obj = Mock()
    result_obj.web_url = None
    result_obj._raw_url = "https://raw"
    fake_client, _put = _build_upload_chain(put_return=result_obj)

    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    url = await g.upload_to_sharepoint("site", "folder", "x.png", b"data")

    assert url == "https://raw"


@pytest.mark.asyncio
async def test_upload_to_sharepoint_returns_none_on_error(caplog):
    fake_client, _put = _build_upload_chain(put_side_effect=Exception("graph 500"))

    g = GraphClient(provider=Mock())
    g._build_client = lambda: fake_client  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        url = await g.upload_to_sharepoint("site", "folder", "x.png", b"data")

    assert url is None
    assert any("upload_to_sharepoint" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Caching of built client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_client_caches_built_client():
    fake_client, _get = _build_hosted_chain(return_value=b"x")
    g = GraphClient(provider=Mock())
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return fake_client

    g._build_client = builder  # type: ignore[method-assign]

    await g.download_hosted_content("t", "c", "m", "h")
    await g.download_hosted_content("t", "c", "m", "h")

    assert calls["n"] == 1
