"""Microsoft Graph → SharePoint (OneDrive) file upload, for sending files into Teams chats.

Bots can't post files via ``/me/drive``; group chats and channels need the file
uploaded to a SharePoint site (configured via ``sharePointSiteId``) and shared with
a link + a native Teams file card. The flow mirrors the Graph contract:

    upload → sharing link → driveItem (eTag/webDavUrl) → file-info card

Requires the bot's AAD app to hold the Graph **application** permission
``Sites.ReadWrite.All`` (admin-consented). When unconfigured, callers fall back to
text-only delivery. Uses Graph *simple* upload (≤4 MB) — fine for minutes/cards;
larger files would need an upload session (follow-up).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
FILE_INFO_CARD_TYPE = "application/vnd.microsoft.teams.card.file.info"
UPLOAD_FOLDER = "HermesShared"  # organizes bot-uploaded files in the site drive


def resolve_sharepoint(extra: Optional[dict] = None) -> Optional[dict]:
    """Resolve ``{site_id, client_id, client_secret, tenant_id}`` or ``None``.

    Site id from ``TEAMS_SHAREPOINT_SITE_ID`` / config ``share_point_site_id``
    (``sharePointSiteId`` accepted too); credentials reuse the Bot Framework app."""
    extra = extra or {}
    site_id = (
        os.getenv("TEAMS_SHAREPOINT_SITE_ID")
        or extra.get("share_point_site_id")
        or extra.get("sharePointSiteId")
        or ""
    ).strip()
    client_id = (os.getenv("TEAMS_CLIENT_ID") or extra.get("client_id") or "").strip()
    client_secret = (os.getenv("TEAMS_CLIENT_SECRET") or extra.get("client_secret") or "").strip()
    tenant_id = (os.getenv("TEAMS_TENANT_ID") or extra.get("tenant_id") or "").strip()
    if not (site_id and client_id and client_secret and tenant_id):
        return None
    return {"site_id": site_id, "client_id": client_id, "client_secret": client_secret, "tenant_id": tenant_id}


def build_file_info_card(props: dict) -> dict:
    """Build a native Teams file-info card attachment from driveItem properties."""
    raw_etag = str(props.get("eTag") or "")
    unique_id = raw_etag.strip("\"'").replace("{", "").replace("}", "").split(",")[0] or raw_etag
    name = str(props.get("name") or "file")
    dot = name.rfind(".")
    file_type = name[dot + 1:].lower() if dot >= 0 else ""
    return {
        "contentType": FILE_INFO_CARD_TYPE,
        "contentUrl": props.get("webDavUrl"),
        "name": name,
        "content": {"uniqueId": unique_id, "fileType": file_type},
    }


async def _graph_token(session, creds: dict) -> Optional[str]:
    url = f"https://login.microsoftonline.com/{creds['tenant_id']}/oauth2/v2.0/token"
    async with session.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "scope": GRAPH_SCOPE,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ) as r:
        if r.status >= 400:
            logger.warning("[teams] graph token failed (%s): %s", r.status, (await r.text())[:200])
            return None
        return (await r.json()).get("access_token")


async def _upload(session, token: str, site_id: str, filename: str, content: bytes, content_type: str) -> Optional[dict]:
    path = f"/{UPLOAD_FOLDER}/{quote(filename)}"
    url = f"{GRAPH_ROOT}/sites/{site_id}/drive/root:{path}:/content"
    async with session.put(
        url, data=content,
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
    ) as r:
        if r.status >= 400:
            logger.warning("[teams] sharepoint upload failed (%s): %s", r.status, (await r.text())[:200])
            return None
        return await r.json()


async def _sharing_link(session, token: str, site_id: str, item_id: str) -> Optional[str]:
    url = f"{GRAPH_ROOT}/sites/{site_id}/drive/items/{item_id}/createLink"
    async with session.post(
        url, json={"type": "view", "scope": "organization"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    ) as r:
        if r.status >= 400:
            return None
        return ((await r.json()).get("link") or {}).get("webUrl")


async def _drive_item_props(session, token: str, site_id: str, item_id: str) -> Optional[dict]:
    url = f"{GRAPH_ROOT}/sites/{site_id}/drive/items/{item_id}?$select=eTag,webDavUrl,name"
    async with session.get(url, headers={"Authorization": f"Bearer {token}"}) as r:
        if r.status >= 400:
            return None
        return await r.json()


async def upload_and_build_card(
    creds: dict, filename: str, content: bytes, content_type: str = "application/octet-stream"
) -> Optional[dict]:
    """Upload ``content`` to SharePoint and return ``{card, share_url, name}`` or None.

    Best-effort: any Graph failure returns ``None`` so the caller degrades to text."""
    try:
        import aiohttp
    except ImportError:
        return None
    timeout = aiohttp.ClientTimeout(total=30.0)
    try:
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            token = await _graph_token(session, creds)
            if not token:
                return None
            uploaded = await _upload(session, token, creds["site_id"], filename, content, content_type)
            if not uploaded or not uploaded.get("id"):
                return None
            item_id = uploaded["id"]
            share_url = await _sharing_link(session, token, creds["site_id"], item_id)
            props = await _drive_item_props(session, token, creds["site_id"], item_id)
            if not props or not props.get("webDavUrl"):
                return None
            return {"card": build_file_info_card(props), "share_url": share_url or uploaded.get("webUrl"), "name": props.get("name") or filename}
    except Exception:  # noqa: BLE001 — file delivery is best-effort
        logger.warning("[teams] sharepoint upload_and_build_card failed", exc_info=True)
        return None
