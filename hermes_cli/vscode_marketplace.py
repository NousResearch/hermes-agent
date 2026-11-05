"""VS Code Marketplace color-theme fetcher and search proxy.

Provides the backend service for searching and downloading themes from the
VS Code Extension Marketplace so they can be installed across both the native
desktop app and the webapp.
"""

from __future__ import annotations

import io
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any

_log = logging.getLogger("hermes_cli.vscode_marketplace")

GALLERY_QUERY_URL = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery"
VSIX_ASSET_TYPE = "Microsoft.VisualStudio.Services.VSIXPackage"
MAX_VSIX_BYTES = 40 * 1024 * 1024  # 40 MB cap
REQUEST_TIMEOUT_SECONDS = 20

ID_RE = re.compile(r"^[\w-]+\.[\w-]+$")
ICON_THEME_RE = re.compile(r"\b(icon theme|file icons?|product icons?|icon pack|fileicons)\b", re.IGNORECASE)


class MarketplaceError(Exception):
    """Raised when marketplace communication or extraction fails."""


def _request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    max_bytes: int = MAX_VSIX_BYTES,
) -> bytes:
    req_headers = {"User-Agent": "Hermes-Desktop"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            content = resp.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise MarketplaceError("Response exceeded size limit")
            return content
    except urllib.error.HTTPError as exc:
        raise MarketplaceError(f"Marketplace request failed with status {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise MarketplaceError(f"Marketplace request timed out or failed: {exc}") from exc


def _query_gallery(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json;api-version=3.0-preview.1",
        "Content-Type": "application/json",
    }
    raw = _request(GALLERY_QUERY_URL, method="POST", headers=headers, data=body, max_bytes=4 * 1024 * 1024)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise MarketplaceError("Invalid JSON from Marketplace gallery") from exc


def _looks_like_icon_theme(extension: dict[str, Any]) -> bool:
    tags = [str(t).lower() for t in (extension.get("tags") or [])]
    if "icon-theme" in tags or "product-icon-theme" in tags:
        return True

    text = f"{extension.get('displayName') or ''} {extension.get('shortDescription') or ''}"
    return bool(ICON_THEME_RE.search(text))


def search_marketplace_themes(query: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Search the Marketplace for color-theme extensions."""
    text = (query or "").strip()
    page_size = min(max(int(limit) if str(limit).isdigit() else 20, 1), 50)

    criteria = [
        {"filterType": 8, "value": "Microsoft.VisualStudio.Code"},
        {"filterType": 5, "value": "Themes"},
        {"filterType": 12, "value": "4096"},  # Exclude unpublished
    ]
    if text:
        criteria.append({"filterType": 10, "value": text})

    payload = {
        "filters": [
            {
                "criteria": criteria,
                "pageNumber": 1,
                "pageSize": min(page_size * 2, 50),
                "sortBy": 4,
                "sortOrder": 0,
            }
        ],
        "flags": 772,  # IncludeStatistics | IncludeLatestVersionOnly | IncludeCategoryAndTags
    }

    data = _query_gallery(payload)
    results = data.get("results") or []
    extensions = (results[0].get("extensions") or []) if results else []

    items = []
    for ext in extensions:
        if _looks_like_icon_theme(ext):
            continue

        pub = ext.get("publisher") or {}
        pub_name = pub.get("publisherName") or ""
        ext_name = ext.get("extensionName") or ""
        display_name = ext.get("displayName") or ext_name
        description = ext.get("shortDescription") or ""

        stats = ext.get("statistics") or []
        install_stat = next((s for s in stats if s.get("statisticName") == "install"), None)
        installs = round(install_stat.get("value", 0)) if install_stat else 0

        items.append(
            {
                "extensionId": f"{pub_name}.{ext_name}",
                "displayName": display_name,
                "publisher": pub.get("displayName") or pub_name,
                "description": description,
                "installs": installs,
            }
        )

        if len(items) >= page_size:
            break

    return items


def _theme_entry_name(path: str) -> str:
    clean = str(path).lstrip("./").lstrip("/")
    return f"extension/{clean}"


def fetch_marketplace_themes(extension_id: str) -> dict[str, Any]:
    """Download and extract all contributed color themes for an extension id."""
    trimmed = (extension_id or "").strip()
    if not ID_RE.match(trimmed):
        raise MarketplaceError('Expected a Marketplace id like "publisher.extension"')

    payload = {
        "filters": [
            {
                "criteria": [{"filterType": 7, "value": trimmed}],
                "pageNumber": 1,
                "pageSize": 1,
            }
        ],
        "flags": 914,
    }

    data = _query_gallery(payload)
    results = data.get("results") or []
    extensions = (results[0].get("extensions") or []) if results else []
    if not extensions:
        raise MarketplaceError(f'Extension "{trimmed}" was not found on the Marketplace.')

    ext = extensions[0]
    display_name = ext.get("displayName") or trimmed
    versions = ext.get("versions") or []
    if not versions:
        raise MarketplaceError(f'Extension "{trimmed}" has no published versions.')

    files = versions[0].get("files") or []
    asset = next((f for f in files if f.get("assetType") == VSIX_ASSET_TYPE), None)
    if not asset or not asset.get("source"):
        raise MarketplaceError(f'Could not find a downloadable package for "{trimmed}".')

    vsix_url = asset["source"]
    vsix_bytes = _request(vsix_url)

    try:
        with zipfile.ZipFile(io.BytesIO(vsix_bytes)) as zf:
            try:
                pkg_data = zf.read("extension/package.json")
                pkg = json.loads(pkg_data.decode("utf-8"))
            except Exception as exc:
                raise MarketplaceError("Package manifest missing from extension") from exc

            contributed = (pkg.get("contributes") or {}).get("themes") or []
            if not isinstance(contributed, list) or not contributed:
                return {"extensionId": trimmed, "displayName": display_name, "themes": []}

            themes = []
            for entry in contributed:
                if not isinstance(entry, dict) or not entry.get("path"):
                    continue

                entry_name = _theme_entry_name(entry["path"])
                try:
                    theme_bytes = zf.read(entry_name)
                    themes.append(
                        {
                            "label": entry.get("label") or entry.get("id") or pkg.get("displayName") or pkg.get("name") or "VS Code Theme",
                            "uiTheme": entry.get("uiTheme"),
                            "contents": theme_bytes.decode("utf-8"),
                        }
                    )
                except KeyError:
                    continue
                except Exception as exc:
                    _log.debug("Failed to read theme entry %s: %s", entry_name, exc)

            return {
                "extensionId": trimmed,
                "displayName": display_name,
                "themes": themes,
            }
    except zipfile.BadZipFile as exc:
        raise MarketplaceError("Downloaded package is not a valid zip archive") from exc
