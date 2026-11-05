"""Marketplace theme proxy endpoints.

Exposes search and fetch endpoints so the webapp renderer can install themes
from the VS Code Marketplace identically to the native desktop application.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hermes_cli import vscode_marketplace

router = APIRouter()
_log = logging.getLogger("hermes_cli.web_routers.themes")


@router.get("/api/themes/marketplace/search")
def search_marketplace_themes(
    q: str = Query(default="", alias="q"),
    limit: int = Query(default=20, ge=1, le=50),
) -> list[dict[str, Any]]:
    try:
        return vscode_marketplace.search_marketplace_themes(query=q, limit=limit)
    except vscode_marketplace.MarketplaceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        _log.warning("Marketplace search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Marketplace search failed") from exc


@router.get("/api/themes/marketplace/fetch")
def fetch_marketplace_theme(
    id: str = Query(..., alias="id"),
) -> dict[str, Any]:
    try:
        return vscode_marketplace.fetch_marketplace_themes(extension_id=id)
    except vscode_marketplace.MarketplaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.warning("Marketplace fetch failed for %s: %s", id, exc)
        raise HTTPException(status_code=500, detail="Marketplace fetch failed") from exc
