"""Spotify Web Playback dashboard plugin — backend API routes.

Mounted at ``/api/plugins/spotify/`` by ``hermes_cli/web_server.py`` (see
``_mount_plugin_api_routes``).  The browser-side Web Playback SDK can't
read ``~/.hermes/auth.json`` directly; it asks our endpoint for a fresh
Spotify access token on init and again whenever its previous token
expires (the SDK calls a user-supplied ``getOAuthToken`` callback on
demand).  This module is the entire backend surface that callback needs.

The heavy lifting lives in ``hermes_cli/auth.py`` —
``resolve_spotify_runtime_credentials`` already handles "load state →
refresh if expiring → persist → return access_token".  We just expose
its result to a same-origin browser fetch and translate ``AuthError``
into HTTP status codes the JS layer can branch on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()


def _serialise_credentials(creds: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the refresh_token before returning to the browser.

    The browser only needs the short-lived access token; the refresh
    token is a long-lived secret that never leaves the host process.
    """
    return {
        "access_token": creds["access_token"],
        "token_type": creds.get("token_type", "Bearer"),
        "scope": creds.get("scope", ""),
        "expires_at": creds.get("expires_at"),
    }


@router.get("/token")
async def get_spotify_token() -> Dict[str, Any]:
    """Return a fresh Spotify access token for the Web Playback SDK.

    Refreshes via the existing OAuth state if the cached token is
    within the skew window of expiry.  Same code path as every other
    Spotify tool call, so the refresh stays consistent with the agent
    side.
    """
    try:
        from hermes_cli.auth import (
            AuthError,
            resolve_spotify_runtime_credentials,
        )
    except ImportError as exc:
        logger.warning("Spotify auth helpers unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "spotify_auth_unavailable",
                "message": "Spotify auth helpers are not available in this install.",
            },
        ) from exc

    try:
        creds = resolve_spotify_runtime_credentials(refresh_if_expiring=True)
    except AuthError as exc:
        # AuthError carries a stable `code` and a `relogin_required` flag
        # that the JS layer uses to decide between "show a refresh button"
        # vs. "tell the user to re-run `hermes auth spotify`".
        status = 401 if getattr(exc, "relogin_required", False) else 502
        raise HTTPException(
            status_code=status,
            detail={
                "code": getattr(exc, "code", "spotify_auth_failed"),
                "message": str(exc),
                "relogin_required": bool(getattr(exc, "relogin_required", False)),
            },
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error resolving Spotify credentials")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "spotify_auth_internal",
                "message": str(exc),
            },
        ) from exc

    return _serialise_credentials(creds)


@router.get("/status")
async def get_spotify_status() -> Dict[str, Any]:
    """Lightweight login-state probe — no token returned.

    The widget calls this before attempting SDK init so it can render
    the right initial state (logged out vs. logged in vs. config error)
    without exposing the access token in the status response.
    """
    try:
        from hermes_cli.auth import get_spotify_auth_status
    except ImportError:
        return {"logged_in": False, "available": False}

    try:
        status = get_spotify_auth_status()
    except Exception as exc:
        logger.warning("get_spotify_auth_status failed: %s", exc)
        return {"logged_in": False, "available": True, "error": str(exc)}

    return {
        "logged_in": bool(status.get("logged_in")),
        "available": True,
        # Pass through fields useful to the widget for diagnostic display,
        # but never the tokens themselves.
        "scope": status.get("scope") or status.get("granted_scope"),
        "expires_at": status.get("expires_at"),
    }
