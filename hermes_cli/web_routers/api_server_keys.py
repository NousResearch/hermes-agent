"""Dashboard admin endpoints for managed Hermes API keys.

Mounted by ``hermes_cli/web_server.py`` behind the existing Dashboard session
token / OAuth gate (``_require_token``) — same auth mechanism every other
``/api/*`` admin route uses. The plaintext secret is NEVER returned from
``GET`` or ``DELETE``; only the create endpoint emits it (once, at creation
time, so the operator can paste it into the desktop app).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from hermes_cli import api_server_keys as _keys
from hermes_cli.web_routers._common import http_failure

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()


# -- request/response models --------------------------------------------------

class CreateApiKeyBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)


class ApiKeyPublic(BaseModel):
    id: str
    name: str
    description: str
    prefix: str
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None
    active: bool


class CreateApiKeyResponse(ApiKeyPublic):
    plaintext: str
    """The full bearer secret. The Dashboard renders this ONCE in a copy
    dialog and never asks the backend for it again."""


# -- base URL derivation ------------------------------------------------------

def _api_server_base_url(request: Request) -> str:
    """Derive the externally-facing base URL of the Hermes OpenAI-compatible
    API server. We don't bind the gateway from inside the dashboard process;
    instead we read the operator's configured host/port (the same sources the
    gateway's ``listen_address`` consults) and combine with the inbound
    dashboard request's scheme so the operator sees the URL the desktop app
    should hit.

    Order of precedence:

    1. ``HERMES_API_SERVER_PUBLIC_URL`` env var (explicit override).
    2. ``gateway.api_server.host`` / ``gateway.api_server.port`` from
       ``~/.hermes/config.yaml`` (the values the gateway is actually binding).
    3. ``API_SERVER_HOST`` / ``API_SERVER_PORT`` env vars.
    4. ``127.0.0.1:8642`` defaults — the loopback test listener.

    The path suffix is always ``/v1`` (the OpenAI-compatible surface).
    """
    explicit = (os.environ.get("HERMES_API_SERVER_PUBLIC_URL") or "").strip()
    if explicit:
        # Strip trailing ``/v1`` or ``/`` so we can re-add ``/v1`` ourselves.
        cleaned = explicit.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[:-3]
        return f"{cleaned}/v1"

    host, port = _resolve_api_server_address()
    scheme = request.url.scheme or "http"
    return f"{scheme}://{host}:{port}/v1"


def _resolve_api_server_address() -> tuple[str, str]:
    """Mirror of ``gateway.platforms.api_server.listen_address``'s precedence:
    config extras → ``API_SERVER_HOST``/``API_SERVER_PORT`` env → defaults."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    extra = ((cfg.get("gateway") or {}).get("api_server")) or {}
    host = (
        extra.get("host")
        or os.environ.get("API_SERVER_HOST")
        or "127.0.0.1"
    )
    raw_port = (
        extra.get("port")
        or os.environ.get("API_SERVER_PORT")
        or "8642"
    )
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = 8642
    if not (1 <= port <= 65535):
        port = 8642
    return str(host), str(port)


# -- endpoints ----------------------------------------------------------------

@router.get("/api/api-server/info")
async def api_server_info(request: Request):
    """Lightweight metadata for the Dashboard's API Keys page header.

    Returns the public base URL plus a ``legacy_configured`` boolean so the
    UI can show ``Legacy API_SERVER_KEY is set'' without an extra round trip.
    Plaintext secrets are NEVER returned from this endpoint.
    """
    legacy_configured = bool((os.environ.get("API_SERVER_KEY") or "").strip())
    return {
        "base_url": _api_server_base_url(request),
        "legacy_configured": legacy_configured,
        "managed_active": _keys.has_active_keys(),
    }


@router.get("/api/api-server/keys")
async def list_managed_keys(include_revoked: bool = False):
    """List managed API keys. NEVER returns plaintext or secret material."""
    try:
        rows = _keys.list_keys(include_revoked=include_revoked)
    except Exception as exc:
        _log.warning("list_managed_keys failed: %s", exc)
        with http_failure("GET /api/api-server/keys failed", 500,
                          detail="Failed to list API keys"):
            pass
    return {"keys": rows}


@router.post("/api/api-server/keys", status_code=201)
async def create_managed_key(body: CreateApiKeyBody):
    """Create a new managed API key.

    Response includes the plaintext ``secret`` **exactly once** — the
    Dashboard must capture and display it to the operator immediately. It
    is never persisted, never returned by ``GET``, and never returned by
    ``DELETE``.
    """
    try:
        row = _keys.create_api_key(name=body.name, description=body.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.warning("create_managed_key failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create API key")
    # The response is structurally identical to ``list`` rows + ``plaintext``.
    # We do NOT log the plaintext anywhere.
    return row


@router.delete("/api/api-server/keys/{key_id}", status_code=200)
async def revoke_managed_key(key_id: str):
    """Revoke a managed API key (idempotent). The plaintext is NEVER returned."""
    try:
        ok = _keys.revoke_api_key(key_id)
    except Exception as exc:
        _log.warning("revoke_managed_key(%s) failed: %s", key_id, exc)
        raise HTTPException(status_code=500, detail="Failed to revoke API key")
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown API key id")
    return {"ok": True, "id": key_id}