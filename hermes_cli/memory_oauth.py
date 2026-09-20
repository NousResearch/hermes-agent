"""HTTP routes for memory-provider OAuth connect, mounted by ``web_server``."""

from __future__ import annotations

import importlib
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter(prefix="/api/memory/providers")

# Clients only ever see these states and this detail text; provider strings never cross.
STATE_DETAIL = {
    "idle": "",
    "pending": "Waiting for browser consent",
    "connected": "Connected",
    "error": "Authorization did not complete",
}
AUTH_KINDS = frozenset({"oauth", "apikey"})
_UNSUPPORTED = {"supported": False, "state": "unsupported", "connected": False, "auth": None, "detail": ""}


def _resolve_flow(provider: str):
    """Return a provider's OAuth flow module by convention, or raise 404."""
    if not provider.isidentifier():
        raise HTTPException(status_code=404, detail=f"unknown memory provider {provider!r}")
    try:
        return importlib.import_module(f"plugins.memory.{provider}.oauth_flow")
    except ImportError:
        raise HTTPException(status_code=404, detail=f"{provider} does not support OAuth connect")


def normalize_status(raw: Any) -> dict:
    """Reduce a hook's dict to state, connected and auth."""
    data = raw if isinstance(raw, dict) else {}
    state = data.get("state") if data.get("state") in STATE_DETAIL else "error"
    status: dict = {"state": state, "detail": STATE_DETAIL[state]}
    if data.get("connected") is True:
        status["connected"] = True
    if "auth" in data:
        status["auth"] = data["auth"] if data["auth"] in AUTH_KINDS else None
    return status


def _oauth_response(provider: str, *, start: bool, declared: bool) -> dict:
    from plugins.memory import find_provider_dir

    try:
        flow = _resolve_flow(provider)
    except HTTPException:
        if declared and find_provider_dir(provider) is not None:
            return dict(_UNSUPPORTED)
        raise
    try:
        # The flow resolves its config path eagerly inside this scope; its worker thread outlives it.
        raw = flow.start_loopback_flow_background() if start else flow.get_flow_status()
    except Exception as exc:
        action = "start" if start else "read"
        raise HTTPException(status_code=500, detail=f"Failed to {action} {provider} OAuth{'' if start else ' status'}: {exc}")
    status = normalize_status(raw)
    return {**status, "supported": True} if declared else status


@router.post("/{provider}/oauth/start")
async def start_memory_oauth(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Begin a provider's zero-CLI OAuth flow (browser + loopback listener); returns immediately, poll status."""
    return await scoped_to_thread(profile, lambda: _oauth_response(provider, start=True, declared=surface == "declared"))


@router.get("/{provider}/oauth/status")
async def memory_oauth_status(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Poll a provider's OAuth flow: idle | pending | connected | error."""
    return await scoped_to_thread(profile, lambda: _oauth_response(provider, start=False, declared=surface == "declared"))
