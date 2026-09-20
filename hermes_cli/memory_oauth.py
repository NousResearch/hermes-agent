"""HTTP routes for memory-provider OAuth connect, mounted by ``web_server``."""

from __future__ import annotations

import atexit
import importlib
from typing import Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_routers._common import scoped_to_thread
from plugins.memory import contract

router = APIRouter(prefix="/api/memory/providers")
_UNSUPPORTED = {"supported": False, "state": "unsupported", "connected": False, "auth": None, "detail": ""}

atexit.register(contract.shutdown_oauth)


def _resolve_flow(provider: str):
    """Return a provider's OAuth flow module by convention, or raise 404."""
    if not provider.isidentifier():
        raise HTTPException(status_code=404, detail=f"unknown memory provider {provider!r}")
    try:
        return importlib.import_module(f"plugins.memory.{provider}.oauth_flow")
    except ImportError:
        raise HTTPException(status_code=404, detail=f"{provider} does not support OAuth connect")


def _oauth_response(provider: str, *, start: bool, declared: bool) -> dict:
    from hermes_constants import get_hermes_home
    from plugins.memory import find_provider_dir

    external = contract.for_provider(provider)
    try:
        status = external.oauth(get_hermes_home().resolve(), start=start) if external is not None else None
    except contract.CompanionError:
        raise HTTPException(
            status_code=500,
            detail=f"Could not use {provider} OAuth connect. Check the provider installation and retry.",
        ) from None
    if status is None:
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
        status = contract.normalize_status(raw)
    return {**status, "supported": True} if declared else status


@router.post("/{provider}/oauth/start")
async def start_memory_oauth(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Begin a provider's zero-CLI OAuth flow (browser + loopback listener); returns immediately, poll status."""
    return await scoped_to_thread(profile, lambda: _oauth_response(provider, start=True, declared=surface == "declared"))


@router.get("/{provider}/oauth/status")
async def memory_oauth_status(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Poll a provider's OAuth flow: idle | pending | connected | error."""
    return await scoped_to_thread(profile, lambda: _oauth_response(provider, start=False, declared=surface == "declared"))
