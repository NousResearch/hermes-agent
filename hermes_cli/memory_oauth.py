"""HTTP routes for memory-provider OAuth connect, mounted by ``web_server``.

Kept out of ``web_server.py`` so the memory feature's surface stays in the
memory layer. Dispatch is by convention: a provider's flow lives at
``plugins.memory.<provider>.oauth_flow`` exposing ``start_loopback_flow_background``
and ``get_flow_status``; a provider without that module simply 404s. No provider
is named here.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/memory/providers")


def _resolve_flow(provider: str):
    """Return a provider's OAuth flow module by convention, or raise 404."""
    if not provider.isidentifier():
        raise HTTPException(status_code=404, detail=f"unknown memory provider {provider!r}")
    try:
        return importlib.import_module(f"plugins.memory.{provider}.oauth_flow")
    except ImportError:
        raise HTTPException(status_code=404, detail=f"{provider} does not support OAuth connect")


@contextmanager
def _scope_to_profile(profile: Optional[str]):
    """Scope config resolution to ``profile`` so the flow's eager path resolve
    targets that profile's honcho.json. None/""/"current" leaves it untouched."""
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield
        return

    from hermes_cli import profiles as profiles_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    try:
        profiles_mod.validate_profile_name(requested)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(requested):
        raise HTTPException(status_code=404, detail=f"Profile '{requested}' does not exist.")

    token = set_hermes_home_override(str(profiles_mod.get_profile_dir(requested)))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@router.post("/{provider}/oauth/start")
async def start_memory_oauth(provider: str, profile: Optional[str] = None):
    """Begin a provider's zero-CLI OAuth flow — opens the browser and captures
    the grant via the loopback listener. Returns immediately; poll status."""
    flow = _resolve_flow(provider)
    try:
        # The flow resolves its config path eagerly inside this scope; the worker
        # thread it spawns outlives the request and the override.
        with _scope_to_profile(profile):
            return flow.start_loopback_flow_background()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start {provider} OAuth: {exc}")


@router.get("/{provider}/oauth/status")
async def memory_oauth_status(provider: str, profile: Optional[str] = None):
    """Poll a provider's OAuth flow: idle | pending | connected | error."""
    flow = _resolve_flow(provider)
    try:
        with _scope_to_profile(profile):
            return flow.get_flow_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {provider} OAuth status: {exc}")
