"""HTTP routes for memory-provider OAuth connect, mounted by ``web_server``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/memory/providers")


def _resolve_flow(provider: str):
    """Return a provider's ``oauth_flow`` module (bundled or user-dir copy), or raise 404."""
    if not provider.isidentifier():
        raise HTTPException(status_code=404, detail=f"unknown memory provider {provider!r}")
    from plugins.memory import import_provider_module

    try:
        return import_provider_module(provider, "oauth_flow")
    except ImportError:
        raise HTTPException(status_code=404, detail=f"{provider} does not support OAuth connect")


@contextmanager
def _scope_to_profile(profile: Optional[str]):
    """Scope config resolution to ``profile`` so the flow's eager path resolve targets that profile's
    honcho.json. None/""/"current" leaves it untouched."""
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
    """Begin a provider's zero-CLI OAuth flow (browser + loopback listener); returns immediately, poll status."""
    try:
        # The flow resolves its config path eagerly inside this scope; its worker thread outlives it.
        with _scope_to_profile(profile):
            flow = _resolve_flow(provider)
            return flow.start_loopback_flow_background()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start {provider} OAuth: {exc}")


@router.get("/{provider}/oauth/status")
async def memory_oauth_status(provider: str, profile: Optional[str] = None):
    """Poll a provider's OAuth flow: idle | pending | connected | error."""
    try:
        with _scope_to_profile(profile):
            flow = _resolve_flow(provider)
            return flow.get_flow_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {provider} OAuth status: {exc}")
