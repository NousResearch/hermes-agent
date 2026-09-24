"""HTTP routes for memory-provider OAuth connect, mounted by ``web_server``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

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


def normalize_status(raw: Any) -> dict:
    """Reduce a hook's dict to state, connected and auth for the Desktop surface."""
    data = raw if isinstance(raw, dict) else {}
    state = data.get("state") if data.get("state") in STATE_DETAIL else "error"
    status: dict = {"state": state, "detail": STATE_DETAIL[state]}
    if data.get("connected") is True:
        status["connected"] = True
    if "auth" in data:
        status["auth"] = data["auth"] if data["auth"] in AUTH_KINDS else None
    return status


def _oauth_response(provider: str, profile: Optional[str], *, start: bool, declared: bool) -> dict:
    from plugins.memory import find_provider_dir

    try:
        # The flow resolves its config path eagerly inside this scope; its worker thread outlives it.
        with _scope_to_profile(profile):
            try:
                flow = _resolve_flow(provider)
            except HTTPException:
                if declared and find_provider_dir(provider) is not None:
                    return dict(_UNSUPPORTED)
                raise
            raw = flow.start_loopback_flow_background() if start else flow.get_flow_status()
    except HTTPException:
        raise
    except Exception as exc:
        action = "start" if start else "read"
        raise HTTPException(status_code=500, detail=f"Failed to {action} {provider} OAuth{'' if start else ' status'}: {exc}")
    # The legacy dashboard reads the hook's dict as-is; only the Desktop surface gets the reduced shape.
    return {**normalize_status(raw), "supported": True} if declared else raw


@router.post("/{provider}/oauth/start")
async def start_memory_oauth(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Begin a provider's zero-CLI OAuth flow (browser + loopback listener); returns immediately, poll status."""
    return _oauth_response(provider, profile, start=True, declared=surface == "declared")


@router.get("/{provider}/oauth/status")
async def memory_oauth_status(provider: str, profile: Optional[str] = None, surface: Optional[str] = None):
    """Poll a provider's OAuth flow: idle | pending | connected | error."""
    return _oauth_response(provider, profile, start=False, declared=surface == "declared")
