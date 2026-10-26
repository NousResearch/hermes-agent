"""``/p/<profile>/`` prefix resolution for the shared api_server listener."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The @handle Bot Mode advertises a host's default profile under (apps/desktop hermes-bots
# ``botHandle``). New profiles cannot take the name, but a pre-reservation profiles/hermes can exist.
DEFAULT_PROFILE_HANDLE = "hermes"
# 404 body for a prefix naming no profile here; peers key their @hermes -> /p/default fallback on it.
UNKNOWN_PROFILE_ERROR = "Unknown or unconfigured profile"


def _handle_is_free() -> bool:
    """The handle aliases the default profile only on a host with no profile of that name: a
    pre-reservation ``profiles/hermes`` (served, parked or standalone) keeps it, so its traffic is
    never answered by the default profile. Fail closed: unknown counts as taken."""
    try:
        from hermes_cli.profiles import profile_exists
        return not profile_exists(DEFAULT_PROFILE_HANDLE)
    except Exception:
        logger.warning("Cannot check for a '%s' profile", DEFAULT_PROFILE_HANDLE, exc_info=True)
        return False


def _prefix_profile(request: Any) -> str:
    """The prefix as a canonical profile id. Profile ids are lowercase on disk and every other
    reader (``get_profile_dir``, ``profile_exists``, the peer client) normalizes case, so
    ``/p/Hermes/`` must name the same profile ``/p/hermes/`` does, never fall to the alias."""
    raw = (request.match_info.get("profile") or "").strip()
    if not raw:
        return ""
    from hermes_cli.profiles import normalize_profile_name
    return normalize_profile_name(raw)


def rejection_response(request: Any):
    """The 404 for a rejected prefix. A 'hermes' profile this gateway does not serve gets its own
    body, so a peer reads it as "exists elsewhere", never as "no such profile, use default"."""
    from aiohttp import web
    if _prefix_profile(request) == DEFAULT_PROFILE_HANDLE and not _handle_is_free():
        return web.json_response({"error": {
            "message": f"Profile '{DEFAULT_PROFILE_HANDLE}' exists on this host but this gateway does not serve it",
            "code": "profile_not_served"}}, status=404)
    return web.json_response({"error": UNKNOWN_PROFILE_ERROR}, status=404)


def resolve_request_profile(adapter: Any, request: Any):
    """Resolve + validate the /p/<profile>/ prefix: ``None`` (no prefix, or multiplexing off and the
    prefix names this gateway's own profile), the served profile name, or ``_PROFILE_REJECTED``
    (-> 404). Fail closed: a foreign prefix must never be ignored. The default profile's handle
    ``hermes`` reaches the default profile, because peers address the mirror by the handle their
    roster shows (#118076), unless the host has a profile literally named ``hermes``."""
    from gateway.platforms.api_server import _PROFILE_REJECTED, _prefix_names_served_profile
    profile = _prefix_profile(request)
    if not profile:
        return None
    alias = profile == DEFAULT_PROFILE_HANDLE
    cfg = getattr(adapter.gateway_runner, "config", None)
    if not getattr(cfg, "multiplex_profiles", False):
        served = _prefix_names_served_profile(profile) or (
            alias and _prefix_names_served_profile("default") and _handle_is_free())
        return None if served else _PROFILE_REJECTED
    try:
        from hermes_cli.profiles import profiles_to_serve
        served = {name for name, _ in profiles_to_serve(multiplex=True)}
    except Exception:
        # Fail closed: a prefix this gateway cannot prove it serves is a 404.
        logger.warning("Cannot list served profiles for /p/%s/", profile, exc_info=True)
        return _PROFILE_REJECTED
    if profile in served:
        return profile
    if alias and "default" in served and _handle_is_free():
        return "default"
    return _PROFILE_REJECTED
