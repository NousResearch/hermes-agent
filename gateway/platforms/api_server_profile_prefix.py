"""``/p/<profile>/`` prefix resolution for the shared api_server listener."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The @handle Bot Mode advertises a host's default profile under (apps/desktop hermes-bots
# ``botHandle``); it is a reserved profile name, so it never names a profile of its own.
DEFAULT_PROFILE_HANDLE = "hermes"


def resolve_request_profile(adapter: Any, request: Any):
    """Resolve + validate the /p/<profile>/ prefix: ``None`` (no prefix, or multiplexing off and the
    prefix names this gateway's own profile), the served profile name, or ``_PROFILE_REJECTED``
    (-> 404). Fail closed: a foreign prefix must never be ignored. The default profile's handle
    ``hermes`` reaches the default profile, because peers address the mirror by the handle their
    roster shows (#118076); a served profile literally named ``hermes`` still wins."""
    from gateway.platforms.api_server import _PROFILE_REJECTED, _prefix_names_served_profile
    profile = (request.match_info.get("profile") or "").strip()
    if not profile:
        return None
    alias = profile == DEFAULT_PROFILE_HANDLE
    cfg = getattr(adapter.gateway_runner, "config", None)
    if not getattr(cfg, "multiplex_profiles", False):
        served = _prefix_names_served_profile(profile) or (alias and _prefix_names_served_profile("default"))
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
    if alias and "default" in served:
        return "default"
    return _PROFILE_REJECTED
