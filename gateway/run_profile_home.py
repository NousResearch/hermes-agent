"""Which profile's HERMES_HOME serves an inbound source (``GatewayRunner._resolve_profile_home_for_source``)."""

import logging
from pathlib import Path

logger = logging.getLogger("gateway.run")


def _served_names(config) -> set:
    from gateway.run import _multiplex_profile_homes
    return {name for name, _home in _multiplex_profile_homes(config)}


def resolve_profile_home_for_source(runner, source) -> Path:
    """The pinned identity's runtime home, else ``source.profile``, then
    ``_profile_name_for_source`` (sources bypassing ``build_source``), then the active profile.

    Under multiplexing an explicitly named profile that is missing, not served (parked) or
    unresolvable raises ``ProfileRouteRejected``: the launch home is another profile, with its own
    config, memory and secrets (#154610). Only the no-profile case and single-profile gateways
    keep the launch-home fallback."""
    from gateway.profile_routing import ProfileRouteRejected
    from gateway.session_identity import identity_of
    from hermes_cli.profiles import (
        get_active_profile_name, get_profile_dir, normalize_profile_name, profile_exists)
    from hermes_constants import get_hermes_home
    identity = identity_of(source)
    if identity is not None:
        return identity.runtime_home
    config = getattr(runner, "config", None)
    multiplexed = bool(getattr(config, "multiplex_profiles", False))
    where = (getattr(source.platform, "value", source.platform), source.chat_id)
    explicit_profile = None  # explicitly requested (source or routing) vs. default fallback
    try:
        name = (source.profile or "").strip() or runner._profile_name_for_source(source)
        explicit_profile = name or None
        if not name:
            name = get_active_profile_name() or "default"
        profile_dir = get_profile_dir(name)
        if explicit_profile and multiplexed and (
                not profile_exists(name) or normalize_profile_name(name) not in _served_names(config)):
            logger.warning("Refusing source %s/%s: profile %r is not served by this gateway",
                           *where, explicit_profile)
            raise ProfileRouteRejected(explicit_profile, explicit_profile)
        if explicit_profile and not profile_exists(name):
            logger.warning(
                "Profile %r does not exist for source %s/%s (guild_id=%s), "
                "falling back to global HERMES_HOME",
                explicit_profile, *where, getattr(source, "guild_id", None))
            return get_hermes_home()
        return profile_dir
    except ProfileRouteRejected:
        raise
    except Exception as exc:
        if multiplexed and explicit_profile:
            logger.warning("Refusing source %s/%s: profile %r could not be resolved",
                           *where, explicit_profile, exc_info=True)
            raise ProfileRouteRejected(explicit_profile, explicit_profile) from exc
        logger.warning(
            "Failed to resolve profile directory for source %s/%s (guild_id=%s), "
            "falling back to global HERMES_HOME: %s",
            *where, getattr(source, "guild_id", None),
            explicit_profile or "(no profile)", exc_info=True)
        return get_hermes_home()


def completion_event_scope_or_refuse(runner, evt: dict, what: str):
    """``runner._completion_event_scope(evt)``, or ``None`` when the event's profile is not served.

    Such an event is refused for good on this gateway (never retried, never run under the launch
    profile). Its durable ledger row, if any, lives in that profile's own store and replays when the
    profile is served again."""
    from gateway.profile_routing import ProfileRouteRejected
    try:
        return runner._completion_event_scope(evt)
    except ProfileRouteRejected as exc:
        logger.warning("Dropping %s for %s: profile %r is not served by this gateway",
                       what, evt.get("session_key") or evt.get("session_id") or "<unknown>", exc.profile)
        return None
