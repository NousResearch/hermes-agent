"""Which profile's HERMES_HOME serves an inbound source (``GatewayRunner._resolve_profile_home_for_source``)."""

import contextlib
import logging
from pathlib import Path

from agent.runtime_scope import scope_for_profile
from agent.secret_scope import _is_process_home, is_multiplex_active
from hermes_constants import get_hermes_home, hermes_home_key
from tui_gateway.launch_profile_policy import launch_profile_runtime_scope

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


async def completion_event_scope_or_refuse(runner, evt: dict, what: str):
    """``runner._completion_event_scope(evt)`` already entered (``async with`` the result exits it),
    or ``None`` when the event's profile is not served.

    Such an event is refused for good on this gateway (never retried, never run under the launch
    profile). Its durable ledger row, if any, lives in that profile's own store and replays when the
    profile is served again. Entering here also covers scopes that resolve the profile on entry
    (a raw api_server event's ``_served_api_server_event_scope``)."""
    from contextlib import AsyncExitStack
    from gateway.profile_routing import ProfileRouteRejected
    stack = AsyncExitStack()
    try:
        await stack.enter_async_context(runner._completion_event_scope(evt))
    except ProfileRouteRejected as exc:
        logger.warning("Dropping %s for %s: profile %r is not served by this gateway",
                       what, evt.get("session_key") or evt.get("session_id") or "<unknown>", exc.profile)
        return None
    return stack


def completion_ledger_home(runner, evt: dict) -> Path | None:
    """Validate private queue provenance against the gateway's served profiles.

    The producing ledger stamps this after decoding a stored event. A deleted
    raw local parent cannot rediscover that owner from its session row. Unknown
    owners stay retryable rather than falling through to the launch profile.
    """
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    owner = evt.get("_completion_ledger_home")
    if owner is None:
        return None
    if not isinstance(owner, str) or not owner or not Path(owner).is_absolute():
        raise LookupError("Invalid completion ledger owner")
    homes = dict(getattr(runner, "_served_profile_homes", None) or {})
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    routing_home = getattr(getattr(runner, "session_store", None), "_routing_home", None)
    homes.setdefault(primary, routing_home or get_routing_process_hermes_home())
    key = hermes_home_key(owner)
    for home in homes.values():
        if hermes_home_key(home) == key:
            return Path(home)
    raise LookupError("Completion ledger owner is not served by this gateway")


@contextlib.asynccontextmanager
async def completion_ledger_scope(home: Path):
    """Bind the validated home while preserving the launch profile's environment policy."""
    scope_for_profile(home)  # Refuse hosted callers even when the launch binding below is a no-op.
    if _is_process_home(home):
        # Preserve a standalone launch body's ambient/prepared policy; a foreign
        # caller must still re-enter the launch home rather than borrow its scope.
        scope = (contextlib.nullcontext()
                 if not is_multiplex_active() and hermes_home_key(get_hermes_home()) == hermes_home_key(home)
                 else launch_profile_runtime_scope(home))
        with scope:
            yield
    else:
        from gateway.run import _async_profile_runtime_scope
        async with _async_profile_runtime_scope(home):
            yield


def completion_event_scope(runner, evt: dict):
    """Bind the ledger owner before source/session heuristics for legacy events."""
    from gateway.run import _async_profile_runtime_scope
    from hermes_constants import get_hermes_home_override
    from gateway.run_notifications import _raw_process_event_session_id
    owner = completion_ledger_home(runner, evt)
    if owner is not None:
        return completion_ledger_scope(owner)
    source = runner._build_process_event_source(evt)
    if source is None or not getattr(source, "profile", None):
        raw_sid = _raw_process_event_session_id(evt)
        if raw_sid and getattr(getattr(runner, "config", None), "multiplex_profiles", False):
            return runner._served_api_server_event_scope(evt, raw_sid)
        # No routed profile: the launch profile's own completion. Bind ITS scope once the
        # process multiplexes — unscoped, a fail-closed ledger read raises on a legitimate
        # launch-profile event (no-op while single-profile).
        from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed
        return async_launch_profile_scope_if_multiplexed()
    profile_home = runner._resolve_profile_home_for_source(source)
    if get_hermes_home_override() == str(profile_home):
        return contextlib.nullcontext()  # already inside this profile's scope
    return _async_profile_runtime_scope(profile_home)
