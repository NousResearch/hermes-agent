"""The profile homes the gateway's in-process cron ticker visits on each tick."""

from __future__ import annotations

from pathlib import Path


def _cron_tick_profile_homes(config: object) -> list[tuple[str, Path]]:
    """Profile homes the in-process ticker visits: the served set PLUS the process-active
    profile: ``profiles_to_serve`` lists default + every live named profile, but a ``--profile
    <name>`` gateway's own profile may sit outside ``profiles/`` (custom HERMES_HOME). One host
    process ticks all of them regardless of ``gateway.multiplex_profiles``. Adapter startup
    already skips ``active``.

    A ``gateway.standalone`` profile's gateway serves only itself, so it ticks only its own
    store. The host's profiles belong to the host gateway: ``_cron_profile_gate`` stands a
    standalone gateway down only while that gateway is alive, so with the host gateway stopped
    every standalone gateway would fire the host's jobs (the stopped host's cron keeps firing),
    deliver them through ``SharedRouteAdapters`` or fail closed instead of the owning profile's
    adapters, and race the other standalone gateways for each store's ``cron/.tick.lock``."""
    from hermes_cli.profiles import get_active_profile_name, get_profile_dir, profile_is_standalone

    active = get_active_profile_name() or "default"  # launch profile, pre-identity (ticker boot)
    if active != "default":
        own_home = get_profile_dir(active)
        if profile_is_standalone(own_home):
            return [(active, own_home)]
    from gateway.run import _multiplex_profile_homes  # late: gateway.run imports this module

    homes = _multiplex_profile_homes(config)
    if any(name == active for name, _home in homes):
        return homes
    try:
        return homes + [(active, get_profile_dir(active))]
    except ValueError:  # get_profile_dir refuses a name that is not a valid profile id
        return homes
