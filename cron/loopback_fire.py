"""Does the active cron provider fire over the gateway's loopback api_server?

External providers that declare ``CronScheduler.fires_over_loopback`` (Chronos) deliver every
fire as an HTTP callback the dashboard forwards to ``http://127.0.0.1:<port>/api/cron/fire`` on
the gateway. Without that listener every fire 503s forever, so the api_server stops being an
optional messaging platform and becomes required infrastructure for the profile. The gateway
(``gateway/cron_loopback_listener.py``) and the dashboard Channels toggle both ask here.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def provider_fires_over_loopback(provider: Any) -> bool:
    """True only for a provider that explicitly declares loopback fires (never the ticker)."""
    from cron.scheduler_provider import InProcessCronScheduler

    if provider is None or isinstance(provider, InProcessCronScheduler):
        return False
    return getattr(provider, "fires_over_loopback", False) is True


def served_cron_home_count() -> int:
    """How many profile stores one host gateway ticks (``profiles_to_serve(multiplex=True)``).

    External providers serve exactly one home; with more, the gateway falls back to the
    built-in ticker (``scheduler_for_profile_mode``) and no fire arrives over loopback.
    """
    try:
        from hermes_cli.profiles import profiles_to_serve

        return len(profiles_to_serve(multiplex=True)) or 1
    except Exception:
        logger.debug("cron loopback: could not enumerate served profiles; assuming one", exc_info=True)
        return 1


def loopback_api_server_required(home_count: Optional[int] = None) -> bool:
    """True when the cron provider this profile's gateway will run needs the loopback api_server.

    Reads ``cron.provider`` for the CURRENT home (``get_hermes_home()``): callers bind the
    profile scope first. ``home_count`` is the number of homes the gateway ticks; pass the
    gateway's own count when known, otherwise it is enumerated.
    """
    count = served_cron_home_count() if home_count is None else home_count
    if count > 1:
        return False
    try:
        from cron.scheduler_provider import resolve_cron_scheduler

        return provider_fires_over_loopback(resolve_cron_scheduler())
    except Exception:
        logger.debug("cron loopback: provider resolution failed; not requiring api_server", exc_info=True)
        return False
