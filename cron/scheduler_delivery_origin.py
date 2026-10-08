"""Relay egress discriminators a cron fire carries from its persisted job origin."""

from __future__ import annotations

from typing import Any, Optional


# Gateway platforms whose adapter declares ``supports_async_delivery = False`` (request/response
# only, ``send()`` is a stub) — a cron report can never reach them, so they are never a
# deliver=origin destination.
_NON_PUSH_ORIGIN_PLATFORMS = frozenset({"api_server"})


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job. Non-dict origins (provenance strings, hand-edited
    jobs.json) are treated as missing — otherwise every fire crashed on ``origin.get``.

    Without this guard, a job tagged with e.g. ``"combined-digest-replaces-x-and-y"`` crashed every fire
    attempt with ``'str' object has no attribute 'get'`` — ``mark_job_run`` recorded the failure, but the
    next tick re-loaded the same poisoned origin and crashed identically until the field was patched
    manually (#18722).
    """
    origin = job.get("origin")
    if isinstance(origin, dict) and origin.get("platform") and origin.get("chat_id"):
        # Jobs stamped before non-push origins stopped being captured (#69304): the api_server
        # adapter's send() is a stub, so honouring this origin fails every fire with
        # last_status=ok. Treat it as missing so deliver=origin takes the home-channel fallback.
        if str(origin["platform"]).lower() in _NON_PUSH_ORIGIN_PLATFORMS:
            return None
        return origin
    return None


def stamp_origin_discriminators(t: Any, route_metadata: dict, media_metadata: dict) -> None:
    """Stamp the origin's ``scope_id`` / ``user_id`` onto a live send's metadata.

    Relay egress is fail-closed on a discriminator and the RelayAdapter's caches are cold after every
    boot, so the persisted origin supplies them. Origin targets only (a fan-out target's recipient is not
    the origin's author); ``setdefault`` never overrides router or home stamping; ``user_id`` is read by
    relay transports only.
    """
    discriminators = (
        ("scope_id", t.origin.get("scope_id") if t.origin_target else None),
        ("user_id", t.origin_user_id if t.is_relay else None),
    )
    for key, value in discriminators:
        if value:
            route_metadata.setdefault(key, str(value))
            media_metadata.setdefault(key, str(value))
