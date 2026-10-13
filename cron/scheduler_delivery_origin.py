"""Relay egress discriminators a cron fire carries from its persisted job origin."""

from __future__ import annotations

from typing import Any


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


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Secret-free provenance suffix (origin platform/chat/source-IP fields) for security warnings
    about a bad stored ``context_from`` reference, where no live request object exists."""
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""
    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""
