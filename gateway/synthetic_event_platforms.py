"""Which messaging platform a synthetic process event names, and whether this gateway can resolve it.

``_build_process_event_source`` turns an event's platform name into a ``Platform`` only for a builtin
value or a plugin platform the registry knows. For any other name it answers ``None`` — and a durable
async completion whose name never resolves stays on the drain loop's retry path forever: the event is
re-offered every tick, its ledger row stays ``pending`` (so restart recovery replays it again), and
every attempt logs the same warning. Nothing about the route can change between two attempts, which is
what made the loop unbounded (#152734) even after the retry cadence itself was paced (#141153).

Registration happens at plugin load, including the runtime ``reload-plugins`` verb, so an unrecognised
name is not provably permanent on first sight: it gets a bounded number of delivery attempts, then its
row is settled ``dropped`` — nothing was delivered, so not ``delivered`` — and the event leaves the
loop with one actionable warning instead of a warning per tick.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

# Log-record parity with the origin module (``gateway.run``), as the other notification siblings do.
logger = logging.getLogger("gateway.run")

# Delivery attempts an unregistered platform's completion gets before it is dropped. The drain paces
# retries with a capped backoff, so this is about a minute — long enough for a plugin reloaded
# mid-flight to register the platform, short enough that a stale name does not tick forever.
UNREGISTERED_PLATFORM_ATTEMPTS = 5


def resolve_platform(name: str):
    """The ``Platform`` for *name*, or ``None`` when nothing in this process can resolve it.

    Mirrors what ``_build_process_event_source`` requires of a name: a builtin value, or a plugin
    platform the registry knows (``Platform._missing_`` mints pseudo-members only for those).
    """
    if not name:
        return None
    try:
        from gateway.config import _BUILTIN_PLATFORM_VALUES, Platform
        platform = Platform(name)
    except (ImportError, ValueError):
        # Unknown to the enum and to every bundled plugin manifest: nothing can resolve it here.
        return None
    if platform.value in _BUILTIN_PLATFORM_VALUES:
        return platform
    try:
        from gateway.platform_registry import platform_registry
    except ImportError:
        return None
    return platform if platform_registry.is_registered(platform.value) else None


def event_platform_name(evt: dict[str, Any]) -> str:
    """The platform name an event carries, read exactly as ``_build_process_event_source`` reads it."""
    name = str(evt.get("platform") or "").strip().lower()
    if name:
        return name
    session_key = str(evt.get("session_key") or "").strip()
    if not session_key:
        return ""
    from gateway.run import _parse_session_key
    return str((_parse_session_key(session_key) or {}).get("platform") or "").strip().lower()


def warn_unregistered_platform(runner: Any, platform_name: str) -> None:
    """Say once per name that this gateway cannot resolve the platform; repeats log at DEBUG.

    The per-attempt copy of this line is what filled ``gateway.log`` (#152734): it says nothing that
    the first occurrence did not, and the outcome is now bounded, so the rest is debug noise.
    """
    warned = runner.__dict__.setdefault("_unregistered_platform_warned", set())
    if platform_name in warned:
        logger.debug("Synthetic process event still names the unregistered platform %r", platform_name)
        return
    warned.add(platform_name)
    logger.warning(
        "Synthetic process event names the unregistered platform %r; nothing in this gateway can "
        "route it until a plugin registers that platform. Completions for it are dropped after %d "
        "delivery attempts (their result stays queryable in the delegation records).",
        platform_name, UNREGISTERED_PLATFORM_ATTEMPTS)


def _drop_durable_completion(evt: dict[str, Any]) -> None:
    """Settle the ledger row of a completion that can never be delivered by this gateway.

    ``dropped`` keeps the acknowledgement honest (nothing was delivered) and — unlike ``pending`` —
    keeps restart recovery from replaying the row into the same fail-closed drop. An interim
    per-task notice shares the batch's ``delegation_id`` and must not settle the batch's final
    result, so it is left alone; the final completion settles it.
    """
    if evt.get("type") != "async_delegation" or evt.get("task_failure_notice"):
        return
    delegation_id = str(evt.get("delegation_id") or "").strip()
    if not delegation_id:
        return
    try:
        from tools.async_delegation import claim_completion_delivery, drop_completion_delivery
        claim_id = f"gateway-unroutable:{os.getpid()}:{uuid.uuid4().hex}"
        if claim_completion_delivery(delegation_id, claim_id):
            drop_completion_delivery(delegation_id, claim_id)
    except Exception:
        logger.debug("Could not drop the durable row of unroutable completion %s", delegation_id,
                     exc_info=True)


async def unroutable_completion_exhausted(runner: Any, evt: dict[str, Any]) -> bool:
    """Count one refused attempt at a completion whose platform this gateway cannot resolve.

    ``True`` means the event must leave the drain loop: the attempt cap is spent, the row is settled
    ``dropped`` and the operator was told why. ``False`` keeps the caller's normal requeue, which
    covers every other refusal reason (a restart wait, a parked profile, a disconnected transport, a
    full owner FIFO) — those can still clear, so they must never drop a completion.
    """
    name = event_platform_name(evt)
    if not name or resolve_platform(name) is not None:
        return False
    if await asyncio.to_thread(runner._build_process_event_source, evt) is not None:
        return False
    attempts = int(evt.get("_unroutable_platform_attempts") or 0) + 1
    evt["_unroutable_platform_attempts"] = attempts
    if attempts < UNREGISTERED_PLATFORM_ATTEMPTS:
        return False
    warn_unregistered_platform(runner, name)
    logger.warning(
        "Async completion for the unregistered platform %r was undeliverable in %d attempts; "
        "dropping the queued event", name, attempts)
    # Delivery attempts happen after the group scope exited; settle the row under
    # the session's profile, never under the gateway's launch/default profile.
    async with runner._completion_event_scope(evt):
        await asyncio.to_thread(_drop_durable_completion, evt)
    return True
