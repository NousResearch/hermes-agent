"""Coalesced recovery and resume continuation after a transport becomes available."""

import asyncio
import logging

from gateway.config import Platform
from gateway.run_delivery_recovery import (
    _transport,
    deliverable_targets,
    relay_platforms,
)

logger = logging.getLogger("gateway.run")

# Local store faults need a continuation even when the adapter remains healthy.
# The cap bounds a persistent fault without spending the reply's send budget.
RECOVERY_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 16.0)


def recovery_allowed(runner):
    return (
        getattr(runner, "_running", False)
        and not getattr(runner, "_draining", False)
        and not getattr(runner, "_restart_requested", False)
    )


def resume_candidates(runner, entries, platform, profile, protected):
    candidates = [
        entry
        for entry in entries
        if entry.resume_pending
        and not entry.suspended
        and entry.origin is not None
        and entry.resume_reason in runner._AUTO_RESUME_REASONS
        and (platform is None or entry.origin.platform == platform)
    ]
    for entry in candidates:
        if entry.session_key in protected:
            logger.debug(
                "Deferring auto-resume for %s: a saved reply still owns delivery",
                entry.session_key,
            )
    candidates = [entry for entry in candidates if entry.session_key not in protected]
    if profile is None:
        return candidates
    return [
        entry
        for entry in candidates
        if (transport := _transport(runner, entry.origin.platform, profile)) is not None
        and runner._delivery_adapter_for(runner._restored_source(entry))
        is transport.adapter
    ]


async def _recover_once(runner, platform, profile):
    if not recovery_allowed(runner):
        return True
    try:
        recovered = await runner._redeliver_failed_obligations_for_platform(
            platform, profile=profile
        )
        if recovered is None:
            return False
        if not recovery_allowed(runner):
            return True
        logical = (
            relay_platforms(runner, profile)
            if platform == Platform.RELAY
            else [platform]
        )
        for target in logical:
            # The persisted receiving bot, not the runtime profile namespace, owns resume.
            if (
                runner._schedule_resume_pending_sessions(
                    platform=target, profile=profile or "default"
                )
                is None
            ):
                return False
        return True
    except Exception:
        logger.warning(
            "Reply recovery/resume failed for %s:%s",
            platform.value,
            profile or "default",
            exc_info=True,
        )
        return False


async def recover_reconnected_sessions(runner, platform, profile=None):
    """Try once inline; retain a bounded continuation if storage is temporarily unavailable."""
    if not await _recover_once(runner, platform, profile):
        schedule_recovery(runner, platform, profile=profile, delayed=True)


def schedule_served_recovery(runner):
    """A boot storage fault must not strand replies or unattempted resume work."""
    for name, profile in deliverable_targets(runner):
        if profile is not None:
            schedule_recovery(runner, Platform(name), profile=profile, delayed=True)


def schedule_recovery(runner, platform, *, profile=None, delayed=False):
    """One retained worker per bot; reconnects coalesce without resetting its failure budget."""
    if not recovery_allowed(runner):
        return
    transport = _transport(runner, platform, profile)
    if transport is not None and transport.is_relay:
        platform = Platform.RELAY
    key = (platform, profile or "default")
    pending = getattr(runner, "_reconnect_recovery_tasks", None)
    if pending is None:
        pending = runner._reconnect_recovery_tasks = {}
    existing = pending.get(key)
    if existing is not None:
        existing[1].set()
        return
    wake = asyncio.Event()

    async def run():
        failures = 0
        wait_first = delayed
        try:
            while recovery_allowed(runner):
                if wait_first:
                    if failures >= len(RECOVERY_RETRY_DELAYS):
                        logger.error(
                            "Reply recovery/resume for %s:%s remains incomplete after %d retries; "
                            "saved replies are retained. Check session-store/delivery-ledger errors; "
                            "reconnect or restart retries recovery.",
                            platform.value,
                            key[1],
                            failures,
                        )
                        return
                    await asyncio.sleep(RECOVERY_RETRY_DELAYS[failures])
                    failures += 1
                wake.clear()
                if not recovery_allowed(runner):
                    return
                complete = await _recover_once(runner, platform, profile)
                if complete and not wake.is_set():
                    return
                wait_first = not complete
        finally:
            pending.pop(key, None)

    task = runner._retain_background_task(
        asyncio.create_task(run(), name=f"reconnect-recovery:{platform.value}:{key[1]}")
    )
    pending[key] = (task, wake)
    task.add_done_callback(
        runner._late_failure_callback("reconnect recovery worker failed")
    )
