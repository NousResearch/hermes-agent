"""Restore ledger replies through the same scoped transports as ordinary sends."""

import asyncio
import logging

from gateway.config import Platform
from gateway.delivery import resolve_delivery_transport
from gateway.delivery_work import delivery_work
from hermes_state_keys import profile_from_session_key

logger = logging.getLogger("gateway.run")


def _transport(runner, platform, profile):
    adapters = runner._adapters_for_profile(profile)
    configs = getattr(runner, "_profile_configs", {}) or {}
    config = (
        runner.config
        if adapters is runner.adapters
        else configs.get(profile, runner.config)
    )
    return resolve_delivery_transport(platform, config, adapters)


def deliverable_targets(runner):
    """Advertised logical platforms, bound to the exact credential-owning profile."""
    targets = set()
    profiles = {"default", *(getattr(runner, "_profile_adapters", {}) or {})}
    primary = getattr(runner, "_primary_profile_name", None)
    if primary:
        profiles.add(primary)
    for profile in profiles:
        adapters = runner._adapters_for_profile(profile)
        for platform in set(adapters) | set(relay_platforms(runner, profile)):
            if platform == Platform.RELAY:
                continue
            if _transport(runner, platform, profile) is not None:
                targets.add((platform.value, profile))
    if not (getattr(runner, "_profile_adapters", {}) or {}):
        targets.update((platform, None) for platform, _ in targets.copy())
    return targets


def relay_platforms(runner, profile=None):
    relay = runner._adapters_for_profile(profile).get(Platform.RELAY)
    advertised = getattr(relay, "fronted_platforms", None)
    platforms = []
    for name in advertised() if callable(advertised) else ():
        try:
            platform = Platform(name)
        except ValueError:
            logger.warning(
                "Ignoring unregistered relay platform %r during recovery", name
            )
            continue
        if platform != Platform.RELAY:
            platforms.append(platform)
    return platforms


def relay_recovery_pending(runner):
    """Keep reconnect-only replies awake without adding them to restart drain work."""
    maps = [runner.adapters, *(getattr(runner, "_profile_adapters", {}) or {}).values()]
    if not any(Platform.RELAY in adapters for adapters in maps):
        return False
    from gateway.delivery_ledger import ledger_enabled, pending_reconnects

    if not ledger_enabled():
        return False
    for platform, profile in pending_reconnects():
        transport = _transport(runner, Platform(platform), profile)
        if transport is not None and transport.is_relay:
            return True
    return False


async def claim_reconnected_replies(runner, platform, profile, targets):
    """Adopt offline-at-boot replies before claiming this process's failed sends."""
    from gateway.delivery_ledger import (
        sweep_failed_for_runtime,
        sweep_recoverable,
    )

    owner = profile or "default"
    eligible = {
        (name, bot)
        for name, bot in targets
        if name == platform.value
        and (bot == owner or (bot is None and owner == "default"))
    }
    inherited = await asyncio.to_thread(sweep_recoverable, deliverable_targets=eligible)
    try:
        return inherited + await asyncio.to_thread(
            sweep_failed_for_runtime, platform.value, profile=profile
        )
    except Exception:
        # A later read failing must not strand the earlier claims as attempting.
        await refund_unsent_claims(runner, inherited)
        raise


async def refund_unsent_claims(runner, rows=()):
    """Return undispatched claims without losing their pre-claim refusal or send budget."""
    from gateway.run_delivery_settlement import settle_claim

    for row in rows:
        if not row.get("adopted"):
            await settle_claim(
                runner, row, "unsent", row.get("last_error") or "send_path_degraded"
            )


async def prepare_recovery_claims(runner, claimed):
    """Clear every answered turn's resume flag before any replay can be dispatched."""
    sendable = await runner._clear_resume_pending_for_claimed_obligations(
        claimed, require_success=True
    )
    ids = {row["obligation_id"] for row in sendable}
    unsent = [row for row in claimed if row["obligation_id"] not in ids]
    await refund_unsent_claims(runner, unsent)
    return sendable, not unsent


async def redeliver_after_reconnect(runner, platform, profile=None):
    """Sent count, or None when recovery is incomplete and auto-resume must be deferred."""
    with delivery_work(runner):
        if platform == Platform.RELAY:
            delivered = 0
            complete = True
            for logical in relay_platforms(runner, profile):
                count = await runner._redeliver_failed_obligations_for_platform(
                    logical, profile=profile
                )
                if count is None:
                    complete = False
                else:
                    delivered += count
            return delivered if complete else None
        try:
            from gateway.delivery_ledger import ledger_enabled
            from gateway.run_delivery_settlement import flush_settlements

            if not await asyncio.to_thread(ledger_enabled):
                return 0
            if not await flush_settlements(runner, platform, profile):
                return None
            targets = deliverable_targets(runner)
            claimed = await claim_reconnected_replies(
                runner, platform, profile, targets
            )
        except Exception:
            logger.warning(
                "delivery recovery failed after %s reconnect",
                platform.value,
                exc_info=True,
            )
            return None
        if not claimed:
            await runner._arm_flood_timers_for_waiting_rows()
            return 0
        sendable, complete = await prepare_recovery_claims(runner, claimed)
        delivered = await runner._redeliver_claimed_obligations(sendable)
        if not complete:
            logger.warning(
                "Reply recovery could not clear resume flags; deferring auto-resume"
            )
            return None
        return delivered


async def obligation_transport(runner, row):
    try:
        platform = Platform(row["platform"])
    except ValueError:
        logger.warning(
            "obligation %s: unknown platform %r",
            row["obligation_id"],
            row.get("platform"),
        )
        return None
    transport = _transport(runner, platform, row.get("profile"))
    if transport is None:
        from gateway.delivery_ledger import is_flood_error

        error = row.get("last_error")
        await refund_unsent_claims(
            runner,
            [
                {
                    **row,
                    "last_error": error
                    if is_flood_error(error)
                    else "send_path_degraded",
                }
            ],
        )
        wake_ready_relay(runner, platform, row.get("profile"))
    return transport


async def redeliver_claimed(runner, claimed):
    """Send already-claimed replies, preserving the logical platform and runtime route."""
    from gateway.delivery_ledger import RECOVERED_MARKER
    from gateway.run_delivery_settlement import settle_replay_result

    redelivered = 0
    with delivery_work(runner):
        for row in claimed:
            if row.get("adopted"):
                continue
            transport = await obligation_transport(runner, row)
            if transport is None:
                continue
            content = row["content"]
            if row.get("needs_marker"):
                content = row.get("marker", RECOVERED_MARKER) + content
            metadata = {"thread_id": row["thread_id"]} if row.get("thread_id") else {}
            if transport.is_relay:
                metadata["profile"] = (
                    profile_from_session_key(row["session_key"]) or "default"
                )
                # This is an older saved answer, not the final of a live turn
                # that may now be streaming into this same chat/thread.
                metadata["_interim_send"] = True
            try:
                if transport.is_relay:
                    result = await transport.send(
                        Platform(row["platform"]),
                        row["chat_id"],
                        content,
                        metadata or None,
                    )
                else:
                    result = await transport.adapter.send(
                        chat_id=row["chat_id"],
                        content=content,
                        metadata=metadata or None,
                    )
            except Exception as error:
                logger.warning(
                    "obligation %s: redelivery send raised: %s",
                    row["obligation_id"],
                    error,
                    exc_info=True,
                )
                result = None
            count = await settle_replay_result(runner, row, transport, result)
            redelivered += count
            if count:
                logger.info(
                    "Redelivered recovered final response to %s:%s (obligation %s, attempt %d)",
                    row["platform"],
                    row["chat_id"],
                    row["obligation_id"],
                    row["attempts"],
                )
        try:
            await runner._arm_flood_timers_for_waiting_rows()
        except Exception:
            logger.warning("Arming reply redelivery timers failed", exc_info=True)
            from gateway.run_reconnect_recovery import schedule_served_recovery

            schedule_served_recovery(runner)
    return redelivered


def schedule_relay_recovery(adapter):
    """A transport reconnect does not recreate its RelayAdapter in the runner."""
    runner = adapter.gateway_runner
    if runner is None or not getattr(runner, "_running", False):
        return
    from gateway.run_reconnect_recovery import schedule_recovery

    schedule_recovery(
        runner, Platform.RELAY, profile=getattr(adapter, "_owner_profile", None)
    )


def wake_ready_relay(runner, platform, profile):
    """The reconnect sweep may have run before an async claim refund committed."""
    transport = _transport(runner, platform, profile)
    if (
        transport is not None
        and transport.is_relay
        and getattr(transport.adapter._transport, "delivery_ready", False) is True
    ):
        schedule_relay_recovery(transport.adapter)


def wake_after_relay_failure(runner, platform, profile, error):
    from gateway.delivery_ledger import RELAY_DISCONNECTED_ERRORS

    if str(error or "").strip().lower() in RELAY_DISCONNECTED_ERRORS:
        wake_ready_relay(runner, platform, profile)


async def recover_after_reconnect_failure(producer, event, delivery_adapter, error):
    """Preserve native replacement recovery and catch same-relay late failure writes."""
    runner = producer.gateway_runner
    redeliver = getattr(runner, "_redeliver_failed_obligations_for_platform", None)
    live = producer._final_delivery_adapter(event.source)
    profile = getattr(delivery_adapter, "_owner_profile", None)
    if live is not delivery_adapter and callable(redeliver):
        await redeliver(event.source.platform, profile=profile)
    elif runner is not None:
        wake_after_relay_failure(runner, event.source.platform, profile, error)
