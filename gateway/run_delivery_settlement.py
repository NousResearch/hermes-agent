"""Finish known replay outcomes without redispatching after a transient ledger write failure."""

import asyncio
import logging

from gateway.config import Platform

logger = logging.getLogger("gateway.run")


async def _apply_settlement(runner, pending, oid):
    from gateway.delivery_ledger import (
        mark_delivered,
        mark_failed,
        release_runtime_claim,
    )

    # Take it before yielding. Two flushes must not refund the same claim after
    # the first one has already made it available for a new send.
    settlement = pending.pop(oid, None)
    if settlement is None:
        return True
    row, state, error = settlement
    try:
        if state == "unsent":
            await asyncio.to_thread(release_runtime_claim, oid, error)
        elif state == "delivered":
            await asyncio.to_thread(mark_delivered, oid)
        else:
            await asyncio.to_thread(mark_failed, oid, error)
    except Exception:
        pending.setdefault(oid, settlement)
        logger.warning(
            "Delivery claim %s could not be settled as %s", oid, state, exc_info=True
        )
        from gateway.run_reconnect_recovery import schedule_recovery

        schedule_recovery(
            runner, Platform(row["platform"]), profile=row.get("profile"), delayed=True
        )
        return False
    return True


async def settle_claim(runner, row, state, error=""):
    pending = runner.__dict__.setdefault("_delivery_settlements", {})
    oid = row["obligation_id"]
    pending[oid] = (row, state, error)
    return await _apply_settlement(runner, pending, oid)


async def flush_settlements(runner, platform, profile):
    """Retry only DB updates, before the next claim pass for this credential owner."""
    pending = runner.__dict__.get("_delivery_settlements", {})
    complete = True
    for oid, (row, _, _) in list(pending.items()):
        if (row["platform"], row.get("profile") or "default") == (
            platform.value,
            profile or "default",
        ):
            if not await _apply_settlement(runner, pending, oid):
                complete = False
    return complete


async def settle_replay_result(runner, row, transport, result):
    """A pre-write local refusal refunds; partial sends and lost ACKs retain the attempt."""
    from gateway.run_delivery_recovery import wake_after_relay_failure

    if result is not None and getattr(result, "success", False):
        await settle_claim(runner, row, "delivered")
        return 1
    error = str(getattr(result, "error", "") or "send failed")
    raw = getattr(result, "raw_response", None)
    # Relay sends the full replay as one frame. The transport alone sets not_sent
    # before writing; remote errors, chunks and ambiguous ACKs cannot assert it.
    unsent = (
        transport.is_relay
        and isinstance(raw, dict)
        and raw.get("not_sent") is True
        and not raw.get("ambiguous")
        and not getattr(result, "message_id", None)
    )
    await settle_claim(runner, row, "unsent" if unsent else "failed", error)
    wake_after_relay_failure(
        runner, Platform(row["platform"]), row.get("profile"), error
    )
    return 0


async def finalize_delivery(producer, obligation_id, result, event, delivery_adapter):
    """Persist the original send's result and wake the appropriate recovery path."""
    try:
        from gateway.dead_targets import classify_dead_error
        from gateway.delivery_ledger import (
            is_reconnect_only,
            mark_delivered,
            mark_failed,
        )

        success = bool(getattr(result, "success", False))
        error = str(getattr(result, "error", "") or "")
        if producer.gateway_runner is not None:
            await settle_claim(
                producer.gateway_runner,
                {
                    "obligation_id": obligation_id,
                    "platform": event.source.platform.value,
                    "profile": getattr(delivery_adapter, "_owner_profile", None)
                    or "default",
                },
                "delivered" if success else "failed",
                error,
            )
        elif success:
            await asyncio.to_thread(mark_delivered, obligation_id)
        else:
            await asyncio.to_thread(mark_failed, obligation_id, error)
        if success:
            return
        if is_reconnect_only(error):
            from gateway.run_delivery_recovery import (
                recover_after_reconnect_failure,
            )

            await recover_after_reconnect_failure(
                producer, event, delivery_adapter, error
            )
        elif (
            classify_dead_error(error) is None
        ):  # a dead chat is never retried: no timer to wake
            schedule = getattr(
                producer.gateway_runner, "_schedule_flood_redelivery", None
            )
            if callable(schedule):
                schedule(
                    event.source.platform,
                    profile=getattr(delivery_adapter, "_owner_profile", None),
                )
    except Exception:
        logger.debug("delivery ledger update failed", exc_info=True)
