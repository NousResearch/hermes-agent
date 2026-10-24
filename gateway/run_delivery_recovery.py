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
        for platform in Platform:
            if _transport(runner, platform, profile) is not None:
                targets.add((platform.value, profile))
    if not (getattr(runner, "_profile_adapters", {}) or {}):
        targets.update((platform, None) for platform, _ in targets.copy())
    return targets


def relay_platforms(runner, profile=None):
    relay = runner._adapters_for_profile(profile).get(Platform.RELAY)
    fronts = getattr(relay, "fronts_platform", None)
    return [platform for platform in Platform if callable(fronts) and fronts(platform)]


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
        await runner._release_runtime_claim_quiet(
            row["obligation_id"],
            "failed to release undispatched runtime obligation %s",
            error=error if is_flood_error(error) else "send_path_degraded",
        )
    return transport


async def redeliver_claimed(runner, claimed):
    """Send already-claimed replies, preserving the logical platform and runtime route."""
    from gateway.delivery_ledger import RECOVERED_MARKER, mark_delivered, mark_failed
    from gateway.run_shutdown import _log_suppressed

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
            try:
                result = await transport.send(
                    Platform(row["platform"]), row["chat_id"], content, metadata or None
                )
            except Exception as error:
                logger.warning(
                    "obligation %s: redelivery send raised: %s",
                    row["obligation_id"],
                    error,
                    exc_info=True,
                )
                result = None
            with _log_suppressed(
                logging.DEBUG, "delivery ledger update failed", exc_info=True
            ):
                if result is not None and getattr(result, "success", False):
                    await asyncio.to_thread(mark_delivered, row["obligation_id"])
                    redelivered += 1
                    logger.info(
                        "Redelivered recovered final response to %s:%s (obligation %s, attempt %d)",
                        row["platform"],
                        row["chat_id"],
                        row["obligation_id"],
                        row["attempts"],
                    )
                else:
                    await asyncio.to_thread(
                        mark_failed,
                        row["obligation_id"],
                        str(getattr(result, "error", "") or "send failed"),
                    )
        with _log_suppressed(
            logging.DEBUG, "arming flood redelivery timers failed", exc_info=True
        ):
            await runner._arm_flood_timers_for_waiting_rows()
    return redelivered


def schedule_relay_recovery(adapter):
    """A transport reconnect does not recreate its RelayAdapter in the runner."""
    runner = adapter.gateway_runner
    if runner is None or not getattr(runner, "_running", False):
        return
    task = runner._retain_background_task(
        asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(
                Platform.RELAY, profile=getattr(adapter, "_owner_profile", None)
            ),
            name="relay-delivery-recovery",
        )
    )
    task.add_done_callback(
        runner._late_failure_callback("relay delivery recovery failed")
    )
