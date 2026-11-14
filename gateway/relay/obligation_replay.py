"""Resolve transports for delivery obligations and wake relay replay on handshake."""

import asyncio
import logging

from gateway.config import Platform

logger = logging.getLogger("gateway.run")


def relay_obligation_adapter(runner, platform, profile=None):
    # The ledger names the receiving adapter's profile, which can differ from
    # the session's runtime profile. Relay is shared by the primary gateway.
    if profile not in (None, "", "default") or platform in runner.adapters:
        return None
    relay = runner.adapters.get(Platform.RELAY)
    if relay is None:
        return None
    native = getattr(getattr(runner, "config", None), "platforms", {}).get(platform)
    if native is not None and native.enabled:
        return None  # a configured native bot can be offline, but still owns its rows
    fronts = getattr(relay, "fronts_platform", None)
    return relay if callable(fronts) and fronts(platform) else None


def relay_platform_ready(adapter, platform):
    ready = getattr(getattr(adapter, "_transport", None), "is_platform_ready", None)
    return not callable(ready) or ready(platform.value)


async def resolve_obligation_adapter(runner, row: dict, *, relay_adapter=None):
    """Resolve the adapter for a claimed ledger row, or None when it cannot be delivered now."""
    try:
        platform = Platform(row["platform"])
    except (KeyError, ValueError):
        logger.debug(
            "obligation %s: unknown platform %r",
            row["obligation_id"],
            row.get("platform"),
        )
        return None
    if "profile" in row:
        adapter = runner._authorization_adapter(platform, row.get("profile"))
    else:
        # Startup rows preserve the historical default-adapter route.
        adapter = runner.adapters.get(platform)
    if adapter is None:
        adapter = relay_obligation_adapter(runner, platform, row.get("profile"))
    if relay_adapter is not None and adapter is not relay_adapter:
        adapter = None
    relay_unready = (
        getattr(adapter, "platform", None) is Platform.RELAY
        and platform is not Platform.RELAY
        and not relay_platform_ready(adapter, platform)
    )
    # A claim (runtime or boot) whose adapter vanished before dispatch was never sent: release it
    # without spending an attempt so the reconnect sweep, which only takes 'failed' rows, delivers it
    # instead of it waiting 'attempting' for the next restart.
    # Only a flood row keeps its error (the platform's wait must be honoured); any other row
    # becomes reconnect-only, or the redelivery timer would claim and release it until the
    # adapter is back.
    if adapter is None or relay_unready:
        from gateway.delivery_ledger import is_flood_error

        last_error = str(row.get("last_error") or "")
        reconnect_error = (
            "relay transport not connected" if relay_unready else "send_path_degraded"
        )
        await runner._release_runtime_claim_quiet(
            row["obligation_id"],
            "failed to release undispatched runtime obligation %s",
            error=last_error if is_flood_error(last_error) else reconnect_error,
        )
        return None
    return adapter


async def send_relay_obligation(runner, adapter, row, content, metadata):
    platform = Platform(row["platform"])
    routing = row.get("routing_metadata") or {}
    if not (routing.get("scope_id") or routing.get("user_id")):
        await runner._release_runtime_claim_quiet(
            row["obligation_id"],
            "failed to defer unrouted relay obligation %s",
            error="relay routing metadata unavailable",
        )
        return None
    # The specific failed send owns its routing. Session creation origins and
    # live caches may now describe another author in this same thread.
    metadata = {**(metadata or {}), **routing}
    return await adapter.send_for_platform(
        platform,
        row["chat_id"],
        content,
        metadata=metadata,
        replay=True,
    )


def _schedule_replay(runner, adapter, platform):
    if not runner._running or relay_obligation_adapter(runner, platform) is not adapter:
        return
    # The socket reader delivers outbound_result frames. Run the sweep off
    # the reader so it can keep receiving the replay's acknowledgements.
    task = asyncio.create_task(
        runner._redeliver_failed_obligations_for_platform(
            platform, relay_adapter=adapter
        ),
        name=f"relay-obligation-replay:{platform.value}",
    )
    runner._track_task_in(runner._background_tasks, task)
    task.add_done_callback(
        runner._late_failure_callback("relay obligation replay failed")
    )


def wire_relay_obligation_replay(runner, adapter):
    register = getattr(
        getattr(adapter, "_transport", None), "set_platform_ready_handler", None
    )
    if not callable(register):
        return

    def platform_ready(platform_name):
        try:
            platform = Platform(platform_name)
        except ValueError:
            return
        _schedule_replay(runner, adapter, platform)

    register(platform_ready)


def replay_on_relay_replacement(runner, adapter):
    """A replacement may have acknowledged its platforms before it was registered."""
    for platform in Platform:
        if relay_obligation_adapter(
            runner, platform
        ) is adapter and relay_platform_ready(adapter, platform):
            _schedule_replay(runner, adapter, platform)
