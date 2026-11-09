"""Route the primary gateway's persisted relay replies and wake replay on handshake."""

import asyncio

from gateway.config import Platform


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
