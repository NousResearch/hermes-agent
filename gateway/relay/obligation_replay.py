"""Resolve transports for delivery obligations and wake relay replay on handshake."""

import asyncio
import logging

from gateway.config import Platform

logger = logging.getLogger("gateway.run")

RELAY_NOT_CONNECTED = "relay transport not connected"
# What the transport's own guards answer before writing a frame.
RELAY_LOCAL_REFUSALS = frozenset({RELAY_NOT_CONNECTED, "relay transport closed"})
SETTLE_RETRY_DELAYS = (0.05, 0.25, 1.0)
RELAY_WROTE_NOTHING = "_relay_wrote_nothing"  # set on a claimed row by send_relay_obligation


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


def relay_fronted_platforms(adapter):
    """Platforms the relay's transport advertises. Plugin platforms are not members that
    ``for p in Platform`` yields, so the advertised names are the only complete list."""
    identities = getattr(getattr(adapter, "_transport", None), "_identities", None) or ()
    platforms = []
    for name in dict.fromkeys(str(name) for name, _ in identities):
        try:
            platforms.append(Platform(name))
        except ValueError:
            continue
    return platforms


def relay_refused_before_sending(adapter, row, result) -> bool:
    """The transport's own guard refused this send and wrote no frame while it ran, so the
    claim's attempt bought nothing and is refunded. The words alone prove nothing: a connector
    can answer with them after the frame arrived. A write that raised, a lost or ambiguous
    acknowledgement, a remote rejection or a message id may all mean the reply was delivered."""
    raw = getattr(result, "raw_response", None)
    return (
        getattr(adapter, "platform", None) is Platform.RELAY
        and row.get(RELAY_WROTE_NOTHING) is True
        and str(getattr(result, "error", "") or "") in RELAY_LOCAL_REFUSALS
        and not getattr(result, "message_id", None)
        and not (isinstance(raw, dict) and raw.get("ambiguous"))
    )


async def record_delivered(obligation_id) -> bool:
    """Record a redelivery that already succeeded. A failed write is retried on its own: the
    reply is out, and a row left ``attempting`` would be sent again at the next boot."""
    from gateway.delivery_ledger import mark_delivered

    for delay in (0, *SETTLE_RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            await asyncio.to_thread(mark_delivered, obligation_id)
            return True
        except Exception:
            logger.debug("delivery ledger update failed", exc_info=True)
    logger.warning(
        "obligation %s was redelivered but could not be recorded; the next boot may send it "
        "again with a recovery marker", obligation_id)
    return False


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
            RELAY_NOT_CONNECTED if relay_unready else "send_path_degraded"
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
    transport = getattr(adapter, "_transport", None)
    written = getattr(transport, "write_attempts", None)
    result = await adapter.send_for_platform(
        platform,
        row["chat_id"],
        content,
        metadata=metadata,
        replay=True,
    )
    row[RELAY_WROTE_NOTHING] = type(written) is int and getattr(transport, "write_attempts", None) == written
    return result


def _schedule_replay(runner, adapter, platform):
    if not runner._running or relay_obligation_adapter(runner, platform) is not adapter:
        return
    # One worker per relay platform. A handshake that lands while it runs asks for one more
    # pass instead of starting a second sweep beside it.
    workers = getattr(runner, "_relay_replay_workers", None)
    if workers is None:
        workers = runner._relay_replay_workers = {}
    key = (id(adapter), platform.value)
    if key in workers:
        workers[key] = True
        return
    workers[key] = False

    async def replay():
        try:
            while True:
                await runner._redeliver_failed_obligations_for_platform(platform, relay_adapter=adapter)
                again, workers[key] = workers[key], False
                if not again or not runner._running or relay_obligation_adapter(runner, platform) is not adapter:
                    return
        finally:
            workers.pop(key, None)

    # The socket reader delivers outbound_result frames. Run the sweep off
    # the reader so it can keep receiving the replay's acknowledgements.
    task = asyncio.create_task(replay(), name=f"relay-obligation-replay:{platform.value}")
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
    for platform in relay_fronted_platforms(adapter):
        if relay_obligation_adapter(
            runner, platform
        ) is adapter and relay_platform_ready(adapter, platform):
            _schedule_replay(runner, adapter, platform)
