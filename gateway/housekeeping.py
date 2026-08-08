"""Gateway background-loop daemons (planned-stop watcher, housekeeping sweep,
cron ticker shim, thread-exit waiter, health-export shutdown).

Extracted verbatim from ``gateway/run.py`` (god-file slice 30 of #54962). These
module-level functions run the long-lived background threads of the gateway:
the planned-stop marker watcher (Windows signal bridge), the gateway-only
periodic housekeeping sweep (cache cleanup / curator / auto-archive / memory
trim), the deprecated in-process cron ticker shim, the cooperative
loop-friendly thread-exit waiter, and the idempotent Gateway Health OTLP export
shutdown. ``gateway.run`` re-imports the names at module level, so
``gateway.run.<name>`` references (tests, monkeypatches) stay green.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from agent.async_utils import safe_schedule_threadsafe

# Keep the exact "gateway.run" logger name (matching restart_loop_guard.py) so
# daemon log lines land in the same logger as before the extraction.
logger = logging.getLogger("gateway.run")


def _run_planned_stop_watcher(
    stop_event: threading.Event,
    runner,
    loop: asyncio.AbstractEventLoop,
    shutdown_handler,
    *,
    poll_interval: float = 0.5,
) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown.

    On Windows, ``asyncio.add_signal_handler`` raises NotImplementedError
    for SIGTERM/SIGINT, so the standard signal-driven shutdown path
    never runs when ``hermes gateway stop`` signals the gateway. The
    consequence is that the drain loop is skipped — in-flight agent
    sessions are killed mid-turn and ``resume_pending`` is never set,
    so the next gateway boot has no idea those sessions need to be
    auto-resumed (issue #33778, v0.13.0 session-resume feature broken
    on native Windows).

    This watcher runs on every platform (cheap, defensive) and bridges
    the gap on Windows by translating a filesystem marker into the
    same shutdown-handler invocation a real SIGTERM would have produced
    on POSIX. The CLI's ``hermes_cli.gateway_windows.stop()`` writes
    the marker via ``write_planned_stop_marker(pid)`` and then waits
    for the gateway PID to exit; this watcher is what makes that
    exit happen cleanly.

    On POSIX this is a no-op safety net — the signal handler always
    races us to consuming the marker file because it fires synchronously
    from the kernel's signal delivery.

    Args:
        stop_event: cleared by start_gateway() during normal shutdown
            to tell the watcher to exit.
        runner: the GatewayRunner instance; we check ``_running`` and
            ``_draining`` to avoid triggering shutdown if the gateway
            is already in one of those states.
        loop: the asyncio event loop the shutdown handler must run on.
        shutdown_handler: same callable that's wired to SIGTERM —
            tolerates a ``None`` signal argument (planned stop case)
            and consumes the marker via
            ``consume_planned_stop_marker_for_self()``.
        poll_interval: seconds between marker checks. 0.5s gives a
            responsive shutdown without burning CPU.
    """
    from gateway.status import (
        _get_planned_stop_marker_path,
        planned_stop_marker_targets_self,
    )
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)
            ):
                # A marker existing is NOT sufficient — it may have been
                # written for a PREVIOUS gateway instance (different PID)
                # and left behind because that process exited before the
                # CLI's stop() could clean it up. Firing the handler on a
                # stale/foreign marker drives the gateway into shutdown,
                # then consume_planned_stop_marker_for_self() correctly
                # reports a PID mismatch — but by then we're already
                # stopping, so it's logged as an unexpected "UNKNOWN" exit
                # and the watchdog crash-loops the gateway (issue #34597,
                # a regression from PR #33798 which added this watcher
                # without the PID check).
                #
                # Only fire when the marker actually targets us. The probe
                # is non-destructive on a match (the handler does the
                # authoritative consume on the loop thread) and self-heals
                # by unlinking stale/malformed markers so they cannot wedge
                # a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Drive the same path as a real signal handler.
                # Pass signal=None — the handler tolerates that and consumes
                # the marker via consume_planned_stop_marker_for_self,
                # which also validates target_pid + start_time match us.
                loop.call_soon_threadsafe(shutdown_handler, None)
                # Done — the handler will set _draining; we exit on next tick.
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)



def _start_gateway_housekeeping(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """Background thread for gateway-only periodic chores (NOT cron).

    Split out of the historical ``_start_cron_ticker`` so the cron *trigger*
    can live behind the ``CronScheduler`` provider (built-in or external) while
    these gateway-specific chores keep running independently of which provider
    fires cron. An external scale-to-zero provider has no 60s loop at all, but
    this housekeeping still wants its hourly cadence — so it owns its own loop.

    Refreshes the channel directory every 5 minutes and prunes the
    image/audio/video/document/screenshot caches + expired ``hermes debug
    share`` pastes once per hour, and polls the curator hourly (its inner
    gate enforces the real weekly cadence).
    """
    from gateway.platforms.base import (
        cleanup_audio_cache,
        cleanup_document_cache,
        cleanup_image_cache,
        cleanup_screenshot_cache,
        cleanup_video_cache,
    )
    from hermes_cli.debug import _sweep_expired_pastes

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes
    PASTE_SWEEP_EVERY = 60   # ticks — once per hour
    CURATOR_EVERY = 60       # ticks — poll hourly (inner gate handles the real cadence)
    AUTO_ARCHIVE_EVERY = 60  # ticks — poll hourly (state_meta gate owns the real cadence)
    MEMORY_TRIM_EVERY = 1    # shared helper cooldown bounds actual allocator work

    # Every platform media cache prunes on the same hourly cadence — one loop
    # over (name, cleanup_fn), not a copy-pasted try/except per cache.
    MEDIA_CACHE_CLEANUPS = (
        ("Image", cleanup_image_cache),
        ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache),
        ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache),
    )

    logger.info("Gateway housekeeping started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                if loop is not None:
                    # build_channel_directory is async (Slack web calls), and
                    # this runs in a background thread. Schedule onto the
                    # gateway event loop and wait briefly for completion so
                    # refresh failures are still logged via the except.
                    fut = safe_schedule_threadsafe(
                        build_channel_directory(adapters), loop,
                        logger=logger,
                        log_message="Channel directory refresh scheduling error",
                    )
                    if fut is not None:
                        fut.result(timeout=30)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            for cache_name, cleanup_fn in MEDIA_CACHE_CLEANUPS:
                try:
                    removed = cleanup_fn(max_age_hours=24)
                    if removed:
                        logger.info("%s cache cleanup: removed %d stale file(s)", cache_name, removed)
                except Exception as e:
                    logger.debug("%s cache cleanup error: %s", cache_name, e)

        if tick_count % PASTE_SWEEP_EVERY == 0:
            try:
                deleted, remaining = _sweep_expired_pastes()
                if deleted:
                    logger.info(
                        "Paste sweep: deleted %d expired paste(s), %d pending",
                        deleted, remaining,
                    )
            except Exception as e:
                logger.debug("Paste sweep error: %s", e)

        # Curator — piggy-back on the housekeeping loop so long-running
        # gateways get weekly skill maintenance without needing restarts.
        # maybe_run_curator() is internally gated by config.interval_hours
        # (7 days by default), so CURATOR_EVERY is just the poll rate — the
        # real work only fires once per config interval.
        if tick_count % CURATOR_EVERY == 0:
            try:
                from agent.curator import maybe_run_curator
                maybe_run_curator(
                    idle_for_seconds=float("inf"),
                    on_summary=lambda msg: logger.info("curator: %s", msg),
                )
            except Exception as e:
                logger.debug("Curator tick error: %s", e)

            # Skill Sync — best-effort periodic pull on the same cadence.
            # Inert unless the access gate is open and a sync base URL is
            # configured; never raises.
            try:
                from tools.skills_sync_client import maybe_pull_skills
                maybe_pull_skills()
            except Exception as e:
                logger.debug("Sync pull tick error: %s", e)

            # Org-shared skills. Gated on real org membership (the token must
            # carry an org role), so a solo account never reaches the network.
            try:
                from tools.skills_sync_client import maybe_pull_org_skills
                maybe_pull_org_skills()
            except Exception as e:
                logger.debug("Org sync pull tick error: %s", e)

        # Stale-session auto-archive — a live timer, so gateways that stay up
        # for weeks keep sweeping on schedule (the startup hook fires once).
        # maybe_auto_archive() is gated by sessions.min_interval_hours in
        # state_meta; this is just the poll rate. Opens its own SessionDB —
        # SQLite connections are thread-bound and this runs off-loop.
        if tick_count % AUTO_ARCHIVE_EVERY == 0:
            try:
                from hermes_cli.config import load_config as _load_full_config
                from hermes_state import SessionDB
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_archive", False):
                    _adb = SessionDB()
                    try:
                        _adb.maybe_auto_archive(
                            idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                            min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        )
                    finally:
                        _adb.close()
            except Exception as e:
                logger.debug("Auto-archive tick error: %s", e)

        # This is the long-lived messaging-gateway counterpart to the TUI idle
        # reaper. The helper is config-gated and rate-limited, so calling it on
        # the 60s housekeeping cadence does not create a trim storm.
        if tick_count % MEMORY_TRIM_EVERY == 0:
            try:
                from hermes_cli.mem_trim import trim_memory

                trim_memory(reason="messaging gateway housekeeping")
            except Exception as exc:
                # debug, not warning: sibling housekeeping branches all log
                # failures at debug, and a persistent failure (e.g. broken
                # import after a partial update) would otherwise warn every
                # 60s forever.
                logger.debug(
                    "gateway housekeeping memory trim failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        stop_event.wait(timeout=interval)
    logger.info("Gateway housekeeping stopped")



def _start_cron_ticker(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """DEPRECATED shim — preserved for backward compatibility.

    The cron trigger now lives behind the ``CronScheduler`` provider
    (``cron.scheduler_provider``); the gateway resolves a provider and runs its
    ``start()`` directly (see ``start_gateway``). This shim runs ONLY the
    built-in in-process tick loop, exactly as before, for any external caller
    or test that still references this symbol (e.g. hermes_cli/debug.py). It no
    longer runs gateway housekeeping — that moved to
    ``_start_gateway_housekeeping``.
    """
    from cron.scheduler_provider import InProcessCronScheduler
    InProcessCronScheduler().start(stop_event, adapters=adapters, loop=loop, interval=interval)



# Upper bound for cooperatively draining the cron ticker on shutdown. The cron
# thread delivers via ``safe_schedule_threadsafe`` and blocks on
# ``future.result(timeout=60)`` (see cron/scheduler.py::_deliver_result), so a
# single in-flight delivery unblocks within ~60s. The extra margin covers the
# hop back through run_one_job's bookkeeping.
_CRON_SHUTDOWN_DRAIN_TIMEOUT = 65.0

# Upper bound for cooperatively draining the housekeeping ticker on shutdown.
# Housekeeping periodically refreshes the channel directory via
# ``safe_schedule_threadsafe(build_channel_directory(...), loop)`` and blocks on
# ``fut.result(timeout=30)`` (see ``_start_gateway_housekeeping``) — the same
# loop-scheduled-future pattern as cron. So the cooperative bound must cover
# that 30s future (plus margin) rather than the old 5s join, otherwise a
# channel-directory refresh in flight at shutdown gets abandoned mid-resolve.
# Unlike a dropped cron delivery this is not user-facing (it self-heals on the
# next tick), but bounding it correctly keeps the drain honest.
_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT = 35.0



async def _await_thread_exit(
    thread: Optional[threading.Thread], timeout: float, poll: float = 0.1
) -> bool:
    """Wait for a daemon thread to exit WITHOUT blocking the event loop.

    A synchronous ``thread.join()`` here would freeze the event loop — fatal
    for the cron ticker, whose in-flight delivery is a coroutine scheduled onto
    *this* loop via ``safe_schedule_threadsafe``. Blocking the loop deadlocks
    that delivery (the loop can never run it), so ``join(timeout=5)`` always
    times out and the message is silently dropped on restart (#58818).

    Polling ``is_alive()`` with ``await asyncio.sleep`` keeps the loop running
    so the pending delivery completes, then the ticker sees ``stop_event`` and
    exits. Returns True if the thread exited within ``timeout``.
    """
    if thread is None:
        return True
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
    while thread.is_alive() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(poll)
    return not thread.is_alive()



def _shutdown_gateway_health_export(runner: Any) -> None:
    """Idempotently drain and detach Gateway Health OTLP export."""
    runtime = getattr(runner, "_gateway_health_export_runtime", None)
    if runtime is None:
        return
    runner._gateway_health_export_runtime = None
    try:
        runtime.shutdown()
    except Exception:
        logger.debug("gateway health OTLP export shutdown failed", exc_info=True)
