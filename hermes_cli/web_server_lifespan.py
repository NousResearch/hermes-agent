"""Web-server process lifespan: startup/shutdown of the background work every surface runs,
plus the in-process cron ticker for surfaces no gateway serves.

``web_server`` builds ``FastAPI(lifespan=_lifespan)``. Helpers that stay on the facade
(``_warm_gateway_module``, ``PTY_REGISTRY``, ``run_reaper``, the selftest/auto-archive loops,
``_terminate_desktop_managed_gateway``, ...) are read as ``web_server.<name>`` at call time so
``monkeypatch.setattr(web_server, ...)`` still intercepts them.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from hermes_cli.process_identity import is_desktop_owned_backend
from hermes_cli.web_server_surface import policy

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fastapi import FastAPI

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The desktop spawns a ``hermes dashboard`` backend, not a gateway, so without
    this a cron created in the app would never fire (no live adapters; delivery
    falls back to the per-platform send path). The primary backend outlives the
    per-profile pool (reaped after ~10 idle minutes), so it ticks EVERY local
    profile's store like a multiplex gateway; external providers keep the
    single-store behavior (registries are not profile-scoped). Cross-process
    safe: the built-in tick takes the per-store ``cron/.tick.lock``.

    Every local profile's store is ticked, not just this backend's own (#69377's desktop sibling): the
    desktop pools per-profile backends and reaps them after ~10 idle minutes, so a secondary profile's
    ticker dies with its backend and that profile's jobs silently stop firing until the user next opens it
    ("tasks on the sleeping profile could be idle" — community report, Aug 2026).
    """
    from cron.scheduler_provider import InProcessCronScheduler, resolve_cron_scheduler

    provider = resolve_cron_scheduler()

    start_kwargs: dict = {"interval": interval}
    if isinstance(provider, InProcessCronScheduler):
        try:
            from hermes_cli.profiles import (
                _check_gateway_running, _served_by_running_multiplexer, profiles_to_serve)

            # Same served set as the multiplexer: default + every live profile under profiles/.
            # The ticker re-enumerates this callable every cycle. Passing a
            # startup snapshot leaves deleted profiles in the scheduler until
            # restart, which both writes their removed stores and keeps stale
            # profiles alive in Desktop's background work.
            profile_homes = lambda: list(profiles_to_serve(multiplex=True))
            initial_profile_homes = profile_homes()
            if initial_profile_homes:
                # Even one profile needs the per-tick gateway gate; otherwise
                # Desktop races its dedicated gateway for the same cron store.
                start_kwargs["profile_homes"] = profile_homes
                # Stand down, per tick, for a profile already owned by a gateway — its OWN
                # process, or the live default multiplexer (a served satellite has no gateway.pid
                # of its own). That gateway ticks with live adapters; winning the tick-lock race
                # here would deliver through the standalone path (#100489, #107485).
                start_kwargs["profile_gate"] = lambda name, home: not (
                    _check_gateway_running(Path(home))
                    or (name != "default" and _served_by_running_multiplexer(name)))
                from hermes_logging import enable_profile_log_routing

                enable_profile_log_routing(initial_profile_homes)
                _log.info(
                    "Desktop cron scheduler will tick %d profile(s): %s",
                    len(initial_profile_homes),
                    [name for name, _home in initial_profile_homes],
                )
        except Exception:
            # Fail open to the single-store ticker so the active profile keeps firing.
            _log.exception("Desktop cron: profile enumeration failed; ticking active profile only")

    _log.info("Desktop cron scheduler started (provider=%s, interval=%ds)", provider.name, interval)
    provider.start(stop_event, **start_kwargs)


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    from hermes_cli import web_server

    app.state.event_channels = {}  # dict[str, set]
    app.state.event_lock = asyncio.Lock()
    app.state.pty_active_session_files = {}  # dict[str, Path]
    # Serializes chat-argv resolution so concurrent /api/pty connections don't
    # overlap ``npm install`` / ``npm run build``. Locks live on app.state (not
    # module globals) so they bind to the running loop, not the import-time one.
    app.state.chat_argv_lock = asyncio.Lock()

    # Bring state.db schema current BEFORE the first session-list poll
    # (#79531/#80037): a store left behind by `hermes update` otherwise 500s
    # every poll while the read-probe heal loses to sibling lock contention.
    # Off-thread so a locked store never delays the socket (Desktop
    # ready-probe times out at 10s, GH-73083). NOT a daemon, and joined at
    # shutdown: its sqlite connection must be closed by the thread that is
    # stepping it. A daemon copy that outlived the lifespan had its
    # connection closed from the main thread mid-probe (pytest's leaked-DB
    # sweep) and segfaulted the interpreter. The worker is time-bounded by
    # SessionDB's lock patience, so the join cannot hang shutdown.
    eager_reconcile_thread = threading.Thread(
        target=web_server._eager_reconcile_own_session_db,
        name="statedb-eager-reconcile",
    )
    eager_reconcile_thread.start()

    # Import hermes_cli.gateway *before* the yield: on Windows + 3.11 the
    # import holds the GIL, so run_in_executor still froze the loop 15-22s and
    # the Desktop's 10s ready-probe timed out (GH-73083).
    web_server._warm_gateway_module()

    # Snapshot the checkout revision so lazy-import paths (model picker) can
    # refuse with "restart required" after `hermes update` replaced the code
    # (#86207); the update flow does not reliably restart the dashboard.
    from gateway.code_skew import record_boot_fingerprint

    record_boot_fingerprint()

    # Hosted Bot rooms belong to the backend process. Recovery may need a
    # contended state.db migration, so keep it off the pre-yield path: Group
    # Chat must degrade on its own rather than block every Desktop feature.
    from tui_gateway import methods_groups as _hosted_groups
    import tui_gateway.server  # noqa: F401

    try:
        tui_gateway.server.install_tui_message_injector()
    except Exception:
        _log.warning("TUI message injector did not install", exc_info=True)

    hosted_room_start_cancel = threading.Event()

    def _start_hosted_rooms() -> None:
        try:
            _hosted_groups.start_hosted_room_service()
        except Exception:
            _log.exception("Hosted Group Chat recovery failed during backend startup")
        finally:
            if hosted_room_start_cancel.is_set():
                _hosted_groups.stop_hosted_room_service(timeout=1.0)

    hosted_room_start_thread = threading.Thread(
        target=_start_hosted_rooms,
        daemon=True,
        name="hosted-room-startup",
    )
    hosted_room_start_thread.start()

    # Gateway lifecycle ownership remains exclusive to Electron-spawned backends.
    cron_stop: "threading.Event | None" = None
    cron_thread: "threading.Thread | None" = None
    desktop_owned = is_desktop_owned_backend()
    if desktop_owned:
        # Reap an orphaned gateway from an abnormal previous exit (reparented to
        # launchd, still holding the platform WebSocket) before forking a fresh
        # one that would race the same credential (#77276). Runs
        # unconditionally; protection of a healthy standalone gateway lives
        # INSIDE the reaper (registration probed with cleanup_stale=False).
        try:
            from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

            _reap_unsupervised_gateway_orphans()
        except Exception:
            _log.exception("Desktop startup: orphan gateway reap failed")

    # Standalone Webapp, like Desktop, has no gateway to run its scheduled jobs.
    # Reuse the profile-aware ticker without adopting Electron's gateway cleanup.
    # Plain dashboard deployments still rely on their existing gateway.
    if desktop_owned or policy(app.state).cron_ticker:
        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(cron_stop,),
            daemon=True,
            name="desktop-cron-ticker",
        )
        cron_thread.start()

    # Reap idle/dead keep-alive PTY sessions (30-min TTL).
    web_server.PTY_REGISTRY._closed = False
    pty_reaper_task = asyncio.create_task(web_server.run_reaper(web_server.PTY_REGISTRY))
    from hermes_cli.web_host_terminal_sessions import host_terminal_lifespan
    # Host shells, and the reaper that retires them, exist only where the surface allows them.
    host_terminals = host_terminal_lifespan(app) if policy(app.state).host_terminal else nullcontext()
    await host_terminals.__aenter__()
    # Periodic authenticated self-test feeding the ``dashboard`` component on /api/status.
    selftest_task = asyncio.create_task(web_server._dashboard_selftest_loop())
    # Live auto-archive timer, independent of list requests.
    auto_archive_task = asyncio.create_task(web_server._auto_archive_ticker_loop())

    # Managed local runtime (local_runtime.enabled): bring llama-server back so a
    # restart doesn't strand a llamacpp main model. Off-thread and best-effort;
    # failure falls back to cloud providers like a cold start. Server only —
    # models load on first inference (an empty router holds no VRAM).
    def _boot_local_runtime():
        try:
            from hermes_cli.config import load_config
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            ensure_local_runtime(load_config())
        except Exception as exc:  # noqa: BLE001
            _log.warning("local runtime boot failed: %s", exc)

    threading.Thread(target=_boot_local_runtime, daemon=True, name="local-runtime-boot").start()

    # Nous free tier: the ONE place its identity is created. Inventories credentials, mints only
    # when HERMES_GUEST_ONBOARDING=1, records the answer for setup.status / free_tier.status and
    # broadcasts `setup.ready`. Off-thread so a slow portal never delays the socket; the desktop's
    # first setup.status waits on the record (bounded) instead.
    from hermes_cli.free_tier_bootstrap import start_background_bootstrap

    start_background_bootstrap()

    try:
        yield
    finally:
        try:
            tui_gateway.server.clear_tui_message_injector()
        except Exception:
            _log.debug("TUI message injector clear skipped", exc_info=True)
        hosted_room_start_cancel.set()
        _hosted_groups.stop_hosted_room_service(timeout=5.0)
        hosted_room_start_thread.join(timeout=1.0)
        if cron_stop is not None:
            cron_stop.set()
        pty_reaper_task.cancel()
        selftest_task.cancel()
        auto_archive_task.cancel()
        await web_server.PTY_REGISTRY.close_all()
        await host_terminals.__aexit__(None, None, None)
        # Stop the managed llama-server with its parent (an orphan pins VRAM).
        try:
            from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime

            shutdown_local_runtime()
        except Exception:  # noqa: BLE001
            pass
        if desktop_owned:
            web_server._terminate_desktop_managed_gateway()
        eager_reconcile_thread.join()
