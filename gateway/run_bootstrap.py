"""Process startup phases; diagnostics remain in the gateway facade."""
from __future__ import annotations

import threading
from typing import Optional

from gateway.config import GatewayConfig

async def _start_gateway_replace_existing_instance(existing_pid: int, replace: bool) -> bool:
    """Handle a live gateway PID under this HERMES_HOME: replace it (``--replace``) or refuse.
    Returns False when startup must abort (refused, permission denied, target still alive)."""
    from gateway.run import (_clear_takeover_marker_quiet, _replace_target_belongs_to_other_profile, _wait_for_pid_exit, get_hermes_home, logger, suppress)
    from gateway.status import get_process_start_time, remove_pid_file, terminate_pid
    if not replace:
        hermes_home = str(get_hermes_home())
        logger.error(
            "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
            "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
            existing_pid, hermes_home)
        print(
            f"\n❌ Gateway already running (PID {existing_pid}).\n"
            f"   Use 'hermes gateway restart' to replace it,\n"
            f"   or 'hermes gateway stop' to kill it first.\n"
            f"   Or use 'hermes gateway run --replace' to auto-replace.\n")
        return False

    # Never signal a process not provably ours (a poisoned PID record → cross-profile restart loop).
    if _replace_target_belongs_to_other_profile(existing_pid):
        from gateway.status import _get_process_hermes_home
        logger.error(
            "Refusing --replace: PID %d cannot be proven to belong "
            "to this profile's gateway (HERMES_HOME %s). Remove the "
            "stale PID record or stop the owning profile explicitly.",
            existing_pid, _get_process_hermes_home())
        return False
    existing_start_time = get_process_start_time(existing_pid)
    logger.info("Replacing existing gateway instance (PID %d) with --replace.", existing_pid)
    # Takeover marker: target exits 0 on our SIGTERM (exit 1 → systemd Restart=on-failure flap loop).
    try:
        from gateway.status import write_takeover_marker
        write_takeover_marker(existing_pid)
    except Exception as e:
        logger.debug("Could not write takeover marker: %s", e)
    # Snapshot children BEFORE signalling: reparented orphans are invisible yet hold scoped token locks.
    try:
        from gateway.status import _snapshot_gateway_children
        _old_gateway_children = _snapshot_gateway_children(existing_pid)
    except Exception:
        _old_gateway_children = []
    try:
        terminate_pid(existing_pid, force=False)
    except ProcessLookupError:
        pass  # Already gone
    except (PermissionError, OSError):
        logger.error("Permission denied killing PID %d. Cannot replace.", existing_pid)
        _clear_takeover_marker_quiet()
        return False
    # Up to 10s for SIGTERM, then SIGKILL.
    if not await _wait_for_pid_exit(existing_pid, 20, 0.5):
        logger.warning("Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.", existing_pid)
        old_gateway_exited = False
        try:
            terminate_pid(existing_pid, force=True, expected_start_time=existing_start_time)
        except ProcessLookupError:
            old_gateway_exited = True
        except (PermissionError, OSError):
            pass
        # Confirm SIGKILL took (D-state/zombie) before clearing PID/locks, or two gateways share a token.
        if not old_gateway_exited and not await _wait_for_pid_exit(existing_pid, 20, 0.25):
            logger.error(
                "Old gateway (PID %d) still appears alive after SIGKILL; "
                "aborting replacement to avoid a duplicate gateway.", existing_pid)
            _clear_takeover_marker_quiet()
            return False
    # Reap orphaned children (POSIX; mirrors Windows taskkill /T) so they stop holding scoped token locks.
    try:
        from gateway.status import reap_gateway_children
        reap_gateway_children(_old_gateway_children, parent_pid=existing_pid)
    except Exception:
        logger.debug("Child reap for replaced gateway PID %d failed", existing_pid, exc_info=True)
    remove_pid_file()
    # remove_pid_file() is a no-op when the PID doesn't match; force-unlink covers a crashed old process.
    with suppress(Exception):
        (get_hermes_home() / "gateway.pid").unlink(missing_ok=True)
    # The old process may not have consumed the marker (SIGKILL'd before its handler read it).
    _clear_takeover_marker_quiet()
    # Stopped (Ctrl+Z) processes don't release scoped locks on exit; stale lock files block the new gateway.
    try:
        from gateway.status import release_all_scoped_locks
        _released = release_all_scoped_locks(owner_pid=existing_pid, owner_start_time=existing_start_time)
        if _released:
            logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
    except Exception:
        pass
    return True


def _start_gateway_configure_logging(verbosity: Optional[int]) -> None:
    """Sync bundled skills, set up file logging + startup security audit, and the -v/-q stderr handler."""
    from gateway.run import (_best_effort, _gateway_stderr_formatter, _hermes_home, logging)
    def _sync_skills() -> None:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)

    _best_effort(_sync_skills)

    # Centralized logging (agent.log INFO+, errors.log WARNING+, gateway.log gateway-only); idempotent.
    from hermes_logging import setup_logging, _safe_stderr
    setup_logging(hermes_home=_hermes_home, mode="gateway")

    def _security_audit() -> None:
        # Warn-on-load, never blocks: surfaces root / weak-SSH / unauthenticated-listener exposure.
        from hermes_cli.security_audit_startup import log_startup_security_warnings

        def _raw_cfg():
            from hermes_cli.config import read_raw_config
            return read_raw_config()

        log_startup_security_warnings(hermes_home=_hermes_home, config=_best_effort(_raw_cfg))

    _best_effort(_security_audit, "Startup security audit failed (non-fatal): %s")

    # Optional stderr handler from -v/-q: None (quiet) = none; 0 = WARNING; 1 = INFO; 2+ = DEBUG.
    if verbosity is not None:
        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler(_safe_stderr())
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(_gateway_stderr_formatter())
        root = logging.getLogger()
        root.addHandler(_stderr_handler)
        if _stderr_level < root.level:  # so DEBUG records can reach the handler
            root.setLevel(_stderr_level)


def _start_gateway_make_shutdown_signal_handler(runner, _signal_initiated_shutdown: list):
    """Build the SIGINT/SIGTERM handler; ``_signal_initiated_shutdown[0]`` records an unplanned signal."""
    from gateway.run import (_best_effort, _hermes_home, asyncio, logger, signal)
    def shutdown_signal_handler(received_signal=None):
        # Planned --replace takeover (sibling marked this PID): exit 0 so systemd won't revive us.
        def _takeover() -> bool:
            from gateway.status import consume_takeover_marker_for_self
            return consume_takeover_marker_for_self()

        # Planned stop: CLI marks first, else its SIGTERM looks like an external kill. SIGINT = Ctrl+C.
        def _planned_stop() -> bool:
            from gateway.status import consume_planned_stop_marker_for_self
            return consume_planned_stop_marker_for_self()

        # Fast (<10ms) sync snapshot: stdlib + /proc, no subprocesses (`ps aux` here once blocked ~3s).
        def _snapshot():
            from gateway.shutdown_forensics import snapshot_shutdown_context
            return snapshot_shutdown_context(received_signal)

        planned_takeover = bool(_best_effort(_takeover, "Takeover marker check failed: %s"))
        planned_stop = received_signal == signal.SIGINT or (
            not planned_takeover and bool(_best_effort(_planned_stop, "Planned stop marker check failed: %s")))
        _shutdown_ctx = _best_effort(_snapshot, "snapshot_shutdown_context failed: %s")
        sig_name = _shutdown_ctx["signal"] if _shutdown_ctx else None

        if planned_takeover:
            logger.info("Received %s as a planned --replace takeover — exiting cleanly", sig_name or "SIGTERM")
        elif planned_stop:
            logger.info("Received %s as a planned gateway stop — exiting cleanly", sig_name or "SIGTERM/SIGINT")
        else:
            # Mirrored onto the runner so _stop_impl suppresses the gateway_state=stopped persist for
            # unexpected signals; operator stops take the `planned_stop` branch and leave it False (DO persist).
            _signal_initiated_shutdown[0] = runner._signal_initiated_shutdown = True
            logger.info("Received %s — initiating shutdown", sig_name or "SIGTERM/SIGINT")

        if _shutdown_ctx is not None:
            def _log_context() -> None:
                # The most useful line for "gateway keeps dying" tickets.
                from gateway.shutdown_forensics import format_context_for_log
                logger.warning("Shutdown context: %s", format_context_for_log(_shutdown_ctx))

            def _diagnostic() -> None:
                # Heavyweight (ps auxf, pstree, dmesg), detached so it finishes even if our cgroup is torn
                # down; bounded by an internal timeout, never blocks.
                from gateway.shutdown_forensics import spawn_async_diagnostic
                spawn_async_diagnostic(
                    _hermes_home / "logs" / "gateway-shutdown-diag.log", _shutdown_ctx["signal"], timeout_seconds=5.0)

            _best_effort(_log_context, "format_context_for_log failed: %s")
            _best_effort(_diagnostic, "spawn_async_diagnostic failed: %s")
        asyncio.create_task(runner.stop())
    return shutdown_signal_handler


def _start_gateway_claim_pid_file() -> bool:
    """Claim the runtime lock + PID file (O_EXCL winner is the authoritative gateway). False = lost."""
    from gateway.run import (logger, os)
    import atexit
    from gateway.status import (
        acquire_gateway_runtime_lock, get_running_pid, release_gateway_runtime_lock,
        remove_pid_file, write_pid_file)
    _current_pid = get_running_pid()
    if _current_pid is not None and _current_pid != os.getpid():
        logger.error("Another gateway instance (PID %d) started during our startup. "
                     "Exiting to avoid double-running.", _current_pid)
        return False
    if not acquire_gateway_runtime_lock():
        logger.error("Gateway runtime lock is already held by another instance. Exiting.")
        return False
    try:
        write_pid_file()
    except FileExistsError:
        release_gateway_runtime_lock()
        logger.error("PID file race lost to another gateway instance. Exiting.")
        return False
    atexit.register(remove_pid_file)
    atexit.register(release_gateway_runtime_lock)
    return True


async def _start_gateway_start_control_socket(runner):
    """Start the gateway control socket (identify/status/pause-for-update); None when unavailable."""
    from gateway.run import (asyncio, logger, os, threading)
    import atexit
    _control_server = None
    try:
        # Started immediately after the PID-file claim: winning that O_EXCL race is the moment this process
        # becomes the authoritative gateway for its HERMES_HOME, so from here on "does a socket answer?" is
        # a truthful liveness/identity query for updater and fleet consumers. Strictly non-fatal: a bind
        # failure only means consumers fall back to the process-scan/state-file layer, exactly as before
        # this feature. See #92091.
        from gateway.control_socket import GatewayControlServer, build_identify_payload
        descriptor = runner.session_runtime_descriptor

        def _identify_runtime():
            payload = build_identify_payload()
            payload.update({key: descriptor[key] for key in (
                "instance_id", "runtime_protocol", "authority_epoch", "state", "api_origin",
                "served_profiles", "capabilities") if key in descriptor})
            if getattr(runner, '_draining', False):
                payload.update(state='draining', capabilities=[])
            payload["supervisor"] = {"manual": "none", "desktop": "none"}.get(
                payload.get("supervisor"), payload.get("supervisor", "none"))
            return payload
        # pause-for-update: the updater asks us to drain + exit (freeing venv handles) vs. a tree-kill
        # (same path as SIGUSR1). Handler runs on the socket executor thread, so marshal onto the loop.
        # pause-for-update (#92091 step 2): the updater asks this gateway to drain in-flight turns and exit
        # cleanly — releasing every venv file handle — instead of being tree-killed mid-turn. Same drain
        # path as SIGUSR1/service restarts (request_restart(via_service=True)); the updater (or the service
        # manager) relaunches after the code swap.
        _main_loop = asyncio.get_running_loop()

        def _pause_for_update_handler() -> dict:
            try:
                from hermes_cli.gateway import _get_restart_drain_timeout
                _drain = float(_get_restart_drain_timeout())
            except Exception:
                _drain = 30.0
            accepted_box: list[bool] = []
            _done = threading.Event()

            def _request() -> None:
                try:
                    accepted_box.append(runner.request_restart(detached=False, via_service=True))
                finally:
                    _done.set()

            _main_loop.call_soon_threadsafe(_request)
            _done.wait(timeout=5.0)
            accepted = bool(accepted_box and accepted_box[0])
            return {
                "pausing": accepted, "already_stopping": not accepted,
                "pid": os.getpid(), "drain_timeout": _drain}

        _control_server = GatewayControlServer(
            verb_handlers={"pause-for-update": _pause_for_update_handler, "identify": _identify_runtime})
        _control_server.ticket_store = runner.session_ticket_store
        if not await _control_server.start():
            _control_server = None
        else:
            atexit.register(_control_server.cleanup_files)
    except Exception as _cs_exc:
        logger.debug("Control socket startup failed (non-fatal): %s", _cs_exc)
        _control_server = None
    return _control_server


def _start_gateway_start_cron_and_housekeeping(runner):
    """Start the cron scheduler thread + gateway housekeeping thread; returns
    ``(cron_stop, cron_provider, cron_thread, housekeeping_thread)``."""
    from gateway.run import (Any, Dict, Platform, _cron_tick_profile_homes, _start_gateway_housekeeping, asyncio, logger, threading)
    # The event loop is passed so cron delivery can use live adapters (E2EE support).
    from cron.scheduler_provider import (
        InProcessCronScheduler, resolve_cron_scheduler, scheduler_for_profile_mode)
    cron_stop = threading.Event()
    multiplex_cron = bool(getattr(runner.config, "multiplex_profiles", False))
    cron_provider = scheduler_for_profile_mode(
        resolve_cron_scheduler(), multiplex_profiles=multiplex_cron)
    cron_start_kwargs: Dict[str, Any] = {"adapters": runner.adapters, "loop": asyncio.get_running_loop()}

    # Multiplex: tell the ticker which profile homes to tick, else secondary profiles' jobs never run.
    if isinstance(cron_provider, InProcessCronScheduler) and multiplex_cron:
        try:
            profile_homes = _cron_tick_profile_homes(runner.config)
            if profile_homes:
                cron_start_kwargs["profile_homes"] = profile_homes
                # Per-profile adapters so each profile's cron output goes via its own bot, not the default's.
                cron_start_kwargs["profile_adapters"] = getattr(runner, "_profile_adapters", None)
                # runner.adapters belongs to the LAUNCH profile (default, or the --profile name); naming
                # it keeps the ticker from routing a secondary's cron through that bot and lets a named
                # multiplexer's own jobs reuse its live adapters.
                cron_start_kwargs["default_profile"] = runner._primary_profile_name
                logger.info(
                    "Cron scheduler will tick %d profile(s) under multiplex: %s", len(profile_homes),
                    [p[0] if isinstance(p, tuple) else p for p in profile_homes])
        except Exception as exc:
            logger.warning("Could not resolve profile homes for multiplex cron: %s", exc)

    # Only the in-process ticker polls local due jobs, so only it gets the external-drain dispatch gate.
    if isinstance(cron_provider, InProcessCronScheduler):
        cron_start_kwargs["can_dispatch"] = lambda: not (
            runner._draining or runner._external_drain_active)
    cron_thread = threading.Thread(
        target=cron_provider.start, args=(cron_stop,), kwargs=cron_start_kwargs, daemon=True,
        name="cron-scheduler")
    from gateway.runtime_ownership import process_ownership
    process_ownership.start_writer(cron_thread)

    # External providers fire over loopback HTTP to THIS process's api_server; if it never came up (usually
    # API_SERVER_KEY missing) every fire fails while manual runs work — misread as a job bug. Say it ONCE.
    if not isinstance(cron_provider, InProcessCronScheduler):
        try:
            _has_api_server = Platform.API_SERVER in (runner.adapters or {})
        except Exception:
            _has_api_server = True  # never let the tell break startup
        if not _has_api_server:
            logger.warning(
                "Cron provider '%s' is active but the api_server adapter is "
                "NOT running in this gateway — scheduled fires arrive over "
                "loopback HTTP and will all fail (jobs only run when "
                "triggered manually). Most common cause: API_SERVER_KEY is "
                "missing from this gateway process's environment. Restart "
                "the gateway through its supervisor (`hermes gateway "
                "restart`) so the profile env loads.",
                getattr(cron_provider, "name", "external"))

    # Gateway-only housekeeping runs independently of the cron provider; shares cron_stop for shutdown.
    housekeeping_thread = threading.Thread(
        target=_start_gateway_housekeeping, args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop(),
                "cron_provider": cron_provider, "runner": runner},
        daemon=True, name="gateway-housekeeping")
    process_ownership.start_writer(housekeeping_thread)
    return cron_stop, cron_provider, cron_thread, housekeeping_thread


async def _start_gateway_shutdown_tail(
    runner, _control_server, cron_stop: threading.Event, cron_provider,
    cron_thread: threading.Thread, housekeeping_thread: threading.Thread,
    _planned_stop_watcher_stop: threading.Event, _planned_stop_watcher_thread: threading.Thread,
    _signal_initiated_shutdown: list) -> bool:
    """Post-``wait_for_shutdown`` teardown; returns the process exit verdict (True = exit 0)."""
    from gateway.run import (_CRON_SHUTDOWN_DRAIN_TIMEOUT, _HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT, _await_thread_exit, _best_effort, _exit_with_failure_verdict, _resolve_gateway_exit_verdict, _shutdown_mcp_servers_nonblocking, _stop_cron_provider, logger, suppress)
    # Control socket first: once shutdown begins we are no longer a truthful "serving here" answer and a
    # successor must be able to bind. Early-exit paths rely on the atexit cleanup_files hook instead.
    if _control_server is not None:
        try:
            await _control_server.stop()
        except Exception:
            logger.debug("Control socket stop failed (non-fatal)", exc_info=True)

    def _stop_keepalive() -> None:
        from hermes_cli.nous_auth_keepalive import stop_nous_auth_keepalive
        stop_nous_auth_keepalive()

    _best_effort(_stop_keepalive)
    if _exit_with_failure_verdict(runner):
        return False

    # Never join(): an in-flight cron delivery is a coroutine on THIS loop; a sync join would drop it.
    # Stop cron scheduler + housekeeping cleanly. These MUST be awaited cooperatively, not join()ed. A cron
    # delivery in flight when the gateway restarts is a coroutine scheduled onto THIS event loop
    # (safe_schedule_threadsafe); the ticker thread is blocked on its future.result(). A synchronous
    # cron_thread.join() would block the loop, so that delivery could never run — it timed out and the
    # message was silently dropped (#58818). Awaiting keeps the loop alive so the in-flight delivery
    # finishes before we tear down.
    cron_stop.set()
    _stop_cron_provider(cron_provider)
    if not await _await_thread_exit(cron_thread, timeout=_CRON_SHUTDOWN_DRAIN_TIMEOUT):
        logger.warning("Cron ticker did not exit within %.0fs of shutdown — an in-flight "
                       "delivery may have been dropped.", _CRON_SHUTDOWN_DRAIN_TIMEOUT)
    await _await_thread_exit(housekeeping_thread, timeout=_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT)

    # Stop the planned-stop watcher (daemon=True so this is belt-and-suspenders).
    _planned_stop_watcher_stop.set()
    _planned_stop_watcher_thread.join(timeout=2)

    with suppress(Exception):
        await _shutdown_mcp_servers_nonblocking()

    return _resolve_gateway_exit_verdict(runner, _signal_initiated_shutdown[0])



async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """Start the gateway and run until interrupted; False if it failed to start (non-zero exit so
    systemd can auto-restart). ``replace`` kills any existing instance first (avoids restart-loop deadlocks)."""
    from gateway.run import (
        GatewayRunner,
        _best_effort,
        _discover_gateway_mcp_tools,
        _enable_multiplex_log_routing,
        _ensure_windows_gateway_venv_imports,
        _gateway_loop_exception_handler,
        _multiplex_profile_homes,
        _resolve_gateway_exit_verdict,
        _run_planned_stop_watcher,
        _shutdown_gateway_health_export,
        _shutdown_mcp_servers_nonblocking,
        asyncio,
        get_hermes_home,
        load_gateway_config_for_runner,
        logger,
        os,
        signal,
        suppress,
        threading,
    )
    # Set here (not at import) so incidental gateway.run imports from CLI code don't poison it.
    os.environ["HERMES_EXEC_ASK"] = "1"

    from hermes_cli.resource_limits import apply_nofile_soft_limit
    apply_nofile_soft_limit()

    # Snapshot the revision while sys.modules matches disk so a later `git pull` is detected safely.
    from gateway.code_skew import record_boot_fingerprint
    record_boot_fingerprint()

    # Duplicate-instance guard scoped to HERMES_HOME; distinct-home multi-profile setups coexist.
    from gateway.status import get_running_pid
    existing_pid = get_running_pid()
    if (existing_pid is not None and existing_pid != os.getpid()
            and not await _start_gateway_replace_existing_instance(existing_pid, replace)):
        return False

    from gateway.runtime_ownership import process_ownership, OwnershipConflict
    from gateway.status import remove_pid_file, release_gateway_runtime_lock
    resolved_config = config if config is not None else load_gateway_config_for_runner()
    profile_homes = (_multiplex_profile_homes(resolved_config)
                     if getattr(resolved_config, 'multiplex_profiles', False) else [])
    try:
        process_ownership.reserve([get_hermes_home(), *(home for _, home in profile_homes)])
    except (OwnershipConflict, OSError) as exc:
        logger.error("Cannot reserve gateway profiles: %s", exc)
        return False
    # Freeze discovery: later profile additions must restart and reserve before opening stores.
    if profile_homes:
        resolved_config._runtime_profile_homes = tuple(profile_homes)
    if not _start_gateway_claim_pid_file():
        release_gateway_runtime_lock()
        return False

    _control_server = None
    runner = None
    _planned_stop_watcher_stop = None
    try:
        _start_gateway_configure_logging(verbosity)

        runner = GatewayRunner(resolved_config)
        from gateway.run_runtime import initialize_gateway_runtime
        await initialize_gateway_runtime(runner)
        # Multiplex: swap the launch-home file handlers for per-profile routers so each profile's records
        # land in its own logs/. Must run after the runner resolved (possibly None) config and setup_logging.
        # See #82936.
        _enable_multiplex_log_routing(runner.config)
        # ``--replace`` is explicit startup authority, not a durable reconnect policy: GatewayRunner scopes
        # it to cold adapter connects and clears it before the background reconnect watcher starts.
        runner._platform_lock_takeover_on_start = bool(replace)

        # Unexpected signals exit non-zero so service managers revive us; planned stops write a marker first.
        _signal_initiated_shutdown = [False]

        shutdown_signal_handler = _start_gateway_make_shutdown_signal_handler(
            runner, _signal_initiated_shutdown)

        def restart_signal_handler():
            runner.request_restart(detached=False, via_service=True)

        loop = asyncio.get_running_loop()

        # Swallow transient network errors from background tasks; one unhandled httpx error would kill us.
        # Issues #31066 / #31110: an unhandled ``telegram.error.TimedOut`` (or peer NetworkError / httpx
        # connection error) in any awaited coroutine would propagate to the loop and kill the gateway process,
        # taking down every profile attached to the same runner. systemd then restarts the service after ~5s but
        # the active conversation turn is lost. The fix is intentionally narrow: only well-known transient
        # network errors are swallowed (and logged with full traceback so the originating call site is still
        # discoverable). Anything else is forwarded to the default handler so real bugs still surface.
        loop.set_exception_handler(_gateway_loop_exception_handler)

        if threading.current_thread() is threading.main_thread():
            # add_signal_handler raises NotImplementedError on Windows; SIGUSR1 is POSIX-only.
            handlers = [(sig, shutdown_signal_handler, (sig,)) for sig in (signal.SIGINT, signal.SIGTERM)]
            if hasattr(signal, "SIGUSR1"):
                handlers.append((signal.SIGUSR1, restart_signal_handler, ()))  # windows-footgun: ok — hasattr-guarded
            for sig, handler, args in handlers:
                with suppress(NotImplementedError):
                    loop.add_signal_handler(sig, handler, *args)  # windows-footgun: ok — suppress(NotImplementedError)
        else:
            logger.info("Skipping signal handlers (not running in main thread).")

        # Windows has no add_signal_handler, so `hermes gateway stop`'s SIGTERM would never drain; poll the
        # planned-stop marker (written BEFORE the kill) instead. Runs everywhere so masked-SIGTERM drains.
        # Windows fallback: asyncio.add_signal_handler raises NotImplementedError on Windows, so `hermes gateway
        # stop`'s SIGTERM (which Python maps to TerminateProcess on Windows) never invokes
        # shutdown_signal_handler. That means the drain loop never runs, mark_resume_pending never fires, and
        # sessions are silently lost across restarts (issue #33778). The fix is a marker-polling thread: `hermes
        # gateway stop` writes the planned-stop marker BEFORE killing, and this thread notices it and drives the
        # same shutdown path the signal handler would have. Runs on every platform (cheap, defensive) so
        # non-signal-bearing environments (Windows native, sandboxed CI runners that mask SIGTERM) still get a
        # clean drain.
        _planned_stop_watcher_stop = threading.Event()
        _planned_stop_watcher_thread = threading.Thread(
            target=_run_planned_stop_watcher,
            args=(_planned_stop_watcher_stop, runner, loop, shutdown_signal_handler), daemon=True,
            name="planned-stop-watcher")
        _planned_stop_watcher_thread.start()

        # Right after the PID claim (which makes us authoritative); non-fatal — consumers fall back to scan.
        _control_server = await _start_gateway_start_control_socket(runner)
        if _control_server is None:
            raise RuntimeError("gateway session bootstrap control listener unavailable")
        runner.session_control_server = _control_server
        from gateway.run_runtime import start_gateway_runtime_api
        await start_gateway_runtime_api(runner)

        def _lifecycle_record_startup() -> None:
            # Report if the previous life died uncleanly (SIGKILL / OOM / VM death), then claim the
            # sentinel for this life. After the PID-file claim so a --replace loser can't clobber it.
            from gateway.lifecycle_ledger import record_startup
            record_startup()

        def _start_keepalive() -> None:
            from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive
            start_nous_auth_keepalive()

        _best_effort(_lifecycle_record_startup, "Lifecycle ledger startup record failed: %s")
        _best_effort(_start_keepalive, "Nous auth keepalive did not start: %s")
        _ensure_windows_gateway_venv_imports()

        # discover_mcp_tools() blocks up to 120s; on the loop thread it would freeze platform heartbeats.
        try:
            # MCP tool discovery — run in an executor so the asyncio event loop stays responsive even when a
            # configured MCP server is slow or unreachable.  discover_mcp_tools() uses a blocking 120s wait
            # internally; calling it from the loop thread would freeze platform heartbeats (Discord shard,
            # Telegram polling) until it returned. See #16856.
            await _discover_gateway_mcp_tools(runner.config)
        except Exception as e:
            logger.debug("MCP tool discovery failed: %s", e)

        try:
            success = await runner.start()
        except BaseException:
            _shutdown_gateway_health_export(runner)
            raise
        if not success:
            _shutdown_gateway_health_export(runner)
            return False

        def _recover_pending() -> None:
            from gateway.shutdown_flush import recover_pending_to_db
            recovered = recover_pending_to_db()
            if recovered:
                logger.info("Recovered %d pending message(s) from shutdown flush", recovered)

        _best_effort(_recover_pending)
        if runner.should_exit_cleanly:
            _shutdown_gateway_health_export(runner)
            if runner.exit_reason:
                logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
            # Explicit exit codes (GATEWAY_FATAL_CONFIG_EXIT_CODE) must propagate so s6 finish maps 78 → 125.
            if runner.exit_code is not None:
                raise SystemExit(runner.exit_code)
            return True
        if not runner._running:
            # Startup aborted by restart/shutdown before running mode; preserve that path without starting cron.
            try:
                await runner.wait_for_shutdown()
                with suppress(Exception):
                    await _shutdown_mcp_servers_nonblocking()
                return _resolve_gateway_exit_verdict(runner, _signal_initiated_shutdown[0])
            finally:
                _shutdown_gateway_health_export(runner)

        cron_stop, cron_provider, cron_thread, housekeeping_thread = (
            _start_gateway_start_cron_and_housekeeping(runner))

        # READY only once adapters, cron and housekeeping run; missing systemd state just disables watchdog.
        from gateway.run_runtime import publish_gateway_runtime_ready
        publish_gateway_runtime_ready(runner)
        runner._start_systemd_watchdog()

        from gateway.run_runtime import wait_gateway_runtime
        try:
            await wait_gateway_runtime(runner)
        finally:
            try:
                await runner.stop()
            finally:
                verdict = await _start_gateway_shutdown_tail(
                    runner, _control_server, cron_stop, cron_provider, cron_thread, housekeeping_thread,
                    _planned_stop_watcher_stop, _planned_stop_watcher_thread, _signal_initiated_shutdown)
        return verdict

    finally:
        try:
            if runner is not None:
                from gateway.run_runtime import drain_gateway_runtime
                await drain_gateway_runtime(runner)
                # Startup may have opened adapters before an exception. The same
                # stop path owns their writers and DB handles on every exit.
                await runner.stop()
        finally:
            if _planned_stop_watcher_stop is not None:
                _planned_stop_watcher_stop.set()
            if _control_server is not None:
                await _control_server.stop()
            remove_pid_file()
            release_gateway_runtime_lock()
