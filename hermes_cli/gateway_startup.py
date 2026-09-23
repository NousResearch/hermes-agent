"""Gateway startup, host multiplexer and foreground conflict guards.

Moved from hermes_cli.gateway with original function bodies intact.
Global collaborators resolve through the public facade per call.
"""

def _wait_for_gateway_exit(timeout: float = 10.0, force_after: float | None = 5.0) -> bool:
    """Wait up to ``timeout`` s for the gateway (by gateway.pid, not launchd labels, so multiple
    HERMES_HOMEs work) to exit; SIGKILL it after ``force_after`` s of graceful waiting."""
    from hermes_cli.gateway import (
        terminate_pid,
        time,
    )
    from gateway.status import get_process_start_time, get_running_pid
    deadline = time.monotonic() + timeout
    force_deadline = (time.monotonic() + force_after) if force_after is not None else None
    force_sent = False

    while time.monotonic() < deadline:
        pid = get_running_pid()
        if pid is None:
            return True  # Process exited cleanly.

        if force_after is not None and not force_sent and time.monotonic() >= force_deadline:
            # Grace period expired — force-kill the specific PID.
            try:
                terminate_pid(pid, force=True, expected_start_time=get_process_start_time(pid))
                print(f"⚠ Gateway PID {pid} did not exit gracefully; sent SIGKILL")
            except (ProcessLookupError, PermissionError, OSError):
                return True  # Already gone or we can't touch it.
            force_sent = True

        time.sleep(0.3)

    # Timed out even after force-kill.
    remaining_pid = get_running_pid()
    if remaining_pid is not None:
        print(f"⚠ Gateway PID {remaining_pid} still running after {timeout}s — restart may fail")
        return False
    return True


def _wait_for_tcp_port_free(host: str, port: int, *, timeout: float = 10.0) -> bool:
    """Wait until nothing accepts TCP connections on host:port.

    PID exit is not enough on macOS: api_server disables SO_REUSEADDR, so a restart that wins
    the race logs EADDRINUSE and keeps running with no API. Connection-refused means the
    listener is gone; a timed-out connect is a live listener with a slow accept queue.
    """
    from hermes_cli.gateway import (
        socket,
        time,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                pass
        except ConnectionRefusedError:
            return True
        except TimeoutError:
            pass  # a slow accept queue is still a live listener
        except OSError:
            return True  # unresolvable/unreachable address: nothing to wait for; the bind retry covers it
        time.sleep(0.1)
    return False


def _wait_for_api_server_port_free(*, timeout: float = 10.0) -> bool:
    """Wait for the configured api_server listen address to stop accepting.

    Only when api_server is enabled: with the platform off, a foreign listener on the default
    port is nobody's race and must not delay the restart."""
    from hermes_cli.gateway import (
        _wait_for_tcp_port_free,
        load_gateway_config,
    )
    from gateway.config import Platform
    from gateway.platforms.api_server import listen_address
    pconfig = load_gateway_config().platforms.get(Platform.API_SERVER)
    if pconfig is None or not pconfig.enabled:
        return True
    host, port = listen_address(pconfig.extra or {})
    freed = _wait_for_tcp_port_free(host, port, timeout=timeout)
    if not freed:
        print(
            f"⚠ {host}:{port} still accepting connections — "
            "new api_server may fail to bind"
        )
    return freed



# =============================================================================
# Gateway Runner
# =============================================================================


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_official_docker_checkout() -> bool:
    from hermes_cli.gateway import (
        PROJECT_ROOT,
    )
    return str(PROJECT_ROOT) == "/opt/hermes" and (PROJECT_ROOT / "docker" / "entrypoint.sh").is_file()


def _running_under_gateway_supervisor() -> bool:
    """True when this process IS the supervisor-launched gateway, so the conflict guard never wedges
    the service into a respawn/refuse loop. Markers: systemd INVOCATION_ID, launchd XPC_SERVICE_NAME
    (shells inherit "0"), s6 HERMES_S6_SUPERVISED_CHILD, or ``--external-supervisor``."""
    from hermes_cli.gateway import (
        is_gateway_supervisor_process,
    )
    return is_gateway_supervisor_process()


def host_multiplexer_serving(profile_name: str | None = None):
    """The ONE live host gateway when it serves ``profile_name`` (default: the current profile).

    Companion to :func:`named_profile_served_by_running_multiplexer`, which answers the narrower
    "is this a SATELLITE of the default's multiplexer" and is hard-False for ``default`` — the
    profile that usually owns the shared process. Every lifecycle guard built on it was therefore
    blind to the host process itself. This one is true for ``default`` too, and it reports WHICH
    profiles the host process serves. Returns a ``gateway.host_attach.HostGateway`` or None.
    """
    from hermes_cli.gateway import (
        _current_profile_name,
        logger,
    )
    try:
        from gateway.host_attach import host_gateway_serving
        name = profile_name if profile_name is not None else _current_profile_name()
        return host_gateway_serving(name or "default")
    except Exception:
        logger.debug("Host multiplexer probe failed", exc_info=True)
        return None


def _served_by_another_host_gateway(profile_name: str | None = None):
    """The host gateway serving ``profile_name`` when it is NOT this home's own process.

    The owner must never be guarded out of restarting itself: a refusal keyed on "something serves
    you" would make `hermes gateway restart` impossible for the profile that launched the host
    process. Guards want "ANOTHER process already serves you", which is this.
    """
    from hermes_cli.gateway import (
        host_multiplexer_serving,
        logger,
    )
    gateway = host_multiplexer_serving(profile_name)
    if gateway is None:
        return None
    try:
        from gateway.status import _get_process_hermes_home, _same_hermes_home
        if _same_hermes_home(gateway.home, _get_process_hermes_home()):
            return None
    except Exception:
        logger.debug("Host multiplexer home comparison failed", exc_info=True)
    return gateway


def named_profile_served_by_running_multiplexer(profile_name: str | None = None) -> bool:
    """True when a live default multiplexer already ticks this named profile (a satellite profile has no
    gateway.pid; the multiplexer fires its jobs and serves its platforms). Defaults to the current profile.

    See #97120.
    """
    from hermes_cli.gateway import (
        _current_profile_name,
        host_multiplexer_serving,
        logger,
    )
    try:
        suffix = profile_name if profile_name is not None else _current_profile_name()
    except Exception:
        return False
    if not suffix or suffix == "default":
        return False

    # The host record answers first: it names the live host process whatever home launched it, so a
    # multiplexer started by a named profile is visible here too.
    if host_multiplexer_serving(suffix) is not None:
        return True

    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
    except Exception:
        return False

    try:
        from hermes_cli.gateway_multiplex_served import live_default_gateway_pid, recorded_served_profiles
        if live_default_gateway_pid() is None:
            return False
        from hermes_cli.profiles import normalize_profile_name
        # The live gateway's own record wins: the CLI process cannot see an env-only opt-in on the
        # default profile (`hermes -p X` loads X's .env) and a config edit after start is not live yet.
        # Only a record without the key (pre-multiplex writer) falls through to config derivation.
        recorded = recorded_served_profiles(default_root)
        if recorded is not None:
            return normalize_profile_name(suffix) in {normalize_profile_name(p) for p in recorded}

        # No record (older gateway): only an EXPLICIT opt-in counts. The unset default is settled by
        # the gateway at boot (it may have stayed standalone); a CLI process must not guess it on.
        from hermes_cli.gateway_multiplex_mode import explicit_multiplex_flag
        return explicit_multiplex_flag(default_root) is True  # a multiplexer serves every named profile
    except Exception:
        logger.debug("Multiplexer-serving probe failed", exc_info=True)
        return False


def _served_profile_needs_no_service() -> bool:
    """Print the "already served" note and return True when a setup flow must not install a standalone
    service: a live multiplexing default gateway already serves this named profile, so the unit/plist it
    would register can only sit dead (the start guard refuses it) or double-bind its platforms.
    Shared by ``hermes setup gateway`` / ``hermes setup`` / ``hermes import`` (``ensure_gateway_service``)
    and the ``hermes gateway setup`` wizard. See #111958."""
    from hermes_cli.gateway import (
        _current_profile_name,
        _named_profile_refused_under_multiplexer,
        get_hermes_home,
        named_profile_served_by_running_multiplexer,
        print_info,
        print_success,
    )
    if not named_profile_served_by_running_multiplexer():
        # Not served (yet): a named profile still gets no service of its own — same rule and text
        # as `gateway install`, so `hermes -p X setup` cannot grow a fleet member the verb refuses.
        return _named_profile_refused_under_multiplexer()
    from hermes_cli.profiles import profile_is_standalone
    if profile_is_standalone(get_hermes_home()):
        from gateway.host_attach import standalone_rescan_message
        print_info(standalone_rescan_message(_current_profile_name()))
        return True
    print_success(
        f"Profile '{_current_profile_name()}' is already served by the default multiplexer."
    )
    print_info("  (served now by the running multiplexed gateway — add its bot token and it connects)")
    print_info("  No standalone gateway service was installed or started.")
    return True


def _named_profile_refused_under_multiplexer(force: bool = False) -> bool:
    """Print the refusal and return True when a NAMED profile must not get a gateway of its own.

    One gateway per host serves every profile, so a ``<root>/profiles/<name>`` home never installs or
    starts a standalone gateway: either the host gateway already serves it (a second one would
    double-bind its platforms: two pollers on one token, port fights) or no host gateway runs yet and
    the DEFAULT profile is where it is installed. Refusing only the served case let a host with no
    multiplexer running (or one that had not rescanned yet) grow a brand-new per-profile fleet member.
    ``--force`` is the one escape (a fleet split across UNIX users or a ``HERMES_HOME`` outside
    ``profiles/``); a service it already installed stays startable without it. Shared by ``run`` and the service verbs (``start``/``install``/``restart``): a
    refusal only inside ``gateway run`` leaves the service manager to discover it — systemd parks the
    unit on exit 78 while the CLI prints "started"; launchd (KeepAlive, no exit-status gating)
    respawns it every ThrottleInterval forever."""
    from hermes_cli.gateway import (
        _current_profile_name,
        _is_service_installed,
        _served_by_another_host_gateway,
        get_hermes_home,
        named_profile_served_by_running_multiplexer,
        print_error,
    )
    if force:
        return False
    try:
        suffix = _current_profile_name()
        from hermes_constants import profile_name_for_home
        from hermes_cli.profiles import profile_is_standalone
        # A profile that authored gateway.standalone: true opted out of the host multiplexer: it is
        # allowed a gateway of its own without --force. Only a RUNNING host record that still lists
        # it (the host has not rescanned since the key was set) is refused with the rescan remedy.
        standalone = (profile_name_for_home(get_hermes_home()) not in (None, "default")
                      and profile_is_standalone(get_hermes_home()))
        # A unit/plist/task already registered for this home was installed with --force: that fleet
        # member (and the supervisor relaunching it, whose ExecStart carries no --force) is not NEW.
        new_standalone = (profile_name_for_home(get_hermes_home()) not in (None, "default")
                          and not _is_service_installed())
    except Exception:
        return False
    owner = _served_by_another_host_gateway()
    served = owner is not None or named_profile_served_by_running_multiplexer()
    if standalone:
        if not served:
            return False
        from gateway.host_attach import standalone_rescan_message
        print_error(standalone_rescan_message(suffix))
        return True
    if not served and not new_standalone:
        return False

    if served:
        print_error(f"The host gateway already serves profile '{suffix}'.")
        if owner is not None:
            print(f"  {owner.describe()}")
    else:
        print_error(f"Profile '{suffix}' does not get a gateway of its own.")
    print(
        "  Exactly one gateway per host is the inbound process for every\n"
        "  profile. Starting a separate gateway for this profile would\n"
        "  double-bind its platforms (two pollers on one bot token, port\n"
        "  conflicts).\n"
    )
    if served:
        print("  Manage the host gateway instead:")
        print()
        print(f"    hermes -p {owner.profile_label if owner is not None else 'default'} gateway restart")
    else:
        print("  Install or start the host gateway from the default profile; it serves this one too:")
        print()
        print("    hermes gateway install")
        print()
        print("  Or fold an existing per-profile fleet onto one host gateway:")
        print()
        print("    hermes gateway migrate --multiplex")
    print()
    print("  A separate per-profile gateway (for a fleet split across UNIX users or a")
    print(f"  HERMES_HOME outside profiles/) needs --force:  hermes -p {suffix} gateway install --force")
    print()
    from hermes_constants import display_hermes_home
    from hermes_cli.gateway_multiplex_mode import STANDALONE_DEPRECATION_NOTICE
    print("  Temporary compatibility path while multiplexing gaps are closed: set")
    print(f"  gateway.standalone: true in {display_hermes_home(get_hermes_home())}/config.yaml,")
    print("  then wait for the host gateway to rescan (<=30s) or send its rescan-profiles control verb.")
    print(f"  ({STANDALONE_DEPRECATION_NOTICE})")
    return True


def _guard_named_profile_under_multiplexer(force: bool = False) -> None:
    """Exit-78 form of ``_named_profile_refused_under_multiplexer`` for the CLI entry points."""
    from hermes_cli.gateway import (
        GATEWAY_FATAL_CONFIG_EXIT_CODE,
        _named_profile_refused_under_multiplexer,
        sys,
    )
    if not _named_profile_refused_under_multiplexer(force=force):
        return
    # EX_CONFIG, not 1: the refusal is decided purely by config, so it is permanent. The systemd unit
    # (Restart=always, StartLimitIntervalSec=0) relies on RestartPreventExitStatus=78 as its only
    # backstop — exit 1 turned a correct refusal into an unbounded restart loop; s6 maps 78 to
    # "permanent failure" too.
    # This refusal is decided entirely by configuration (multiplex_profiles plus the allowlist), so it is
    # permanent: no number of retries can change the answer. Exiting 1 made it look transient to a service
    # manager -- and the systemd unit this module generates pairs Restart=always/RestartSec=5 with
    # StartLimitIntervalSec=0, deliberately trading systemd's generic start-rate limiter for the specific
    # RestartPreventExitStatus=GATEWAY_FATAL_CONFIG_EXIT_CODE backstop declared beside it. Returning 1 left
    # that backstop unarmed with the limiter already off, so a correct refusal became an unbounded restart
    # loop. 78 also reaches the s6 finish script's 125 "permanent failure" translation (see #51228), the
    # same path the other fatal-config exits take.
    sys.exit(GATEWAY_FATAL_CONFIG_EXIT_CODE)


def _host_decision_exit_code(decision) -> int:
    """Exit code for a host-attach verdict a supervisor may be watching.

    ``GATEWAY_FATAL_CONFIG_EXIT_CODE`` (78) is the PERMANENT refusal: systemd parks the unit on it
    (``RestartPreventExitStatus``), the s6 finish script maps it to 125, launchd maps it to a
    deliberate stop. That is right for a config-derived refusal and wrong for a runtime one — "some
    other process serves me right now" ends the moment that process goes away, and parking the unit
    on it strands the profile until a human notices. Transient verdicts therefore use
    ``GATEWAY_SERVICE_RESTART_EXIT_CODE`` (75, EX_TEMPFAIL), which every supervisor we generate
    already retries: systemd has ``RestartForceExitStatus=75`` with ``RestartSec=5``, the s6 finish
    script passes it through, and launchd relaunches a non-78 failure. Exit 0 would NOT do: s6
    parks a clean exit too.
    """
    from hermes_cli.gateway import (
        GATEWAY_FATAL_CONFIG_EXIT_CODE,
        GATEWAY_SERVICE_RESTART_EXIT_CODE,
    )
    if getattr(decision, "transient", False):
        return GATEWAY_SERVICE_RESTART_EXIT_CODE
    return GATEWAY_FATAL_CONFIG_EXIT_CODE


def _attach_to_host_gateway_or_guard(force: bool = False, replace: bool = False) -> None:
    """``gateway run`` against the ONE host gateway: attach, rescan-then-attach, replace, or refuse.

    A profile the host process already serves has nothing to run: print who serves it and exit 0
    without spawning anything. Under a service supervisor the SAME situation exits 75 instead, so
    the unit is RETRIED rather than parked (see :func:`_host_decision_exit_code`).

    ``--replace`` and ``--force`` are the two escape hatches this guard must not eat: both return
    here so ``start_gateway`` can act on them (it owns the signalling and the PID claim).
    """
    from hermes_cli.gateway import (
        _guard_named_profile_under_multiplexer,
        _host_decision_exit_code,
        _running_under_gateway_supervisor,
        get_hermes_home,
        logger,
        sys,
    )
    if force:
        return
    try:
        from gateway.host_attach import ATTACH, REFUSE, REPLACE_HOST, decide
        decision = decide(get_hermes_home(), replace=replace)
    except Exception:
        logger.debug("Host gateway attach probe failed", exc_info=True)
        decision = None
    if decision is not None and decision.outcome == REPLACE_HOST:
        return  # start_gateway replaces the owner; the config guard below must not pre-empt it
    if decision is not None and decision.outcome in (ATTACH, REFUSE):
        print(decision.message)
        if decision.outcome == REFUSE:
            code = _host_decision_exit_code(decision)
            # stdout goes to the supervisor's unit log; under launchd a permanent refusal is then
            # mapped to a clean exit and the unit is parked. The profile's own logs (errors.log,
            # WARNING+) are where a parked fleet is diagnosed, so name the verdict and the remedy there.
            logger.warning("gateway run refused (exit %d): %s", code, decision.message)
            sys.exit(code)
        if _running_under_gateway_supervisor():
            sys.exit(_host_decision_exit_code(decision))
        sys.exit(0)
    # No host record (older gateway, unwritable lock dir): the config-derived refusal still applies.
    _guard_named_profile_under_multiplexer(force=force)


def _guard_supervised_gateway_conflict(force: bool = False) -> None:
    """Refuse a foreground gateway when a service manager already supervises one: a shell-launched run
    becomes a second dispatcher that escapes the cgroup, survives ``systemctl restart``, and writes the
    shared kanban DB concurrently (multi-writer SQLite WAL corruption). ``--force`` starts anyway.

    See #35240.
    """
    from hermes_cli.gateway import (
        _running_under_gateway_supervisor,
        get_gateway_runtime_snapshot,
        logger,
        print_error,
        sys,
    )
    if force or _running_under_gateway_supervisor():
        return
    try:
        snapshot = get_gateway_runtime_snapshot()
    except Exception:
        logger.debug("Supervised-gateway conflict probe failed", exc_info=True)
        return
    if not (snapshot.service_installed and snapshot.service_running):
        return

    print_error(f"A gateway is already running under {snapshot.manager} for this profile.")
    print(
        "  Starting another one from a shell leaves an orphan dispatcher that\n"
        "  escapes the service, survives restarts, and writes to the same kanban\n"
        "  DB concurrently — which can corrupt it. Restart the supervised gateway\n"
        "  instead:"
    )
    print()
    print("    hermes gateway restart")
    print()
    print(
        "  Pass --force to start a foreground gateway anyway (not recommended\n"
        "  while the service is running)."
    )
    sys.exit(1)


def _guard_existing_gateway_process_conflict(replace: bool = False) -> None:
    """Cheap PID-file preflight before the expensive ``gateway.run`` import (the authoritative lock check):
    supervisor loops re-running bare ``gateway run`` burned memory on plugin discovery just to fail
    "already running". Same user-facing contract; never scans other HERMES_HOME roots."""
    from hermes_cli.gateway import (
        _running_under_gateway_supervisor,
        logger,
        print_error,
        sys,
    )
    if replace or _running_under_gateway_supervisor():
        return
    try:
        from gateway.status import get_running_pid
        pid = get_running_pid()
    except Exception:
        logger.debug("Existing-gateway process probe failed", exc_info=True)
        return
    if pid is None:
        # get_running_pid() filters by the current profile's HERMES_HOME; warn if the PID file
        # belongs to another profile (user switched profiles while the old gateway still runs).
        try:
            from gateway.status import _read_pid_record, _pid_record_belongs_to_current_profile
            stale = _read_pid_record()
            if stale is not None and not _pid_record_belongs_to_current_profile(stale):
                logger.warning(
                    "PID file belongs to another profile (hermes_home=%s). "
                    "The old gateway may still be running under that profile.",
                    stale.get("hermes_home", "<unknown>"),
                )
        except Exception:
            pass
        return

    print_error(f"A gateway is already running (PID {pid}), so your bots are most likely online already.")
    print("  Check with `hermes gateway status`.")
    print("  To restart it: `hermes gateway restart`. To stop it: `hermes gateway stop`.")
    print("  To replace it from here: `hermes gateway run --replace`.")
    sys.exit(1)


def _guard_official_docker_root_gateway() -> None:
    """Refuse gateway startup when the official Docker privilege drop was bypassed."""
    from hermes_cli.gateway import (
        _is_official_docker_checkout,
        _truthy_env,
        os,
        print_error,
        sys,
    )
    if not hasattr(os, "geteuid") or os.geteuid() != 0 or _truthy_env(os.getenv("HERMES_ALLOW_ROOT_GATEWAY")):
        return
    if not _is_official_docker_checkout():
        return

    print_error("Refusing to run the Hermes gateway as root inside the official Docker image.")
    print(
        "  The image entrypoint normally drops privileges to the 'hermes' user. "
        "If you override entrypoint in Docker Compose, include "
        "/opt/hermes/docker/entrypoint.sh before the Hermes command."
    )
    print(
        "  Running the gateway as root can leave root-owned files in "
        "$HERMES_HOME and break later non-root dashboard/gateway runs."
    )
    print("  Set HERMES_ALLOW_ROOT_GATEWAY=1 only if you intentionally accept this risk.")
    sys.exit(1)


def _apply_startup_watchdog_config() -> None:
    """Idempotent backstop arming of the startup-liveness watchdog. Must run AFTER the conflict guards (a
    --replace loser must not arm one). config.yaml gateway.startup_watchdog* is the user surface; env
    vars bridge it because the argv fast-path arms before config loads, and explicit env wins. arm() is
    idempotent, so a config timeout needs disarm+re-arm. GatewayRunner disarms once the loop is live."""
    from hermes_cli.gateway import (
        os,
    )
    try:
        from hermes_startup_watchdog import (
            ENV_STARTUP_WATCHDOG, ENV_STARTUP_WATCHDOG_TIMEOUT_S, arm_startup_watchdog,
            disarm_startup_watchdog, startup_watchdog_disabled,
        )
        _sw_timeout_bridged = False
        try:
            from hermes_cli.config import load_config as _sw_load_config
            _gw_cfg = (_sw_load_config() or {}).get("gateway", {}) or {}
            if ENV_STARTUP_WATCHDOG not in os.environ and not _gw_cfg.get("startup_watchdog", True):
                os.environ[ENV_STARTUP_WATCHDOG] = "0"
            _sw_timeout = _gw_cfg.get("startup_watchdog_timeout_seconds")
            if ENV_STARTUP_WATCHDOG_TIMEOUT_S not in os.environ and _sw_timeout is not None:
                os.environ[ENV_STARTUP_WATCHDOG_TIMEOUT_S] = str(_sw_timeout)
                _sw_timeout_bridged = True
        except Exception:
            pass
        if startup_watchdog_disabled():
            disarm_startup_watchdog()
        else:
            if _sw_timeout_bridged:
                disarm_startup_watchdog()
            arm_startup_watchdog()
    except Exception:
        pass


def _absorb_windows_console_controls() -> None:
    """Make a detached Windows gateway ignore console-control broadcasts from sibling CLIs."""
    from hermes_cli.gateway import (
        signal,
    )
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except (OSError, ValueError):
        pass  # SetConsoleCtrlHandler unavailable (rare) — best-effort
    # signal only hooks SIGINT/SIGBREAK; SetConsoleCtrlHandler(NULL, TRUE) ignores ALL console
    # control events (CTRL_CLOSE/CTRL_LOGOFF included), as background services should.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, 1)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        pass


def _make_exit_diag():
    """``_exit_diag(tag, **extra)`` recorder writing ``logs/gateway-exit-diag.log`` — captures every way
    ``asyncio.run()`` can return, for chasing silent Windows gateway deaths. HERMES_GATEWAY_EXIT_DIAG=0 opts out."""
    from hermes_cli.gateway import (
        json,
        os,
        sys,
    )
    from datetime import datetime as _dt, timezone as _tz

    def _exit_diag(tag: str, **extra: object) -> None:
        if os.environ.get("HERMES_GATEWAY_EXIT_DIAG", "1") != "1":
            return
        try:
            from hermes_constants import get_hermes_home as _ghh
            log_dir = _ghh() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            line = {
                "ts": _dt.now(_tz.utc).isoformat(), "tag": tag, "pid": os.getpid(),
                "python": sys.version.split()[0], "platform": sys.platform, **extra,
            }
            with open(log_dir / "gateway-exit-diag.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
        except Exception:
            pass  # never let the diagnostic itself crash the gateway

    return _exit_diag


def _respawn_storm_backoff() -> None:
    """Portable app-level respawn-storm breaker (for supervisors without a floor). Defaults mirror
    DEFAULT_CONFIG ``gateway.respawn_storm``; HERMES_GATEWAY_MAX_STARTS / HERMES_GATEWAY_START_WINDOW_S
    override; max_starts <= 0 disables. Never blocks startup."""
    from hermes_cli.gateway import (
        logger,
        os,
        time,
    )
    try:
        from gateway.status import record_start_and_check_storm
        _max_starts = 5
        _win = 120.0
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            _gw = _cfg.get("gateway") if isinstance(_cfg, dict) else None
            _rs = _gw.get("respawn_storm") if isinstance(_gw, dict) else None
            if isinstance(_rs, dict):
                if isinstance(_rs.get("max_starts"), int):
                    _max_starts = _rs["max_starts"]
                if isinstance(_rs.get("window_seconds"), (int, float)):
                    _win = float(_rs["window_seconds"])
        except Exception:
            pass
        try:
            _max_starts = int(os.environ["HERMES_GATEWAY_MAX_STARTS"])
        except (KeyError, ValueError):
            pass
        try:
            _win = float(os.environ["HERMES_GATEWAY_START_WINDOW_S"])
        except (KeyError, ValueError):
            pass
        _storm = record_start_and_check_storm(max_starts=_max_starts, window_s=_win) if _max_starts > 0 else None
        if _storm is not None:
            logger.warning(
                "Gateway (re)started %d times in %.0fs — backing off %.0fs to break a respawn storm.",
                _storm.count, _storm.window_s, _storm.backoff_s,
            )
            # Tell the startup watchdog the backoff sleep is intentional, not a parked deadlock.
            try:
                from hermes_startup_watchdog import kick_startup_watchdog
                kick_startup_watchdog(extra_s=_storm.backoff_s)
            except Exception:
                pass
            time.sleep(_storm.backoff_s)
    except Exception as _be:
        logger.debug("respawn-storm breaker check failed (non-fatal): %s", _be)


def run_gateway(verbose: int = 0, quiet: bool = False, replace: bool = False, force: bool = False):
    """Run the gateway in foreground. verbose 1=INFO/2+=DEBUG on stderr; quiet: no stderr logs; replace:
    kill an existing instance first (avoids systemd restart loops); force: skip the supervised guard."""
    from hermes_cli.gateway import (
        PROJECT_ROOT,
        _absorb_windows_console_controls,
        _apply_startup_watchdog_config,
        _attach_to_host_gateway_or_guard,
        _ensure_user_systemd_env,
        _gateway_detached_env,
        _guard_existing_gateway_process_conflict,
        _guard_official_docker_root_gateway,
        _guard_supervised_gateway_conflict,
        _make_exit_diag,
        _respawn_storm_backoff,
        _stdin_is_tty,
        _windows_console_window_attached,
        _windows_gateway_breakaway_state,
        _windows_gateway_should_absorb_console_controls,
        asyncio,
        is_linux,
        os,
        refresh_systemd_unit_if_needed,
        supports_systemd_services,
        sys,
    )
    _guard_official_docker_root_gateway()
    _attach_to_host_gateway_or_guard(force=force, replace=replace)
    _guard_supervised_gateway_conflict(force=force)
    _guard_existing_gateway_process_conflict(replace=replace)
    sys.path.insert(0, str(PROJECT_ROOT))
    _apply_startup_watchdog_config()

    # Detached Windows runs (HERMES_GATEWAY_DETACHED=1, or non-TTY for older wrappers) ignore
    # console-control broadcasts from sibling CLIs; foreground runs keep Ctrl+C-to-stop.
    stdin_is_tty = bool(_stdin_is_tty())
    _console_window_attached = _windows_console_window_attached()
    _breakaway = _windows_gateway_breakaway_state()
    _absorb = _windows_gateway_should_absorb_console_controls()
    if _absorb:
        _absorb_windows_console_controls()

    # A system-level unit execs us without XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS; adopt our own
    # user bus before any worker env snapshot so `systemd-run --user --scope` works (#104893).
    if is_linux() and os.environ.get("INVOCATION_ID"):
        _ensure_user_systemd_env()

    # Refresh the systemd unit on every boot so restart settings stay current even after an
    # exit-code-75 respawn (stale-code or /restart), which bypasses `hermes gateway restart`.
    if supports_systemd_services():
        try:
            refresh_systemd_unit_if_needed(system=False)
        except Exception:
            pass  # best-effort; don't block gateway startup

    from gateway.run import start_gateway
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ☤ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    # Exit 1 if no platform connects so systemd Restart=always retries transient errors.
    verbosity = None if quiet else verbose

    import atexit as _atexit
    import traceback as _traceback
    _exit_diag = _make_exit_diag()
    _exit_diag(
        "gateway.start", replace=replace, argv=sys.argv, stdin_is_tty=stdin_is_tty,
        console_window_attached=_console_window_attached, detached=_gateway_detached_env(),
        breakaway=_breakaway, absorb_windows_console_controls=_absorb,
    )
    _atexit.register(lambda: _exit_diag("atexit.hook", sys_exc=repr(sys.exc_info())))

    _respawn_storm_backoff()

    def _hard_exit_after_gateway_teardown(code: int) -> None:
        # Mirror gateway.run.main()'s wedge-proof exit: bypass Python finalization so non-daemon
        # threads (in-flight cron jobs) can't delay a /restart by minutes.
        from gateway.run import _exit_after_graceful_shutdown
        _exit_after_graceful_shutdown(code)

    success = False
    try:
        success = asyncio.run(start_gateway(replace=replace, force=force, verbosity=verbosity))
        _exit_diag("asyncio.run.returned", success=success)
    except KeyboardInterrupt:
        # Detached Windows runs absorb SIGINT above; keep the handler for console runs.
        _exit_diag("asyncio.run.KeyboardInterrupt", traceback=_traceback.format_exc())
        print("\nGateway stopped.")
        _hard_exit_after_gateway_teardown(0)
        return  # unreachable in production (os._exit); guard for test stubs
    except SystemExit as e:
        _exit_diag("asyncio.run.SystemExit", code=e.code, traceback=_traceback.format_exc())
        _hard_exit_after_gateway_teardown(0 if e.code is None else e.code if isinstance(e.code, int) else 1)
    except BaseException as e:
        # Everything else (CancelledError, exotic BaseExceptions): log the cause, then re-raise.
        _exit_diag("asyncio.run.exception", exc_type=type(e).__name__, exc_repr=repr(e), traceback=_traceback.format_exc())
        raise
    if not success:
        _exit_diag("gateway.exit_nonzero")
        _hard_exit_after_gateway_teardown(1)
    _exit_diag("gateway.exit_clean")
    _hard_exit_after_gateway_teardown(0)
