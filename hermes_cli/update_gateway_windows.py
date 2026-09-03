"""Windows gateway pause, resume, and cold-start coordination.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""



def _stop_windows_gateway_service(
    name: str,
    *,
    expected_processes: tuple[tuple[int, float], ...] = (),
    expected_service_identity: tuple[int, float] | None = None,
    expected_gateway_identity: tuple[int, float] | None = None,
    timeout: float = 30.0,
) -> None:
    """Stop one verified Windows service and wait until SCM reports it down."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    if expected_service_identity is not None:
        try:
            current_status = str(service.status())
            current_service_pid = int(service.pid() or 0)
        except Exception as exc:
            raise RuntimeError(
                f"Windows service {name} SCM identity is unavailable before stop"
            ) from exc
        if current_status != "running":
            raise RuntimeError(
                f"Windows service {name} is not stably running before stop: {current_status}"
            )
        if current_service_pid != int(expected_service_identity[0]):
            raise RuntimeError(
                f"Windows service {name} SCM process identity changed before stop"
            )
    for label, identity in (
        ("service", expected_service_identity),
        ("gateway", expected_gateway_identity),
    ):
        if identity is None:
            continue
        pid, create_time = identity
        try:
            current = float(psutil.Process(int(pid)).create_time())
        except Exception as exc:
            raise RuntimeError(
                f"Windows {label} process identity is unavailable before stop"
            ) from exc
        if abs(current - float(create_time)) > 0.001:
            raise RuntimeError(
                f"Windows {label} process identity changed before stop"
            )
    if expected_service_identity is not None and expected_gateway_identity is not None:
        service_pid = int(expected_service_identity[0])
        gateway_pid = int(expected_gateway_identity[0])
        try:
            ancestor_pids = {
                int(parent.pid) for parent in psutil.Process(gateway_pid).parents()
            }
        except Exception as exc:
            raise RuntimeError(
                "Windows gateway ancestry is unavailable before service stop"
            ) from exc
        if service_pid not in ancestor_pids:
            raise RuntimeError(
                f"Windows gateway is no longer owned by service {name}"
            )
    result = subprocess.run(
        ["sc.exe", "stop", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "stopped":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe stop failed with {result.returncode}")

    def _original_process_is_alive(pid: int, create_time: float) -> bool:
        try:
            current = float(psutil.Process(pid).create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # A vanished process is clear.
            return False
        except Exception:
            # AccessDenied or any unknown probe failure stays fail-closed
            # because the venv may still be locked.
            return True
        return abs(current - create_time) <= 0.001

    alive = [
        pid
        for pid, create_time in expected_processes
        if _original_process_is_alive(pid, create_time)
    ]
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        service_stopped = service.status() == "stopped"
        alive = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if service_stopped and not alive:
            return
        _time.sleep(0.2)
    if service.status() == "stopped":
        # We only return if the original processes have also exited their identity.
        # A lingering process with a matching creation time means the venv mutation
        # must not proceed — fail closed.
        alive_after_stop = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if alive_after_stop:
            raise RuntimeError(
                f"Windows service {name} stopped but its process tree is still alive: "
                f"{alive_after_stop}"
            )
        return
    # If we reach here, the timeout elapsed without the service reaching a stable stopped state
    # while its original descendants are still alive. Fail closed — venv mutation is unsafe.
    raise RuntimeError(
        f"Windows service {name} did not stop within {timeout:.0f}s; venv mutation unsafe."
    )


def _start_windows_gateway_service(name: str, *, timeout: float = 30.0) -> None:
    """Start one previously paused Windows service and verify it is running."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    result = subprocess.run(
        ["sc.exe", "start", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "running":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe start failed with {result.returncode}")
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if service.status() == "running":
            return
        _time.sleep(0.2)
    raise RuntimeError(f"Windows service {name} did not start within {timeout:.0f}s")


def _restore_windows_gateway_service(name: str, *, timeout: float = 60.0) -> None:
    """Restore a service after an uncertain stop, including STOP_PENDING."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        status = service.status()
        if status == "running":
            return
        if status == "stopped":
            _start_windows_gateway_service(name)
            return
        _time.sleep(0.2)
    raise RuntimeError(
        f"Windows service {name} did not reach a restorable state within {timeout:.0f}s"
    )


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
    """
    if not _m()._is_windows():
        return None

    try:
        from gateway.status import get_process_start_time, terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
            find_windows_gateway_services,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare Windows gateway pause for update: {exc}"
        ) from exc

    try:
        profile_process_list = find_profile_gateway_processes(strict=True)
        profile_processes = {proc.pid: proc for proc in profile_process_list}
    except Exception as exc:
        raise RuntimeError(
            f"Could not map Windows gateway PIDs to profiles: {exc}"
        ) from exc

    try:
        service_gateways = find_windows_gateway_services(
            profile_processes=profile_process_list
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not determine Windows gateway service ownership: {exc}"
        ) from exc

    service_gateway_pids = {int(service.gateway_pid) for service in service_gateways}
    try:
        running_pids = list(
            dict.fromkeys(
                [
                    *find_gateway_pids(all_profiles=True),
                    *sorted(profile_processes),
                    *sorted(service_gateway_pids),
                ]
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover Windows gateway PIDs before update: {exc}"
        ) from exc
    if not running_pids:
        # No gateway is running right now, but the user may have installed an
        # autostart entry (Scheduled Task or Startup-folder login item) — that
        # is an explicit "I want a gateway" signal. A gateway that died between
        # updates (e.g. the spawning terminal/TUI closed, taking its child with
        # it) would otherwise never come back: the autostart entry only fires on
        # the next login, and the update flow's resume path only relaunched
        # gateways that were running when the update began. Cold-start one after
        # the update so an installed gateway is actually up post-update. Users
        # who run gateway-less (no autostart entry) get nothing forced on them.
        #
        # Exception: Desktop currently owns this install's gateway lifecycle
        # (live supervised serve/dashboard). A vestigial Startup/Scheduled
        # Task is not the owner — spawning ``gateway run`` beside Desktop
        # races ports/state (#76129). Serve is the control plane, not proof
        # messaging is served; the skip is ownership, not liveness (#92091).
        try:
            if _desktop_owns_gateway_lifecycle():
                logger.debug(
                    "Skipping Windows gateway cold-start plan: "
                    "Desktop owns gateway lifecycle"
                )
                return None
        except Exception as exc:
            logger.debug(
                "Could not check Desktop gateway-lifecycle ownership before update: %s",
                exc,
            )
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
        return None

    profiles: dict[str, int] = {}
    mapped_pids = []
    socket_acks: list[dict] = []
    for pid in running_pids:
        if pid in service_gateway_pids:
            continue
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))
        # Socket-first pause (#92091 step 2): ask the gateway to drain and
        # exit itself instead of relying on the marker poll + force-kill
        # ladder. A positive ACK means the gateway is running its own
        # graceful restart path (same drain as SIGUSR1/service restarts) and
        # will release its venv handles on the way out. No answer (older
        # gateway, no socket) → the marker watcher / force-kill fallback
        # below behaves exactly as before this verb existed.
        try:
            from gateway.control_socket import pause_gateway_for_update

            ack = pause_gateway_for_update(Path(proc.path))
            if ack and (ack.get("pausing") or ack.get("already_stopping")):
                socket_acks.append(ack)
        except Exception as exc:
            logger.debug(
                "Socket pause unavailable for gateway %s: %s", pid, exc
            )

    # Resolve each mapped worker's venv-side launcher BEFORE draining: the
    # drain stops tracking a PID exactly when it dies, so a gracefully
    # drained worker is gone by the time the wait returns — and a dead pid's
    # parent cannot be recovered (psutil raises NoSuchProcess). The snapshot
    # is stopped after the drain alongside the survivors.
    #
    # Why launchers matter: the drain targets the PID that wrote the PID
    # file (the uv-side worker). On Windows that worker's parent is usually
    # the venv-side ``python.exe`` launcher, which keeps venv ``.pyd`` files
    # mapped and is what ``_detect_venv_python_processes()`` reports
    # downstream. Left alive, it trips the venv-holder guard and aborts the
    # update even though the gateway itself is stopped.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    if socket_acks:
        # A socket-paused gateway drains its ACTIVE TURN before exiting; give
        # it the budget it declared (plus teardown grace) rather than only
        # the local default, so a mid-turn gateway isn't force-killed at the
        # end of a too-short wait — the exact outcome the verb exists to
        # prevent.
        try:
            declared = max(
                float(a.get("drain_timeout") or 0.0) for a in socket_acks
            )
            drain_timeout = max(drain_timeout, declared + 10.0)
        except Exception:
            pass
        print(
            f"  → {len(socket_acks)} gateway(s) ACKed socket pause; "
            f"waiting up to {int(drain_timeout)}s for graceful exit"
        )
    survivors = _m()._wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [
        pid
        for pid in running_pids
        if pid not in profile_processes and pid not in service_gateway_pids
    ]

    # Snapshot each unmapped gateway's command line *before* we force-kill it,
    # so ``_resume_windows_gateways_after_update`` can respawn it by replaying
    # its own argv. Unmapped gateways are ones with no profile→PID-file mapping
    # — e.g. a Windows Scheduled Task running ``pythonw.exe -m hermes_cli.main
    # gateway run``. Without this snapshot they were force-killed and never
    # restarted (the "Restart manually after update" dead-end from #50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Stop drain survivors, unmapped gateways, and the pre-drain launcher
    # snapshot. ``terminate_pid(force=True)`` is a tree kill, so a launcher
    # that outlived its worker takes any stragglers with it; a launcher that
    # already exited with its drained worker raises ProcessLookupError below
    # and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        try:
            pid_int = int(pid)
            terminate_pid(
                pid_int,
                force=True,
                expected_start_time=get_process_start_time(pid_int),
            )
            force_killed.append(pid_int)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    token = {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

    # Stop SCM-supervised gateways only after every fallible preparation step
    # for ordinary gateways is complete. From this point to return, any error
    # restores both the attempted services and the already-paused ordinary
    # gateways before aborting the update.
    paused_services = []
    current_service_name = None
    try:
        for service in service_gateways:
            current_service_name = str(service.name)
            _stop_windows_gateway_service(
                current_service_name,
                expected_processes=tuple(
                    getattr(service, "descendant_identities", ())
                ),
                expected_service_identity=(
                    int(service.service_pid),
                    float(service.service_create_time),
                ),
                expected_gateway_identity=(
                    int(service.gateway_pid),
                    float(service.gateway_create_time),
                ),
            )
            paused_services.append(current_service_name)
            current_service_name = None
        if paused_services:
            token["services"] = paused_services
            token["expected_services"] = list(paused_services)
            token["restarted_services"] = []
            token["service_profiles"] = {
                str(service.name): str(service.profile)
                for service in service_gateways
                if str(service.name) in paused_services
            }
            print(
                "  ✓ Paused Windows gateway service(s): "
                + ", ".join(paused_services)
            )
        return token
    except Exception as exc:
        restore_names = []
        if current_service_name:
            restore_names.append(current_service_name)
        restore_names.extend(reversed(paused_services))
        rollback_failures = []
        for service_name in dict.fromkeys(restore_names):
            try:
                _restore_windows_gateway_service(service_name)
            except Exception as restore_exc:
                rollback_failures.append(f"{service_name}: {restore_exc}")
        if profiles or unmapped:
            try:
                _resume_windows_gateways_after_update(token)
            except Exception as restore_exc:
                rollback_failures.append(f"ordinary gateways: {restore_exc}")
        failed_service = current_service_name or "unknown"
        detail = f"Could not stop Windows gateway service {failed_service}: {exc}"
        if rollback_failures:
            detail += "; rollback failures: " + "; ".join(rollback_failures)
        raise RuntimeError(detail) from exc


def _cold_start_windows_gateway_after_update() -> bool:
    """Start a fresh detached gateway after update when one is installed but down.

    Invoked from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running when the update
    began, but an autostart entry (Scheduled Task / Startup-folder login item)
    is installed, signalling the user wants a gateway. Unlike the relaunch
    paths — which watch an old PID and respawn once it exits — this is a direct
    fresh spawn via the same hidden-console + breakaway path that
    ``hermes gateway start`` uses (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't produce a
    duplicate gateway.

    A successful ``Popen`` only proves the process was created, not that it
    survived (e.g. a Windows job object denying breakaway kills it before it
    logs anything — #84185). So the success line is gated on the same
    post-spawn liveness poll every other ``_spawn_detached`` caller uses
    (``gateway_windows._report_gateway_start``), instead of being printed
    unconditionally from the returned PID.
    """
    if not _m()._is_windows():
        return True
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway cold-start helpers: {exc}"
        ) from exc

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return True
    except Exception as exc:
        raise RuntimeError(
            f"Could not re-check gateway liveness before cold-start: {exc}"
        ) from exc

    try:
        if _desktop_owns_gateway_lifecycle():
            logger.debug(
                "Skipping Windows gateway cold-start: Desktop owns gateway lifecycle"
            )
            return True
    except Exception as exc:
        raise RuntimeError(
            "Could not re-check Desktop gateway-lifecycle ownership before cold-start: "
            f"{exc}"
        ) from exc

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        raise RuntimeError(f"Could not cold-start Windows gateway after update: {exc}") from exc

    if not pid:
        raise RuntimeError("Windows gateway cold-start did not return a process ID")
    ready_pids = gateway_windows._wait_for_gateway_ready()
    if not ready_pids:
        raise RuntimeError(
            f"Windows gateway cold-start PID {pid} did not become ready"
        )
    print()
    print(
        "✓ Gateway started via cold-start after update "
        f"(PID: {', '.join(map(str, ready_pids))})"
    )
    # Persist the PIDs this ✓ vouched for so a death AFTER the updater exits
    # (parent Job Object teardown, #91675) is reported by the next CLI
    # invocation instead of staying silent. Best-effort.
    try:
        gateway_windows._write_start_attestation(
            ready_pids, "cold-start after update"
        )
    except Exception:
        pass
    return True


def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` +
    ``gateway.vbs``) are persistence artifacts written once at install time —
    ``hermes update`` never touched them, so installs created before the
    hidden-console rework (aa2ae36c3f) kept launching the gateway through
    ``pythonw.exe`` forever: every descendant spawn flashed a conhost
    (#54220/#56747) and, since #70344, the console-less gateway died at
    startup with ``RuntimeError: sys.stderr is None`` (#71671).

    The task's /TR points at a stable script path, so rewriting the files in
    place retargets the task without any schtasks call (no UAC needed).
    ``_write_task_script`` is idempotent and renders from current code, so
    this is a no-op for modern installs. Best-effort: a failed refresh must
    never fail the update.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")
    except Exception as exc:
        logger.debug("Could not refresh Windows gateway launchers after update: %s", exc)


def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Sync the installer's bootstrap-cache scripts from the fresh checkout.

    The Desktop GUI updater (``hermes-setup.exe``) executes
    ``$HERMES_HOME/bootstrap-cache/install-<ref>.ps1`` (or ``.sh``) for its
    repair/bootstrap stages. Installer binaries built before the #67193
    cache-refresh fix (June 2026 and earlier) NEVER re-download a cached
    branch-ref script — ``install-main.ps1`` cached at install time is
    reused forever, executing months-stale code with long-fixed bugs (the
    2026-08-09 incident: a June 4 cached script's venv stage lacked the
    #81327 process-tree sweep and died on ``Access denied``). The binary
    has no self-update path, so the poisoned cache outlives every
    ``hermes update``.

    Overwriting the cached script for *branch* with the freshly pulled
    ``scripts/install.ps1`` / ``scripts/install.sh`` on every update turns
    the stale binary's unconditional reuse into a feature: it "reuses" a
    file this function keeps permanently current. Post-#67193 installers
    re-download on each run anyway, so for them this is a harmless
    pre-seed of the same bytes.

    Scope guards, mirroring ``install_script.rs``:

    - Only the cache key for the update-target *branch* is rewritten
      (``sanitize_ref``: non ``[A-Za-z0-9._-]`` chars become ``_``, so
      ``bb/gui`` → ``install-bb_gui.ps1``). Sibling mutable refs cache
      DIFFERENT branches' scripts — updating main must not clobber
      ``install-bb_gui.ps1`` with main's script.
    - Commit-SHA pins are immutable by design and never touched. The
      installer's ``is_valid_commit()`` accepts **7–40** hex chars, so an
      abbreviated pin like ``install-4ce1994.ps1`` is just as immutable as
      a full 40-hex one; the sanitized *branch* is additionally required
      to not itself look like a commit pin (defense in depth against a
      caller passing a SHA as the branch).

    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format
    (#67193 encoding fix). Best-effort: a failed refresh must never fail
    the update.
    """
    try:
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): 7-40 hex chars is an
        # immutable commit pin — abbreviated SHAs included. Never rewrite.
        if _re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            if not src.is_file():
                continue
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                # Match the installer's cache format: PowerShell needs the
                # UTF-8 BOM or localized/em-dash text mis-decodes (#67193).
                data = b"\xef\xbb\xbf" + data
            if cached.read_bytes() == data:
                continue  # already current
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print(
                "  ✓ Refreshed installer bootstrap-cache script(s): "
                + ", ".join(sorted(refreshed))
            )
    except Exception as exc:
        logger.debug("Could not refresh bootstrap-cache scripts after update: %s", exc)


def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    if not token or not token.get("resume_needed"):
        return
    if not _m()._is_windows():
        token["resume_needed"] = False
        return

    # Regenerate the persisted launcher scripts before respawning anything,
    # so a legacy pythonw-era Scheduled Task / Startup entry comes back on
    # current hidden-console design at the next login too.
    _m()._refresh_windows_gateway_launchers()

    services = list(token.get("services") or [])
    token.setdefault("expected_services", list(services))
    verified_restarts = list(token.get("restarted_services") or [])
    restarted_services = []
    failed_services = []
    for service_name in services:
        try:
            _start_windows_gateway_service(str(service_name))
            restarted_services.append(str(service_name))
            if str(service_name) not in verified_restarts:
                verified_restarts.append(str(service_name))
        except Exception as exc:
            logger.warning(
                "Could not restart Windows gateway service %s after update: %s",
                service_name,
                exc,
            )
            print(f"  ⚠ Could not restart Windows gateway service: {service_name}")
            failed_services.append(str(service_name))

    if failed_services:
        token["services"] = failed_services
        token["restarted_services"] = verified_restarts
        raise RuntimeError(
            "Could not restart Windows gateway service(s): "
            + ", ".join(failed_services)
        )
    token["services"] = []
    token["restarted_services"] = verified_restarts
    if restarted_services:
        print()
        print(
            "  ✓ Restarted Windows gateway service(s): "
            + ", ".join(restarted_services)
        )

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            if not _m()._cold_start_windows_gateway_after_update():
                raise RuntimeError("Windows gateway cold-start was not verified")
            token["cold_start_if_installed"] = False
        token["resume_needed"] = False
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway restart helper: {exc}"
        ) from exc

    relaunched = []
    failed_profiles = {}
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
            else:
                failed_profiles[str(profile)] = int(old_pid)
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )
            failed_profiles[str(profile)] = int(old_pid)

    # Surface the outcome on the token (#91277 Phase 2 plan-vs-execution
    # reconciliation): the git-based update path's fleet reconciliation
    # cross-checks every planned runtime against restarted_services /
    # relaunched_profiles / externally_supervised_profiles / killed_pids —
    # bookkeeping this Windows-specific pause/resume never fed, so a
    # correctly-paused-and-relaunched Windows gateway was reported
    # "unaccounted" (loud warning + exit 1) even though the restart
    # succeeded. The caller merges this into the shared
    # relaunched_profiles list before reconciliation runs. A profile whose
    # relaunch genuinely failed is deliberately left off this list — it
    # must still surface as unaccounted so the user is told to restart it
    # manually (Windows has no watcher to recover a failed relaunch).
    token["relaunched_profiles"] = relaunched

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
    unmapped_relaunched = 0
    failed_unmapped = []
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            failed_unmapped.append(entry)
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
            else:
                failed_unmapped.append(entry)
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )
            failed_unmapped.append(entry)

    token["profiles"] = failed_profiles
    token["unmapped"] = failed_unmapped
    if failed_profiles or failed_unmapped:
        raise RuntimeError("Could not restart every paused Windows gateway")

    # A truthy return from the launch helpers only proves the detached
    # watcher process was created — not that the gateway it respawns
    # survived. A parent Job Object that denies CREATE_BREAKAWAY_FROM_JOB
    # kills the freshly respawned gateway on updater teardown before it
    # writes a single log line, yet "✓ Restarting" was printed anyway
    # (#48820, 3rd/4th repro). Verify a stable gateway process actually
    # exists before vouching for the resume, using the same
    # provisional-hit + confirmation-window poll every other spawn path
    # uses (#91675). all_profiles=True because the resume covers the fleet.
    if relaunched or unmapped_relaunched:
        try:
            from hermes_cli import gateway_windows
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Windows gateway liveness helpers: {exc}"
            ) from exc
        ready_pids = gateway_windows._wait_for_gateway_ready(
            timeout_s=30.0, all_profiles=True
        )
        if not ready_pids:
            token["profiles"] = dict(profiles)
            token["unmapped"] = list(unmapped)
            print()
            print(
                "  ⚠ Windows gateway restart could not be verified — no stable "
                "gateway process appeared after relaunch."
            )
            print(
                "    (The respawned gateway may have been killed by a parent "
                "Job Object during updater teardown, #48820.)"
            )
            print("    Recover with: hermes gateway restart")
            raise RuntimeError(
                "Windows gateway relaunch after update was not verified alive"
            )
        # Persist the PIDs this ✓ vouches for so a death AFTER the updater
        # exits (parent Job Object teardown, #91675) is reported by the next
        # CLI invocation instead of staying silent. Best-effort.
        try:
            gateway_windows._write_start_attestation(
                ready_pids, "post-update relaunch"
            )
        except Exception:
            pass

    token["resume_needed"] = False

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )
