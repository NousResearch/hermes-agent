"""Gateway fleet restart orchestration and restart-state helpers.

Extracted mechanically from :mod:`hermes_cli.update_orchestrator` so each
updater module remains below the 2,000-line structural cap.
"""


def _restart_gateway_fleet(
    *,
    gateway_mode: bool,
    desktop_build_ok: bool,
    pre_update_plan,
    windows_gateway_resume,
):
    """Restart and reconcile the gateway fleet after the checkout update."""
    # Write exit code *before* the gateway restart attempt.
    # When running as ``hermes update --gateway`` (spawned by the gateway's
    # /update command), this process lives inside the gateway's systemd
    # cgroup.  A graceful SIGUSR1 restart keeps the drain loop alive long
    # enough for the exit-code marker to be written below, but the
    # fallback ``systemctl restart`` path (see below) kills everything in
    # the cgroup (KillMode=mixed → SIGKILL to remaining processes),
    # including us and the wrapping bash shell.  The shell never reaches
    # its ``printf $status > .update_exit_code`` epilogue, so the
    # exit-code marker file would never be created.  The new gateway's
    # update watcher would then poll for 30 minutes and send a spurious
    # timeout message.
    #
    # Writing the marker here — after git pull + pip install succeed but
    # before we attempt the restart — ensures the new gateway sees it
    # regardless of how we die. The verified summary includes Desktop and
    # SQLite-runtime health, so neither failure is reported as "0" to the
    # gateway watcher (gateway/run.py).
    if gateway_mode:
        _write_gateway_update_exit_code(update_complete)

    gateway_fleet_restart_incomplete = False
    gateway_restart_phase_errors: list[str] = []
    # Snapshot of gateways running before we touch anything. Stays empty
    # until we successfully import the probe and are about to stop/drain —
    # so an exception raised before we touch any gateway keeps this empty
    # (nothing to fail closed on), while a failure after we have stopped a
    # discovered gateway lets the handler fail closed on an empty survivor
    # probe rather than reporting a clean update (#78574).
    _pre_restart_gateway_pids: list | None = []
    # Declared outside the restart try/except below (and never reset
    # to None) so it's always safe to read afterwards even if that
    # block raises before reaching its own restart bookkeeping —
    # needed to forward already-restarted units to
    # ``_finish_dashboard_update_cleanup`` (review on #83595).
    restarted_services: list = []
    # Scope-qualified twin of ``restarted_services`` for systemd units:
    # ``user/hermes-serve`` and ``system/hermes-serve`` are two different
    # processes, and the abort-recovery child needs to know WHICH one was
    # already settled. ``restarted_services`` itself keeps its bare-name
    # vocabulary — the fleet probe, the receipt and the operator summary
    # all read it (#92145 review).
    restarted_scoped_units: set = set()
    # Keep these restart bookkeeping collections defined even when the
    # phase raises before its platform-specific imports initialize them.
    # The abort recovery and the fleet reconciliation both consume the
    # pre-update plan in that early-failure shape.
    failed_or_stale_units: list = []
    relaunched_profiles: list = []
    externally_supervised_profiles: list = []
    # Same outside-the-try treatment: the post-restart fleet version
    # check consults killed_pids to decide whether to wait for
    # freshly-restarted gateways to settle, and the phase's except
    # path forwards it to the update receipt.
    killed_pids: set = set()

    # Auto-restart ALL gateways after update.
    # The code update (git pull) is shared across all profiles, so every
    # running gateway needs restarting to pick up the new code.
    #
    # Purge stale cached Hermes modules FIRST: the import below pulls
    # freshly-updated gateway source into this pre-update interpreter,
    # and any already-cached sibling module (cli_output, status, ...)
    # that the new source expects a new symbol from would otherwise
    # ImportError and abort this whole phase (2026-08-20 field failure:
    # new gateway.py ← stale cli_output missing line_input).
    _m()._purge_stale_hermes_modules()
    try:
        from hermes_cli.gateway import (
            is_macos,
            supports_systemd_services,
            _ensure_user_systemd_env,
            find_gateway_pids,
            find_profile_gateway_processes,
            _prepare_profile_gateway_update_restart,
            _get_service_pids,
            _graceful_restart_via_sigusr1,
            _wait_for_gateway_exit,
        )
        import signal as _signal

        def _wait_for_service_active(
            scope_cmd_: list,
            svc_name_: str,
            timeout: float = 10.0,
        ) -> bool:
            """Poll ``systemctl is-active`` until the unit reports active.

            systemd's Stopped -> Started transition after a graceful exit
            (or a hard restart) is not instantaneous; a one-shot check
            races that window and falsely reports the unit as down.
            Poll every 0.5s up to ``timeout`` seconds before giving up.
            """
            deadline = _time.monotonic() + max(timeout, 0.5)
            while True:
                try:
                    _verify = subprocess.run(
                        scope_cmd_ + ["is-active", svc_name_],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=5,
                    )
                    if _verify.stdout.strip() == "active":
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                if _time.monotonic() >= deadline:
                    return False
                _time.sleep(0.5)

        def _service_restart_sec(
            scope_cmd_: list,
            svc_name_: str,
            default: float = 0.0,
        ) -> float:
            """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

            After a graceful exit-75, systemd waits ``RestartSec`` before
            respawning the unit.  Callers that poll for ``is-active``
            must use a timeout >= ``RestartSec`` + transition slack, or
            they'll give up *during* the cooldown window and wrongly
            conclude the unit didn't relaunch.
            """
            try:
                _show = subprocess.run(
                    scope_cmd_
                    + [
                        "show",
                        svc_name_,
                        "--property=RestartUSec",
                        "--value",
                    ],
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return default
            raw = (_show.stdout or "").strip()
            # systemd emits values like "30s", "100ms", "1min 30s", or
            # "infinity".  Parse conservatively; on any miss return default.
            if not raw or raw == "infinity":
                return default
            total = 0.0
            matched = False
            for part in raw.split():
                for _suf, _mult in (
                    ("ms", 0.001),
                    ("us", 0.000001),
                    ("min", 60.0),
                    ("s", 1.0),
                ):
                    if part.endswith(_suf):
                        try:
                            total += float(part[: -len(_suf)]) * _mult
                            matched = True
                        except ValueError:
                            pass
                        break
            return total if matched else default

        _manage_cmd_cache: dict = {}

        def _resolve_manage_cmd(scope_: str, scope_cmd_: list, svc_name_: str):
            """Resolve the command prefix for manage-units operations.

            Read-only systemctl calls (``is-active``, ``show``,
            ``list-units``) work unprivileged, but manage-units verbs
            (``reset-failed``, ``start``, ``restart``) on a *system*
            service trigger a polkit ``org.freedesktop.systemd1.manage-units``
            authentication prompt when run as a non-root user.  That
            interactive prompt runs inside our captured subprocess with a
            10-15s timeout — the user sees the prompt flash and "exit
            directly" before they can answer, and the resulting
            TimeoutExpired used to be swallowed silently.

            Strategy: if root, plain systemctl.  If not root, try
            non-interactive sudo (``sudo -n``) — first a blanket probe,
            then a targeted ``systemctl reset-failed`` probe so a
            least-privilege sudoers entry scoped to
            ``systemctl ... hermes-gateway*`` also qualifies
            (``reset-failed`` is an idempotent no-op we run before every
            privileged restart anyway).  If neither works, return None —
            the caller must SKIP the restart (without draining the
            gateway first!) and tell the user how to restart manually.
            ``--no-ask-password`` guarantees polkit can never hang a
            captured subprocess on this path.
            """
            if scope_ in _manage_cmd_cache:
                return _manage_cmd_cache[scope_]
            cmd = scope_cmd_ + ["--no-ask-password"]
            if (
                scope_ == "system"
                and hasattr(os, "geteuid")
                and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
            ):
                sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
                sudo_ok = False
                try:
                    _probe = subprocess.run(
                        ["sudo", "-n", "true"],
                        capture_output=True,
                        timeout=5,
                    )
                    sudo_ok = _probe.returncode == 0
                    if not sudo_ok:
                        # Blanket sudo refused — a targeted sudoers entry
                        # (NOPASSWD for systemctl ... hermes-gateway*)
                        # may still allow the exact commands we need.
                        _probe = subprocess.run(
                            sudo_cmd + ["reset-failed", svc_name_],
                            capture_output=True,
                            timeout=5,
                        )
                        sudo_ok = _probe.returncode == 0
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    sudo_ok = False
                cmd = sudo_cmd if sudo_ok else None
            _manage_cmd_cache[scope_] = cmd
            return cmd

        # Wait budget for graceful SIGUSR1 restarts.  In-band restart
        # may defer stop() until active turns finish
        # (``restart_after_turn_timeout``, #77184) and then spend up to
        # ``restart_drain_timeout`` inside stop(). Cover both phases so
        # we don't fall back to a hard kill while the gateway is still
        # patiently waiting for the requesting turn. On older systemd
        # units without SIGUSR1 wiring this wait just times out and we
        # fall back to ``systemctl restart`` (the old behaviour).
        try:
            from hermes_cli.gateway import _get_restart_exit_wait_budget

            _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
        except Exception:
            _drain_budget = 45.0

        failed_or_stale_units = []
        killed_pids = set()
        relaunched_profiles = []
        externally_supervised_profiles = []

        # Record which gateways are running before any stop/drain, so a
        # later failure that leaves the survivor probe empty can still be
        # recognised as "a running gateway was stopped and did not come
        # back" rather than "nothing was running" (#78574). Best-effort:
        # if the probe itself raises, leave the snapshot as-is (the
        # survivor probe's own None result already fails closed).
        try:
            _pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            _pre_restart_gateway_pids = None

        # --- Systemd services (Linux) ---
        # Discover all hermes-gateway* units (default + profiles) plus
        # hermes-serve* units (the Desktop app's backend, #83438).
        if supports_systemd_services():
            try:
                _ensure_user_systemd_env()
            except Exception:
                pass

            for scope, scope_cmd in [
                ("user", ["systemctl", "--user"]),
                ("system", ["systemctl"]),
            ]:
                try:
                    result = subprocess.run(
                        scope_cmd
                        + [
                            "list-units",
                            "hermes-gateway*",
                            "hermes-serve*",
                            "--plain",
                            "--no-legend",
                            "--no-pager",
                        ],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=10,
                    )
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired as exc:
                    # Discovery timeout — skip this scope, keep the other.
                    print(
                        f"  ⚠ systemctl timed out listing {scope}-scope "
                        f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
                        f"Check the gateway with: hermes gateway status"
                    )
                    continue

                def _restart_one_systemd_gateway_unit(svc_name: str) -> None:
                    # Check if active
                    check = subprocess.run(
                        scope_cmd + ["is-active", svc_name],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=5,
                    )
                    if check.stdout.strip() != "active":
                        return

                    # Resolve how we may run manage-units verbs
                    # (reset-failed/start/restart) for this scope.
                    # None ⇒ no non-interactive privilege path; we
                    # must avoid those verbs entirely or polkit will
                    # throw an interactive auth prompt inside our
                    # captured 10-15s subprocess (the user sees it
                    # flash and "exit directly" — reported June 2026).
                    _manage_cmd = _resolve_manage_cmd(
                        scope, scope_cmd, svc_name
                    )

                    # Prefer a graceful SIGUSR1 restart so in-flight
                    # agent runs drain instead of being SIGKILLed.
                    # The gateway's SIGUSR1 handler calls
                    # request_restart(via_service=True) → drain →
                    # exit; systemd's Restart=always respawns the unit.
                    # hermes-serve has no such handler (it isn't
                    # gateway/run.py), so skip straight to the blunt
                    # restart below rather than sending it an unhandled
                    # signal and waiting out the drain budget for
                    # nothing.
                    _main_pid = 0
                    if _service_unit_supports_graceful_sigusr1_restart(svc_name):
                        try:
                            _show = subprocess.run(
                                scope_cmd
                                + [
                                    "show",
                                    svc_name,
                                    "--property=MainPID",
                                    "--value",
                                ],
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=5,
                            )
                            _main_pid = int((_show.stdout or "").strip() or 0)
                        except (
                            ValueError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            _main_pid = 0

                    _graceful_ok = False
                    if _main_pid > 0:
                        # Three-way triage (ancestor fire-and-forget
                        # #100179 / wedged escalation #81642 / normal
                        # graceful drain) — shared with the bare-process
                        # path below.
                        _graceful_ok = _drain_or_signal_gateway_for_update(
                            _main_pid, _drain_budget, svc_name
                        )

                    if _graceful_ok:
                        # Gateway exited after a planned restart.
                        # ``Restart=always`` means systemd WILL respawn
                        # the unit — but only after
                        # ``RestartSec`` (default 60s on our unit
                        # file). That 60s wait is a crash-loop guard,
                        # and is the right default when the gateway
                        # dies unexpectedly. For a voluntary restart
                        # on update, it's dead time the user watches.
                        #
                        # Shortcut it: ``reset-failed`` + ``start``
                        # skips RestartSec entirely (we're manually
                        # initiating the unit, not waiting for
                        # systemd's auto-restart logic). Takes about
                        # as long as the process takes to come up
                        # (~1-3s on a warm box).
                        #
                        # If the unit is already active because
                        # RestartSec elapsed while we were draining,
                        # ``start`` is a no-op and we fall through to
                        # the poll below. Either way we collapse the
                        # 60s+ delay to a ~5s one.
                        #
                        # The shortcut needs manage-units privileges.
                        # Without them (system service, non-root, no
                        # passwordless sudo) skip it — systemd's own
                        # auto-restart still relaunches the unit after
                        # RestartSec, no privileges required.
                        if _manage_cmd is not None:
                            subprocess.run(
                                _manage_cmd + ["reset-failed", svc_name],
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=10,
                            )
                            subprocess.run(
                                _manage_cmd + ["start", svc_name],
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=15,
                            )
                            # Short poll: the gateway should be up
                            # within a few seconds now that we
                            # bypassed RestartSec.
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=10.0,
                            ):
                                restarted_services.append(svc_name)
                                return
                        # Passive poll: systemd's auto-restart fires
                        # after RestartSec regardless of privileges.
                        # This is the primary path when _manage_cmd is
                        # None, and the fallback when the explicit
                        # start didn't take.
                        _restart_sec = _service_restart_sec(
                            scope_cmd,
                            svc_name,
                            default=0.0,
                        )
                        _post_drain_timeout = max(
                            10.0,
                            _restart_sec + 10.0,
                        )
                        if _manage_cmd is None and _restart_sec > 5.0:
                            print(
                                f"  → {svc_name}: waiting for systemd "
                                f"auto-restart (~{int(_restart_sec)}s; "
                                "no root for an immediate restart)..."
                            )
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=_post_drain_timeout,
                        ):
                            restarted_services.append(svc_name)
                            return
                        # Process exited but wasn't respawned (older
                        # unit without Restart=on-failure or
                        # RestartForceExitStatus=75).  Fall through
                        # to systemctl start/restart.
                        print(
                            f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                        )

                    # Forcing a restart requires manage-units
                    # privileges.  Without a non-interactive path,
                    # running systemctl here would spawn a polkit
                    # auth prompt inside a captured 10-15s subprocess
                    # — it flashes and dies before the user can
                    # answer.  Skip with clear instructions instead.
                    if _manage_cmd is None:
                        failed_or_stale_units.append(svc_name)
                        print(
                            f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                            f"    Restart it manually to load the new version:\n"
                            f"      sudo systemctl restart {svc_name}\n"
                            f"    To let `hermes update` restart it automatically, allow\n"
                            f"    passwordless sudo for systemctl, or run updates with sudo."
                        )
                        return

                    # Fallback: blunt systemctl restart.  This is
                    # what the old code always did; we get here only
                    # when the graceful path failed (unit missing
                    # SIGUSR1 wiring, drain exceeded the budget,
                    # restart-policy mismatch).
                    #
                    # Always `reset-failed` first.  If systemd's own
                    # auto-restart attempts already parked the unit
                    # in a failed state (transient CHDIR / OOM /
                    # filesystem race after our drain + exit-75),
                    # a plain `systemctl restart` can wedge against
                    # the RestartSec backoff and leave the unit
                    # dead.  Clearing the failed state first makes
                    # the restart idempotent.  Mirrors the recovery
                    # path in `hermes gateway restart`
                    # (`systemd_restart()`) as of PR #20949.
                    subprocess.run(
                        _manage_cmd + ["reset-failed", svc_name],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=10,
                    )
                    restart = subprocess.run(
                        _manage_cmd + ["restart", svc_name],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=15,
                    )
                    if restart.returncode == 0:
                        # Verify the service actually survived the
                        # restart.  systemctl restart returns 0 even
                        # if the new process crashes immediately.
                        if _wait_for_service_active(
                            scope_cmd,
                            svc_name,
                            timeout=10.0,
                        ):
                            restarted_services.append(svc_name)
                        else:
                            # Retry once — transient startup failures
                            # (stale module cache, import race) often
                            # resolve on the second attempt.  Again
                            # clear any failed state first so the
                            # retry isn't blocked by the previous
                            # crash.
                            print(
                                f"  ⚠ {svc_name} died after restart, retrying..."
                            )
                            subprocess.run(
                                _manage_cmd + ["reset-failed", svc_name],
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=10,
                            )
                            subprocess.run(
                                _manage_cmd + ["restart", svc_name],
                                capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=15,
                            )
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=10.0,
                            ):
                                restarted_services.append(svc_name)
                                print(f"  ✓ {svc_name} recovered on retry")
                            else:
                                failed_or_stale_units.append(svc_name)
                                _scope_flag = "--user " if scope == "user" else ""
                                _sudo_hint = "sudo " if scope == "system" else ""
                                print(
                                    f"  ✗ {svc_name} failed to stay running after restart.\n"
                                    f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                    f"    Recover manually:\n"
                                    f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                    f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                                )
                    else:
                        failed_or_stale_units.append(svc_name)
                        print(
                            f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                        )

                def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
                    # Isolate the timeout to this unit and keep going
                    # (#68523). A scope-wide handler used to abort every
                    # later gateway and leave the fleet on mixed code.
                    failed_or_stale_units.append(svc_name)
                    print(
                        f"  ⚠ systemctl timed out restarting {svc_name} "
                        f"({exc.cmd if exc.cmd else 'unknown command'}); "
                        f"continuing with remaining gateways"
                    )

                # Everything this scope appends to ``restarted_services``
                # belongs to THIS systemd manager; qualify it before the
                # next scope can append a same-named unit. ``finally`` so a
                # phase abort mid-scope still carries the units it settled.
                _scope_mark = len(restarted_services)
                try:
                    _for_each_systemd_gateway_unit(
                        result.stdout,
                        process_unit=_restart_one_systemd_gateway_unit,
                        on_unit_timeout=_on_unit_timeout,
                    )
                finally:
                    restarted_scoped_units.update(
                        f"{scope}/{name}"
                        for name in restarted_services[_scope_mark:]
                    )

        # --- Launchd services (macOS) ---
        # Restart EVERY ai.hermes.gateway* LaunchAgent, not only the
        # invoking profile's — parity with the systemd branch above
        # (#41403). Per-label TimeoutExpired isolation happens inside.
        if is_macos():
            try:
                _restart_macos_launchd_gateways(
                    restarted_services,
                    failed_or_stale_units,
                    _drain_budget,
                )
            except (FileNotFoundError, ImportError):
                pass

        # --- Manual (non-service) gateways ---
        # Kill any remaining gateway processes not managed by a service.
        # Exclude PIDs that belong to just-restarted services so we don't
        # immediately kill the process that systemd/launchd just spawned.
        service_pids = _get_service_pids(all_profiles=True)
        manual_pids = find_gateway_pids(
            exclude_pids=service_pids, all_profiles=True
        )
        profile_processes = {
            proc.pid: proc
            for proc in find_profile_gateway_processes(exclude_pids=service_pids)
            if proc.pid in manual_pids
        }
        # Profile gateways we could not arm a relaunch for.  These must
        # NOT be left running: their modules are the pre-update ones and
        # every lazy import from here on mixes versions against the new
        # code on disk (#88654).  Handing them to the unmapped sweep
        # below stops them and surfaces them in the "Stopped N manual
        # gateway process(es) / Restart manually" summary, which is the
        # contract already used for gateways with no profile mapping.
        unrestartable_pids = set()
        for pid, proc in profile_processes.items():
            restart_mode = _prepare_profile_gateway_update_restart(
                proc.profile, pid
            )
            if restart_mode is None:
                # Previously a bare ``continue``: the gateway was neither
                # relaunched nor stopped nor mentioned, so it kept serving
                # from stale modules with no operator signal at all.
                print(
                    f"  ⚠ {proc.profile}: could not arm an automatic "
                    f"gateway restart for PID {pid} — stopping it instead "
                    "so it cannot keep running pre-update code"
                )
                unrestartable_pids.add(pid)
                continue
            # Prefer a graceful SIGUSR1 drain so in-flight agent runs
            # finish before the watcher respawns the gateway.  If the
            # gateway doesn't support SIGUSR1 or doesn't exit within
            # the drain budget, fall back to SIGTERM — the watcher
            # still sees the exit and relaunches either way.
            # Three-way triage (ancestor fire-and-forget #100179 /
            # wedged escalation #81642 / normal graceful drain) — shared
            # with the systemd path above.  The helper announces the
            # chosen path first: a drain wait can hold for the full
            # budget per gateway with no other output, and on surfaces
            # that stream update progress the silence reads as a hung
            # update (#44515).
            drained = _drain_or_signal_gateway_for_update(
                pid, _drain_budget, proc.profile
            )
            if not drained:
                try:
                    os.kill(pid, _signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            # Wait for the old process to fully exit before the watcher
            # spawns the new gateway.  Telegram holds the previous
            # getUpdates long-poll session open on its servers for up to
            # ~30s after the client disconnects.  If the new gateway
            # connects before that window expires it receives a 409
            # Conflict, which _handle_polling_conflict() recovers from
            # via back-off retries — but a brief wait here reduces the
            # chance of hitting that path at all, especially on fast
            # machines where the watcher loop restarts in < 1s.
            # We wait up to 5s for the process to exit (the OS-level
            # close, not the Telegram server-side expiry), then let the
            # watcher take over.  The Telegram adapter's retry logic
            # handles any remaining 409s if the server session is still
            # live when the new gateway polls.
            _wait_for_gateway_exit(timeout=5.0, force_after=None)
            killed_pids.add(pid)
            if restart_mode == "external-supervisor":
                externally_supervised_profiles.append(proc.profile)
            else:
                relaunched_profiles.append(proc.profile)

        for pid in manual_pids:
            if pid in profile_processes and pid not in unrestartable_pids:
                continue
            try:
                os.kill(pid, _signal.SIGTERM)
                killed_pids.add(pid)
            except (ProcessLookupError, PermissionError):
                pass

        if restarted_services or killed_pids:
            print()
            for svc in restarted_services:
                print(f"  ✓ Restarted {svc}")
            if relaunched_profiles:
                names = ", ".join(relaunched_profiles)
                print(f"  ✓ Restarting manual gateway profile(s): {names}")
            if externally_supervised_profiles:
                names = ", ".join(externally_supervised_profiles)
                print(
                    "  ✓ Handed gateway profile(s) back to their external "
                    f"supervisor: {names}"
                )
            unmapped_count = (
                len(killed_pids)
                - len(relaunched_profiles)
                - len(externally_supervised_profiles)
            )
            if unmapped_count:
                print(f"  → Stopped {unmapped_count} manual gateway process(es)")
                print("    Restart manually: hermes gateway run")
                if unmapped_count > 1:
                    print(
                        "    (or: hermes -p <profile> gateway run  for each profile)"
                    )

        if failed_or_stale_units:
            gateway_fleet_restart_incomplete = True
            if gateway_mode:
                _exit_code_path = get_hermes_home() / ".update_exit_code"
                try:
                    _exit_code_path.write_text("1", encoding="utf-8")
                except OSError:
                    pass
        _warn_incomplete_gateway_fleet_restart(failed_or_stale_units)

        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=bool(failed_or_stale_units),
            )
        except Exception:
            pass

        if not restarted_services and not killed_pids:
            # No gateways were running — nothing to do
            pass

        # --- Post-restart survivor sweep -----------------------------
        # Issue #17648: some gateways ignore SIGTERM (stuck drain,
        # blocked I/O, PID dead but zombie).  The detached profile
        # watchers wait 120s for the old PID to exit — if it never
        # does, no respawn happens and the user keeps hitting
        # ImportError against a stale sys.modules.  Give the
        # graceful paths a brief window to complete, then SIGKILL
        # any remaining pre-update PIDs so the watcher / service
        # manager can relaunch with fresh code.
        try:
            _time.sleep(3.0)
            _service_pids_after = _get_service_pids(all_profiles=True)
            _surviving = find_gateway_pids(
                exclude_pids=_service_pids_after,
                all_profiles=True,
            )
            # Scope to PIDs we already tried to kill during this
            # update (killed_pids).  Anything new is a gateway that
            # started AFTER our restart attempt — respecting user
            # intent, we don't kill those.
            _stuck = [pid for pid in _surviving if pid in killed_pids]
            if _stuck:
                print()
                print(
                    f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
                )
                from gateway.status import (
                    get_process_start_time as _get_process_start_time,
                    terminate_pid as _terminate_pid,
                )
                for pid in _stuck:
                    try:
                        # Routes through taskkill /T /F on Windows,
                        # SIGKILL on POSIX — _signal.SIGKILL doesn't
                        # exist on Windows so the old raw os.kill call
                        # used to crash the entire update path.
                        _terminate_pid(
                            pid,
                            force=True,
                            expected_start_time=_get_process_start_time(pid),
                        )
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                # Give the OS a beat to reap the processes so the
                # watchers see them exit and respawn.
                _time.sleep(1.5)
        except Exception as _sweep_exc:
            logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)

    except Exception as e:
        logger.debug("Gateway restart during update failed: %s", e)
        gateway_restart_phase_errors.append(str(e))
        # An exception escaping the whole phase means the drain/restart
        # output the user relies on never printed. Don't let that pass for
        # a clean update: surface it and treat the fleet as stale unless we
        # can positively prove no gateway is running (#78574).
        #
        # A positive-empty ``_surviving`` is only proof-of-safety when
        # nothing was running before we touched anything. If a gateway was
        # discovered pre-restart and none survive now, it was stopped and
        # its replacement was never verified — the same fail-open contract
        # this fix closes — so we must still fail closed on ``[]``.
        _surviving = _surviving_gateway_pids_after_failed_restart()
        _already_restarted_profiles = set(relaunched_profiles)
        _already_restarted_profiles.update(externally_supervised_profiles)
        for runtime in getattr(pre_update_plan, "runtimes", ()) or ():
            if getattr(runtime, "kind", None) != "gateway":
                continue
            profile = getattr(runtime, "profile", None)
            if not isinstance(profile, str):
                continue
            if any(
                _gateway_service_matches_profile(profile, service)
                for service in restarted_services
            ):
                _already_restarted_profiles.add(profile)
        _recovery_result = _recover_gateway_restart_after_abort(
            pre_update_plan,
            gateway_mode=gateway_mode,
            skip_profiles=_already_restarted_profiles,
            skip_units=set(restarted_scoped_units),
        )
        _recovery_serve_units = _recovery_result.get("serve_units") or {}
        _serve_units_failed = list(_recovery_serve_units.get("failed") or [])
        # Deliberately NOT merged into ``restarted_services``: that list is
        # the gateway phase's own vocabulary and feeds the fleet-probe
        # expectation. Serve coverage lives in the recovery result and the
        # receipt, where it can be read without changing gateway semantics.
        # A serve/dashboard runtime that is still the SAME process the
        # pre-update inventory saw is a live runtime on the pre-update
        # code generation — the exact unsafe state of #92145, and the one
        # the reporter hit through tui_gateway (hosted by `hermes serve`,
        # which is not a gateway profile and which no `gateway restart`
        # command can reach). Recovery may not report success while one
        # is still there, and it must never kill one: a manual or
        # Desktop-owned serve has no relaunch authority.
        _stale_runtime_rows = _surviving_pre_update_serve_runtimes(
            pre_update_plan
        )
        _recovery_result["stale_runtimes"] = _stale_runtime_rows
        # Only systemd-VERIFIED outcomes may claim supervisor coverage.
        # A relaunch that merely exited 0 ("relaunch_attempted") was never
        # observed by the code and must not clear the incomplete flag.
        _recovery_verified = set(_recovery_result.get("verified") or [])
        if _recovery_verified:
            relaunched_profiles.extend(
                profile
                for profile in sorted(_recovery_verified)
                if profile not in relaunched_profiles
            )
        _planned_gateway_runtimes = [
            runtime
            for runtime in getattr(pre_update_plan, "runtimes", ()) or ()
            if getattr(runtime, "kind", None) == "gateway"
            and isinstance(getattr(runtime, "profile", None), str)
        ]
        _planned_gateway_profiles = {
            runtime.profile for runtime in _planned_gateway_runtimes
        }
        _covered_gateway_profiles = (
            _already_restarted_profiles | _recovery_verified
        )
        _recovery_complete = _abort_recovery_is_complete(
            planned_gateway_profiles=_planned_gateway_profiles,
            covered_gateway_profiles=_covered_gateway_profiles,
            recovery_result=_recovery_result,
            stale_runtime_rows=_stale_runtime_rows,
        )
        if _recovery_complete:
            # The fresh child is the recovery terminal result. Leave the
            # final fleet-version matrix below as the authoritative
            # read-back before the update is declared successful.
            gateway_fleet_restart_incomplete = False
        elif (
            _restart_phase_failure_is_incomplete(
                _surviving, _pre_restart_gateway_pids
            )
            or _stale_runtime_rows
            or _serve_units_failed
        ):
            gateway_fleet_restart_incomplete = True
            _warn_gateway_restart_phase_aborted(e, _surviving)
            _warn_stale_serve_runtimes(_stale_runtime_rows)
            if gateway_mode:
                _exit_code_path = get_hermes_home() / ".update_exit_code"
                try:
                    _exit_code_path.write_text("1", encoding="utf-8")
                except OSError:
                    pass
        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=gateway_fleet_restart_incomplete,
                phase_error=str(e),
                fresh_recovery=_recovery_result,
            )
        except Exception:
            pass

    try:
        _m()._resume_windows_gateways_after_update(windows_gateway_resume)
    except Exception as _windows_resume_exc:
        gateway_fleet_restart_incomplete = True
        gateway_restart_phase_errors.append(str(_windows_resume_exc))
        print(
            "  ⚠ Windows gateway service restart incomplete: "
            f"{_windows_resume_exc}"
        )
        if gateway_mode:
            _exit_code_path = get_hermes_home() / ".update_exit_code"
            try:
                _exit_code_path.write_text("1", encoding="utf-8")
            except OSError:
                pass

    if isinstance(windows_gateway_resume, dict):
        # Feed Windows's own pause/resume outcome into the same
        # relaunched_profiles bookkeeping the systemd/launchd restart
        # phase populates, so the #91277 Phase 2 reconciliation below
        # does not report a correctly-relaunched Windows gateway as
        # "unaccounted" (a runtime the plan saw but no bookkeeping
        # mentions — the reconciliation's blind-spot tripwire). A
        # profile whose relaunch genuinely failed is intentionally
        # left out of the token's list, so it still surfaces there.
        # Best-effort: the restart-phase try/except above may have
        # raised before relaunched_profiles was initialized, so this
        # must never itself abort the update.
        try:
            for _win_profile in windows_gateway_resume.get("relaunched_profiles") or []:
                if _win_profile not in relaunched_profiles:
                    relaunched_profiles.append(_win_profile)
        except Exception as _win_reconcile_exc:
            logger.debug(
                "Could not merge Windows relaunch outcome into fleet "
                "reconciliation bookkeeping: %s",
                _win_reconcile_exc,
            )
        windows_restarted = list(
            windows_gateway_resume.get("restarted_services") or []
        )
        for service_name in windows_restarted:
            if service_name not in restarted_services:
                restarted_services.append(service_name)
        service_profiles = windows_gateway_resume.get("service_profiles") or {}
        for service_name in windows_restarted:
            profile_name = service_profiles.get(service_name)
            if profile_name and profile_name not in relaunched_profiles:
                relaunched_profiles.append(profile_name)
        pending_services = list(windows_gateway_resume.get("services") or [])
        for service_name in pending_services:
            label = str(service_profiles.get(service_name) or service_name)
            if label not in failed_or_stale_units:
                failed_or_stale_units.append(label)

        try:
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=restarted_services,
                relaunched_profiles=relaunched_profiles,
                externally_supervised_profiles=externally_supervised_profiles,
                killed_pids=sorted(killed_pids),
                failed_units=failed_or_stale_units,
                incomplete=(
                    gateway_fleet_restart_incomplete
                    or bool(failed_or_stale_units)
                ),
                phase_error="; ".join(gateway_restart_phase_errors) or None,
            )
        except Exception:
            pass

    return (
        gateway_fleet_restart_incomplete,
        gateway_restart_phase_errors,
        _pre_restart_gateway_pids,
        restarted_services,
        failed_or_stale_units,
        relaunched_profiles,
        externally_supervised_profiles,
        killed_pids,
    )


def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped gateway-restart-phase exception must fail the update.

    Fail closed unless we can positively prove the fleet is safe:

    * ``surviving is None`` — the survivor probe could not determine state
      (typically the freshly-pulled ``hermes_cli.gateway`` no longer imports,
      one of the ways the phase aborts). Assume stale.
    * ``surviving`` non-empty — a gateway is still running pre-update code.
    * ``surviving == []`` — nothing is running now. That is proof-of-safety
      ONLY when nothing was running before we touched anything. If a gateway
      was discovered pre-restart (``pre_restart_pids`` non-empty, or ``None``
      meaning the pre-state could not be read), it was stopped without a
      verified replacement, so we still fail closed (#78574).
    """
    if surviving is None or surviving:
        return True
    # surviving == []: safe only if we know nothing was running beforehand.
    return pre_restart_pids is None or bool(pre_restart_pids)


def _fleet_probe_expected_runtimes(
    pre_update_plan,
    pre_restart_pids,
    windows_resume_token,
    restarted_services,
    killed_pids,
) -> bool:
    """Whether the post-update fleet probe should have produced rows.

    The zero-rows fail-open (#93406): ``collect_fleet_versions()`` swallows
    every probe failure via ``logger.debug()`` and ``print_fleet_version_matrix([])``
    early-returns ``False``, so an empty snapshot reads as \"healthy fleet\" and
    the update exits 0.  An empty snapshot is only proof-of-safety when NOTHING
    says a gateway existed before the update.  Any of these signals means at
    least one runtime was (or may have been) live pre-update, so zero rows is
    verification failure, not health:

    * ``restarted_services`` / ``killed_pids`` — the POSIX restart phase
      touched live gateways.
    * ``pre_restart_pids`` non-empty, or ``None`` (pre-state unreadable —
      cannot prove nothing was running; same contract as
      ``_restart_phase_failure_is_incomplete``, #78574).
    * the pre-update plan inventoried ≥1 runtime.

    ``windows_resume_token`` is deliberately EXCLUDED (#93406 residual). The
    pause/resume token is bookkeeping for ``_pause_windows_gateways_for_update``
    / ``_resume_windows_gateways_after_update`` — it is not a runtime
    inventory, and its entries do not correspond to rows
    ``collect_fleet_versions()`` is capable of returning:

    * ``unmapped`` entries (Scheduled-Task gateways) never publish
      ``gateway_state.json`` rows at all, and
    * a paused profile gateway is resumed as a DETACHED relaunch that may not
      republish its identity within the probe window.

    Counting the token therefore made ``_fleet_rows_expected`` True on every
    Windows update that had paused a gateway, the probe's polling window ran
    out with zero rows on a perfectly healthy update, and verification
    reported "no rows … verification incomplete" and exited 1 after a long
    silent wait. Expected-runtimes must key only on signals that map to rows
    the probe can actually see; a genuinely live pre-update Windows gateway
    is already covered by ``pre_restart_pids`` and the plan inventory. The
    parameter stays in the signature so the call site keeps passing the token
    (cheap, explicit, and the docstring is where the exclusion is explained).

    The same condition gates the 2.0s settle sleep: a freshly restarted
    gateway needs the settle window to rewrite ``gateway_state.json``.

    Note this keys ONLY on zero-rows-despite-expected-runtimes.  A non-empty
    snapshot — including rows in ``unknown`` state — is still judged solely by
    ``print_fleet_version_matrix``.
    """
    del windows_resume_token  # excluded on purpose — see docstring (#93406)
    if restarted_services or killed_pids:
        return True
    if pre_restart_pids is None or pre_restart_pids:
        return True
    try:
        if pre_update_plan is not None and pre_update_plan.runtimes:
            return True
    except Exception:
        pass
    return False


def _print_items(items, label, key, fallback_key=None):
    if not items:
        return
    print(f"  {label}:")
    shown = items[:8]
    for it in shown:
        if isinstance(it, dict):
            name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
            desc = (it.get("description") or "").strip()
        else:
            # Defensive: some callers/mocks pass bare name strings.
            name = str(it)
            desc = ""
        if desc:
            print(f"      • {name} — {desc}")
        else:
            print(f"      • {name}")
    extra = len(items) - len(shown)
    if extra > 0:
        print(f"      … and {extra} more")


def _wait_for_service_active(
    scope_cmd_: list,
    svc_name_: str,
    timeout: float = 10.0,
) -> bool:
    """Poll ``systemctl is-active`` until the unit reports active.

    systemd's Stopped -> Started transition after a graceful exit
    (or a hard restart) is not instantaneous; a one-shot check
    races that window and falsely reports the unit as down.
    Poll every 0.5s up to ``timeout`` seconds before giving up.
    """
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        try:
            _verify = subprocess.run(
                scope_cmd_ + ["is-active", svc_name_],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=5,
            )
            if _verify.stdout.strip() == "active":
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)


def _service_restart_sec(
    scope_cmd_: list,
    svc_name_: str,
    default: float = 0.0,
) -> float:
    """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

    After a graceful exit-75, systemd waits ``RestartSec`` before
    respawning the unit.  Callers that poll for ``is-active``
    must use a timeout >= ``RestartSec`` + transition slack, or
    they'll give up *during* the cooldown window and wrongly
    conclude the unit didn't relaunch.
    """
    try:
        _show = subprocess.run(
            scope_cmd_
            + [
                "show",
                svc_name_,
                "--property=RestartUSec",
                "--value",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # systemd emits values like "30s", "100ms", "1min 30s", or
    # "infinity".  Parse conservatively; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in (
            ("ms", 0.001),
            ("us", 0.000001),
            ("min", 60.0),
            ("s", 1.0),
        ):
            if part.endswith(_suf):
                try:
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                except ValueError:
                    pass
                break
    return total if matched else default


def _drain_or_signal_gateway_for_update(
    pid: int,
    drain_budget: float,
    label: str,
) -> bool:
    """Decide how ``hermes update`` hands a running gateway over to new code.

    Three-way triage shared by the systemd and bare-process restart paths:

    1. **Gateway is an ancestor of this process** — THREE-WAY DEADLOCK BREAK
       (#100179). When ``hermes update`` runs INSIDE the gateway's own
       process tree (the hermes-auto-update cron job is the canonical case),
       waiting for that gateway to exit is a circular wait::

           gateway  waits on all in-flight work units (#77184)
             └─ cron agent session waits on the `hermes update` process
                  └─ `hermes update` waits on the gateway to exit  ← back to A

       The wedged-loop probe cannot break it: the cron session posts
       activity every ~180s (process-tool poll return), so it is "actively
       waiting forever" and never marked wedged — the gateway burns the full
       force-drain cap (1800s) before killing its own updater's session.
       Fire-and-forget instead: signal the restart and return immediately;
       the gateway's own restart flow completes once THIS process (and
       therefore the cron work unit holding it) exits.
    2. **Event loop provably wedged** (#81642) — SIGUSR1 can never drain it;
       bounded escalation (SIGTERM grace → SIGKILL) instead of burning the
       full drain budget.
    3. **Live, out-of-tree gateway** — the normal graceful SIGUSR1 drain,
       waiting up to ``drain_budget`` for the process to exit (unchanged,
       including the #86684 cron floor).

    Returns True when the gateway was signalled/stopped successfully.
    """
    from hermes_cli.gateway import (
        GATEWAY_LOOP_WEDGED,
        _escalate_wedged_gateway,
        _graceful_restart_via_sigusr1,
        _is_pid_ancestor_of_current_process,
        _request_gateway_self_restart,
        probe_gateway_loop_liveness,
    )

    if _is_pid_ancestor_of_current_process(pid):
        print(
            f"  → {label}: update is running inside this gateway's "
            "process tree — signalling restart and letting the gateway "
            "drain itself (avoids the cron-update deadlock, #100179)"
        )
        return _request_gateway_self_restart(pid)
    if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        print(
            f"  ⚠ {label}: gateway event loop is unresponsive — "
            "skipping drain, forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        return True
    print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
    return _graceful_restart_via_sigusr1(pid, drain_timeout=drain_budget)
