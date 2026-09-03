"""Incomplete-update markers and fleet restart reconciliation.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from pathlib import Path


def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles so no banner
    reports a stale "commits behind" count after a successful update.

    The git repo is shared across profiles — when one profile runs
    ``hermes update``, every profile is now current.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)
    from hermes_constants import get_default_hermes_root

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass


def _write_marker_file(path: Path, *, label: str) -> None:
    """Drop an update-recovery breadcrumb. Never raises."""
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping %s marker under pytest (live checkout)", label)
        return
    try:
        path.write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write %s marker: %s", label, exc)


def _write_update_incomplete_marker() -> None:
    """Drop the interrupted core-install breadcrumb. Never raises."""
    _write_marker_file(_m()._update_marker_path(), label="update-incomplete")


def _write_lazy_refresh_incomplete_marker() -> None:
    """Drop the interrupted lazy-refresh breadcrumb. Never raises."""
    _write_marker_file(_m()._lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


_FLEET_RESTART_PENDING_NAME = "fleet_restart_pending"


def _fleet_restart_pending_marker_path() -> Path:
    """HERMES_HOME breadcrumb for a pull that has not yet restarted the fleet."""
    return get_hermes_home() / _FLEET_RESTART_PENDING_NAME


def _write_fleet_restart_pending_marker(*, expected_sha: str = "") -> None:
    """Drop the pull→restart obligation breadcrumb. Never raises."""
    path = _fleet_restart_pending_marker_path()
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping fleet-restart-pending marker under pytest (live checkout)")
        return
    try:
        lines = [f"started={_time.time()}", f"pid={os.getpid()}"]
        if expected_sha:
            lines.append(f"expected_sha={expected_sha}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write fleet-restart-pending marker: %s", exc)


def _clear_fleet_restart_pending_marker() -> None:
    """Remove the pull→restart obligation breadcrumb. Never raises."""
    _m()._clear_marker_file(
        _fleet_restart_pending_marker_path(), label="fleet-restart-pending"
    )


def _current_checkout_sha() -> str | None:
    """Current on-disk checkout HEAD, or None if it cannot be resolved."""
    try:
        from hermes_cli.build_info import get_code_identity

        sha = (get_code_identity(refresh=True) or {}).get("sha")
        return str(sha) if sha else None
    except Exception:
        return _capture_head_sha(["git"], _m().PROJECT_ROOT)


def _receipt_looks_unfinished(receipt: dict) -> bool:
    """True when *receipt* is from an update that did not finish cleanly."""
    if receipt.get("stop_reason"):
        return True
    exit_code = receipt.get("exit_code")
    if exit_code not in (0, None):
        return True
    outcome = receipt.get("outcome")
    if outcome in ("failed", "partial", "running"):
        return True
    gateway_restart = receipt.get("gateway_restart")
    if isinstance(gateway_restart, dict) and gateway_restart.get("incomplete"):
        return True
    return False


def _receipt_reports_stale_runtime(expected_sha: str | None = None) -> bool:
    """True when ``update_receipts/latest.json`` records a runtime SHA skew.

    ``plan.runtimes[].code_sha`` is captured *before* the pull of that run,
    so a successful update's receipt always shows pre-update runtime SHAs.
    Those must not retrigger a restart on the next invocation. Use the
    post-restart ``fleet`` matrix when present; fall back to the plan only
    for an unfinished receipt (interrupt / failed / incomplete restart) —
    the #95294 smoking-gun shape.
    """
    try:
        from hermes_cli.update_receipt import read_latest_receipt

        receipt = read_latest_receipt()
    except Exception:
        receipt = None
    if not isinstance(receipt, dict):
        return False
    if not expected_sha:
        expected_sha = _current_checkout_sha()
    if not expected_sha:
        return False

    def _sha_mismatch(code_sha) -> bool:
        return bool(code_sha) and str(code_sha) != str(expected_sha)

    fleet = receipt.get("fleet")
    if isinstance(fleet, list) and fleet:
        for entry in fleet:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "stale":
                return True
            if _sha_mismatch(entry.get("code_sha")):
                return True
        return False

    if not _receipt_looks_unfinished(receipt):
        return False
    plan = receipt.get("plan")
    if not isinstance(plan, dict):
        return False
    for runtime in plan.get("runtimes") or []:
        if isinstance(runtime, dict) and _sha_mismatch(runtime.get("code_sha")):
            return True
    return False


def _pending_fleet_restart_needed() -> bool:
    """True when a prior pull still owes the fleet a restart (#95294)."""
    try:
        if _fleet_restart_pending_marker_path().is_file():
            return True
    except OSError:
        pass
    return _receipt_reports_stale_runtime()


def _warn_pending_fleet_restart(*, startup: bool = False) -> None:
    """Print the specific interrupted-update fleet-restart warning."""
    stream = sys.stderr if startup else sys.stdout
    print(
        "⚠ A previous `hermes update` pulled new code but did not "
        "restart running gateways.",
        file=stream,
    )
    print(
        "  Gateways may still be serving pre-update modules (mixed sys.modules).",
        file=stream,
    )
    if startup:
        print(
            "  Run `hermes update` or `hermes gateway restart`.",
            file=stream,
        )


def _warn_pending_fleet_restart_on_startup() -> None:
    """Cheap CLI-startup hint. Never restarts; never raises."""
    try:
        if not _pending_fleet_restart_needed():
            return
        _warn_pending_fleet_restart(startup=True)
    except Exception:
        pass


def _restart_systemd_gateway_units_best_effort(failed: list) -> None:
    """Best-effort ``systemctl restart`` of every hermes-gateway/serve unit."""
    for scope, scope_cmd in (
        ("user", ["systemctl", "--user"]),
        ("system", ["systemctl"]),
    ):
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
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue

        def process_unit(svc_name: str, _scope=scope, _cmd=scope_cmd) -> None:
            restart_cmd = list(_cmd) + ["--no-ask-password", "restart", svc_name]
            if (
                _scope == "system"
                and hasattr(os, "geteuid")
                and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
            ):
                restart_cmd = ["sudo", "-n"] + restart_cmd
            subprocess.run(
                restart_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

        def on_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
            failed.append(svc_name)

        _for_each_systemd_gateway_unit(
            result.stdout,
            process_unit=process_unit,
            on_unit_timeout=on_timeout,
        )


def _run_pending_fleet_restart() -> bool:
    """Catch-up restart for gateways left on pre-update code (#95294).

    Returns True when restart completed or no services were running.
    Returns False if restart was incomplete. Never raises.
    """
    print("→ Restarting gateways left on pre-update code...")
    try:
        _m()._purge_stale_hermes_modules()
    except Exception:
        pass
    try:
        from hermes_cli.gateway import (
            find_gateway_pids,
            is_macos,
            is_windows,
            kill_gateway_processes,
            supports_systemd_services,
            _wait_for_gateway_exit,
        )
    except Exception as exc:
        _warn_gateway_restart_phase_aborted(exc, None)
        return False

    try:
        pids = list(find_gateway_pids(all_profiles=True))
    except Exception as exc:
        logger.debug("Pending fleet restart: gateway probe failed: %s", exc)
        pids = None

    if pids == []:
        print("  ✓ No running gateways — nothing to restart.")
        return True

    failed: list = []
    try:
        if supports_systemd_services():
            _restart_systemd_gateway_units_best_effort(failed)
        if is_macos():
            restarted: list = []
            try:
                _restart_macos_launchd_gateways(restarted, failed, 45.0)
            except Exception as exc:
                logger.debug("Pending fleet restart: launchd failed: %s", exc)
                failed.append("launchd")
        if is_windows():
            try:
                from hermes_cli import gateway_windows

                if gateway_windows.is_installed():
                    gateway_windows.restart()
            except Exception as exc:
                logger.debug("Pending fleet restart: Windows failed: %s", exc)
                failed.append("windows-gateway")
        leftover: list = []
        try:
            leftover = list(find_gateway_pids(all_profiles=True))
        except Exception:
            leftover = list(pids or [])
        if leftover:
            try:
                kill_gateway_processes(all_profiles=True)
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
            except Exception as exc:
                logger.debug("Pending fleet restart: PID stop failed: %s", exc)
        if failed:
            _warn_incomplete_gateway_fleet_restart(failed)
            return False
        print("  ✓ Pending fleet restart completed.")
        return True
    except Exception as exc:
        surviving = None
        try:
            surviving = list(find_gateway_pids(all_profiles=True))
        except Exception:
            surviving = pids
        _warn_gateway_restart_phase_aborted(exc, surviving)
        return False


def _apply_pending_fleet_restart_catchup() -> None:
    """On an already-up-to-date ``hermes update``, finish a skipped restart.

    No-op when nothing is pending. Exits 1 when the catch-up restart is
    incomplete so automation does not treat the fleet as healthy.
    """
    if not _pending_fleet_restart_needed():
        return
    print()
    _warn_pending_fleet_restart()
    print("→ Running the pending fleet restart...")
    if _run_pending_fleet_restart():
        _clear_fleet_restart_pending_marker()
        return
    print("  ⚠ Fleet restart incomplete. Recover with: hermes gateway restart")
    sys.exit(1)


def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    Returns ``None`` when the answer cannot be determined — most importantly
    when ``hermes_cli.gateway`` itself no longer imports, which is one of the
    ways the restart phase aborts in the first place (the update replaced the
    checkout under a process that already loaded the old modules). ``None`` and
    a non-empty list are both treated as "assume stale" by the caller; only a
    positive empty result is proof that nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids

        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None


_FRESH_RESTART_SUPERVISORS = frozenset({"systemd", "launchd", "service", "s6"})


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label to a profile.

    Profile names must not be matched as substrings: ``foo`` must not claim
    that ``hermes-gateway-foobar.service`` was already restarted.  These are
    the service/label shapes produced by the existing systemd, launchd, and
    s6 lifecycle implementations.
    """
    name = str(service).removesuffix(".service")
    if profile == "default":
        return name in {
            "hermes-gateway",
            "ai.hermes.gateway",
            "gateway",
            "gateway-default",
        }
    return name in {
        f"hermes-gateway-{profile}",
        f"ai.hermes.gateway-{profile}",
        f"gateway-{profile}",
    }


def _gateway_recovery_partition(
    plan, *, skip_profiles: set[str] | None = None
) -> tuple[dict[str, str], list[dict]]:
    """Partition pre-update runtimes into fresh-restart candidates and skips.

    The update inventory is captured before the checkout changes.  It is the
    only safe source here: re-importing ``hermes_cli.gateway`` in the failing
    interpreter is exactly what can raise the original ``ImportError``.

    Returns ``(candidates, skipped)`` where ``candidates`` maps profile →
    supervisor for supervised gateway runtimes the fresh process may restart,
    and ``skipped`` lists every other inventoried runtime the recovery pass
    deliberately does NOT touch *through a profile command*, each with an
    explicit reason.  Nothing from the spawn ledger may vanish silently:
    manual gateways have no relaunch authority, and serve/dashboard runtimes
    (the ``update_inventory`` serve collector) have no per-profile relaunch
    command at all.

    A ``skipped`` serve/dashboard entry is NOT a statement that the runtime is
    unrecoverable.  The fresh child runs a separate ``hermes-serve*`` systemd
    pass that enumerates units from systemd rather than from this inventory,
    precisely because the ledger collector cannot classify a systemd-launched
    ``hermes serve``: it records no spawner, so it reads as ``manual-serve``.
    Whatever that unit pass does not reconcile is caught afterwards by
    :func:`_surviving_pre_update_serve_runtimes` (#92145).
    """
    skip_profiles = skip_profiles or set()
    candidates: dict[str, str] = {}
    skipped: list[dict] = []
    try:
        for runtime in getattr(plan, "runtimes", ()) or ():
            kind = getattr(runtime, "kind", None)
            profile = getattr(runtime, "profile", None)
            supervisor = getattr(runtime, "supervisor", None)
            if not isinstance(profile, str) or not profile:
                continue
            if kind == "gateway":
                if profile in skip_profiles:
                    continue
                if supervisor in _FRESH_RESTART_SUPERVISORS:
                    candidates.setdefault(profile, str(supervisor))
                else:
                    skipped.append(
                        {
                            "profile": profile,
                            "kind": "gateway",
                            "supervisor": str(supervisor),
                            "reason": (
                                "manual gateway has no supervisor relaunch"
                                " authority; left running for explicit operator"
                                " restart"
                            ),
                        }
                    )
            elif kind in ("serve", "dashboard"):
                if supervisor == "desktop":
                    reason = (
                        "desktop app owns and respawns this serve backend;"
                        " the recovery pass must not restart it out from under"
                        " its supervisor"
                    )
                else:
                    # NOT a claim that no supervisor exists. The ledger
                    # collector derives serve/dashboard supervisors from
                    # spawner liveness alone, and a systemd-launched
                    # `hermes serve` sets neither HERMES_SPAWN nor
                    # HERMES_PARENT_PID, so it lands here as "manual-serve".
                    # Unit-backed serve runtimes are recovered by the
                    # fresh child's systemd pass, which enumerates
                    # `hermes-serve*` from systemd instead of trusting this
                    # classification; survivors are reported afterwards by
                    # _surviving_pre_update_serve_runtimes (#92145).
                    reason = (
                        "no per-profile relaunch command reaches a serve/"
                        "dashboard runtime; recovered by the fresh systemd"
                        " unit pass when it owns a hermes-serve* unit, else"
                        " left running for explicit operator restart"
                    )
                skipped.append(
                    {
                        "profile": profile,
                        "kind": str(kind),
                        "supervisor": str(supervisor),
                        "reason": reason,
                    }
                )
    except Exception as exc:
        logger.debug("Could not prepare fresh gateway restart profiles: %s", exc)
    return candidates, skipped


def _gateway_restart_recovery_profiles(
    plan, *, skip_profiles: set[str] | None = None
) -> list[str]:
    """Return supervised gateway profiles that a fresh process may restart."""
    candidates, _ = _gateway_recovery_partition(plan, skip_profiles=skip_profiles)
    return sorted(candidates)


def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    Issue #78574: the gateway auto-restart phase was wrapped in a blanket
    ``except Exception`` that only logged at debug level, so an early failure
    (e.g. importing ``hermes_cli.gateway`` from the freshly pulled checkout)
    erased every drain/restart line from the update output. The update still
    printed "Update complete!" and exited 0 while the running gateway kept
    serving pre-update modules against replaced source files — the next turn
    died with an ImportError.
    """
    print()
    print(f"⚠ Update incomplete — gateway auto-restart failed: {exc}")
    if pids:
        listed = ", ".join(str(pid) for pid in pids)
        print(f"  Gateway process(es) still running pre-update code: {listed}")
    else:
        print("  Any gateway still running is serving pre-update code")
        print("  (mixed sys.modules) against the updated checkout.")
    print("  Restart it manually, then verify:")
    print("    hermes gateway restart")
    print("    hermes gateway status")
