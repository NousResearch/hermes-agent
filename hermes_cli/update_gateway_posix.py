"""systemd and launchd gateway restart coordination.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""



def _for_each_systemd_gateway_unit(
    list_units_stdout: str,
    *,
    process_unit,
    on_unit_timeout,
) -> None:
    """Process each ``hermes-gateway*.service``/``hermes-serve*.service`` unit
    from ``systemctl list-units``.

    ``subprocess.TimeoutExpired`` raised by ``process_unit`` is isolated to
    that unit via ``on_unit_timeout`` so one wedged systemctl call cannot
    abort the rest of the fleet (#68523).
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service"):
            continue
        # list-units is already pattern-filtered, but keep the name gate so a
        # stray non-gateway/serve line cannot enter the restart path.
        # ``unit.startswith("hermes-serve")`` alone would also accept the
        # unrelated ``hermes-server.service`` — require the exact base unit
        # or the hyphenated profile family instead (review on #83595).
        if not (
            unit == "hermes-gateway.service"
            or unit.startswith("hermes-gateway-")
            or unit == "hermes-serve.service"
            or unit.startswith("hermes-serve-")
        ):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)


def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` units run ``gateway/run.py``, which installs the
    SIGUSR1 handler. ``hermes-serve*`` units (#83438) don't, so sending them
    SIGUSR1 would just invoke the default terminate action and burn the full
    drain budget waiting for an exit that was never graceful — go straight to
    the blunt ``systemctl restart`` path for those instead.

    Uses the same strict exact/hyphenated shape as the unit-name gate in
    ``_for_each_systemd_gateway_unit`` so a hypothetical near-prefix unit
    (``hermes-gateway-helper`` is fine — profile units are
    ``hermes-gateway-<profile>`` — but ``hermes-gatewayd``-style names are
    not) can't be sent a SIGUSR1 it doesn't handle.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    from hermes_cli.gateway import is_macos

    if not failed_units:
        return
    # Preserve discovery order while de-duplicating.
    seen = set()
    ordered = []
    for name in failed_units:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    if is_macos():
        # A launchd label reaches this list when launchd was not supervising a
        # live process after the restart (#88848), so the unit is not merely
        # stale — it is very likely deregistered, and `launchctl kickstart`
        # cannot revive a job launchd no longer knows about.
        print("  Listed services may be deregistered from launchd, or still")
        print("  running pre-update code (mixed sys.modules). Recover with:")
        print("    hermes gateway status")
        print("    launchctl list | grep <label>")
        print("    launchctl bootstrap gui/$(id -u) "
              "~/Library/LaunchAgents/<label>.plist")
        return
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    if any(not name.startswith("ai.hermes.") for name in ordered):
        print("    systemctl --user restart <unit>   # user-scope")
        print("    sudo systemctl restart <unit>     # system-scope")
    if any(name.startswith("ai.hermes.") for name in ordered):
        print("    launchctl kickstart -k gui/$UID/<label>   # macOS (or user/$UID)")


def _restart_launchd_gateway_after_update(
    *, supervision_verify: bool = True
) -> tuple[list, list]:
    """Restart the invoking profile's launchd gateway after an update.

    #74973 (salvage #75021 by @jeff-mettel): the restart used to be gated on
    ``launchctl list <label>`` exiting 0. A *booted-out* job — plist present,
    definition deregistered from launchd (crashed helper, manual bootout,
    failed prior update) — fails that check, so the whole branch silently
    skipped: no restart, no message, ``KeepAlive`` unable to revive a
    definition launchd no longer knows, and the update still printed
    "Update complete!". ``launchctl list`` is also session-scoped and can
    exit non-zero while the job is alive in its gui/user domain, so it is
    not a reliable classifier at all.

    The fix performs NO list-based classification: when the plist exists,
    ``launchd_restart()`` always runs — it drains a live PID, kickstarts
    with ``-k``, and owns the bootout/bootstrap/kickstart ladder for the
    genuinely unloaded state. Every failure path is loud and names the
    manual recovery command.

    Returns ``(restarted_labels, failed_labels)``. With
    ``supervision_verify`` (the update path), success additionally requires
    launchd reporting a fresh supervised PID (#88848 — "the call returned"
    is not "the gateway is supervised").
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        wait_for_launchd_gateway_supervision,
    )

    current_label = get_launchd_label()
    try:
        if not get_launchd_plist_path().exists():
            return [], []  # not a launchd install — nothing to do or warn
        try:
            launchd_restart()
        except subprocess.CalledProcessError as e:
            stderr = (getattr(e, "stderr", "") or "").strip()
            print(
                f"  ⚠ Gateway restart failed: {stderr}\n"
                "    The gateway may be DOWN on pre-update code. "
                "Recover manually: hermes gateway restart"
            )
            return [], [current_label]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # A plist exists, so a gateway is SUPPOSED to be supervised here —
        # a broken/missing/wedged launchctl is not proof nothing needs
        # restarting. The old code `pass`ed here (#74973's second silent
        # variant); count it and tell the operator.
        print(
            "  ⚠ Could not restart the gateway "
            f"({e.__class__.__name__}: {e}).\n"
            "    Recover manually: hermes gateway restart"
        )
        return [], [current_label]

    if not supervision_verify:
        return [current_label], []

    # launchd_restart() returning is only "restart REQUESTED" — the
    # self-restart branch hands work to the running gateway, a plist reload
    # to a detached helper; both asynchronous. A helper that dies before its
    # first bootstrap (#88848), or a bootstrap that exits 0 without
    # registering (measured on macOS 26.6.1), otherwise reaches "Update
    # complete!" with nothing supervising the gateway. Verified
    # domain-agnostically (a domain locate fails on macOS-26 hosts whose
    # per-user domains reject service management).
    if wait_for_launchd_gateway_supervision(label=current_label):
        return [current_label], []
    print(
        f"  ✗ {current_label} restarted but launchd is not supervising it.\n"
        "    Check logs, then: hermes gateway restart"
    )
    return [], [current_label]


def _restart_macos_launchd_gateways(
    restarted_services: list,
    failed_or_stale_units: list,
    drain_budget: float,
) -> None:
    """Restart every launchd-managed gateway after an update (macOS).

    The code update (git pull) is shared across all profiles, so every
    ``ai.hermes.gateway*`` LaunchAgent must reload it — restarting only the
    invoking profile's service leaves siblings on pre-update ``sys.modules``
    until their next agent turn imports a symbol the old module generation
    doesn't have (#41403).  Parity with the systemd fleet path.

    The invoking profile keeps the existing ``launchd_restart()`` treatment
    (self-restart request → graceful drain → kickstart).  Siblings get the
    same drain-first sequence, with their launchd domain resolved per label:
    a sibling bootstrapped in the other supported domain (``gui/<uid>`` vs
    ``user/<uid>``) must not be kickstarted in the current profile's domain.
    ``subprocess.TimeoutExpired`` is isolated per label so one wedged
    launchctl call cannot leave the rest of the fleet on old code (#68523).
    """
    from hermes_cli.gateway import (
        get_launchd_label,
        get_launchd_plist_path,
        launchd_restart,
        launchd_gateway_labels_for_install,
        _graceful_restart_via_sigusr1,
        _launchd_kickstart,
        _launchd_service_registered,
        _locate_launchd_gateway_service,
        _wait_for_launchd_service_pid,
        wait_for_launchd_gateway_supervision,
    )

    # --- Current profile: unchanged single-service path ---------------------
    _restarted, _failed = _restart_launchd_gateway_after_update(
        supervision_verify=True
    )
    restarted_services.extend(_restarted)
    failed_or_stale_units.extend(_failed)
    current_label = get_launchd_label()

    # --- Sibling profiles ---------------------------------------------------
    for label in launchd_gateway_labels_for_install():
        if label == current_label:
            continue
        try:
            # Locate = liveness + domain in one domain-explicit probe; the
            # kickstart and fresh-PID verification below reuse the located
            # domain, so a sibling in the other gui/user domain can never be
            # probed in one domain and restarted in another.
            domain, old_pid = _locate_launchd_gateway_service(label)
            if domain is None:
                # Installed but not bootstrapped (stopped/uninstalled
                # mid-way) — nothing is running old code here.
                continue
            graceful_ok = False
            if old_pid is not None and old_pid > 0:
                print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
                graceful_ok = _graceful_restart_via_sigusr1(
                    old_pid, drain_timeout=drain_budget
                )
            if graceful_ok and _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=10.0, domain=domain
            ):
                # Unconditional KeepAlive already respawned it on the new
                # code — a hard kickstart now would kill the fresh process.
                restarted_services.append(label)
                continue
            try:
                _launchd_kickstart(label, domain)
            except subprocess.CalledProcessError as e:
                stderr = (getattr(e, "stderr", "") or "").strip()
                failed_or_stale_units.append(label)
                print(
                    f"  ⚠ Failed to restart {label}: {stderr}\n"
                    f"    Recover manually: launchctl kickstart -k {domain}/{label}"
                )
                continue
            if _wait_for_launchd_service_pid(
                label, old_pid=old_pid, timeout=15.0, domain=domain
            ):
                restarted_services.append(label)
            else:
                failed_or_stale_units.append(label)
                print(
                    f"  ✗ {label} failed to come back after restart.\n"
                    f"    Check logs, then: launchctl kickstart -k {domain}/{label}"
                )
        except subprocess.TimeoutExpired:
            failed_or_stale_units.append(label)
            print(
                f"  ⚠ launchctl timed out restarting {label}; "
                "continuing with remaining gateways"
            )
