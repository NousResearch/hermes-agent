"""Gateway fleet restart + post-update verification for ``hermes update``.

Split out of ``hermes_cli/update_cmd.py``; every name is re-imported there so
``hermes_cli.update_cmd.<name>`` keeps resolving/monkeypatching. Origin helpers are
imported lazily inside each function (no import cycle; test patches stay effective).
"""

import logging
from contextlib import suppress
import os
import subprocess
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

from hermes_cli.update_cmd_common import _best_effort

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

# Under HERMES_HOME (not next to the venv): records the fleet-restart obligation
# after a pull advanced HEAD; cleared only when the restart completes or nothing ran.
# The existing ``.update-incomplete`` / ``.lazy-refresh-incomplete`` markers gate dependency/venv repair;
# this one is the fleet-restart obligation after a git pull that advanced HEAD (#95294).
_FLEET_RESTART_PENDING_NAME = "fleet_restart_pending"

_FRESH_RESTART_SUPERVISORS = frozenset({"systemd", "launchd", "service", "s6"})

_SYSTEMD_SCOPES = (("user", ["systemctl", "--user"]), ("system", ["systemctl"]))
_LIST_GATEWAY_UNITS = ["list-units", "hermes-gateway*", "hermes-serve*", "--plain", "--no-legend", "--no-pager"]


def _write_gateway_update_exit_code(ok: bool) -> None:
    from hermes_cli.update_cmd import get_hermes_home
    path = get_hermes_home() / ".update_exit_code"
    with suppress(OSError):
        path.write_text("0" if ok else "1", encoding="utf-8")


def _fleet_restart_pending_marker_path() -> Path:
    """HERMES_HOME breadcrumb for a pull that has not yet restarted the fleet."""
    from hermes_cli.update_cmd import get_hermes_home
    return get_hermes_home() / _FLEET_RESTART_PENDING_NAME


def _write_fleet_restart_pending_marker(*, expected_sha: str = "") -> None:
    """Drop the pull→restart obligation breadcrumb. Never raises."""
    from hermes_cli.update_cmd import _m
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
    from hermes_cli.update_cmd import _m
    _m()._clear_marker_file(_fleet_restart_pending_marker_path(), label="fleet-restart-pending")


def _current_checkout_sha() -> str | None:
    """Current on-disk checkout HEAD, or None if it cannot be resolved."""
    from hermes_cli.update_cmd import _capture_head_sha, _m
    try:
        from hermes_cli.build_info import get_code_identity
        sha = (get_code_identity(refresh=True) or {}).get("sha")
        return str(sha) if sha else None
    except Exception:
        return _capture_head_sha(["git"], _m().PROJECT_ROOT)


def _receipt_looks_unfinished(receipt: dict) -> bool:
    """True when *receipt* is from an update that did not finish cleanly."""
    gateway_restart = receipt.get("gateway_restart")
    return bool(
        receipt.get("stop_reason")
        or receipt.get("exit_code") not in (0, None)
        or receipt.get("outcome") in ("failed", "partial", "running")
        or (isinstance(gateway_restart, dict) and gateway_restart.get("incomplete"))
    )


def _receipt_reports_stale_runtime(expected_sha: str | None = None) -> bool:
    """True when ``update_receipts/latest.json`` records a runtime SHA skew.

    Prefer the post-restart ``fleet`` matrix. ``plan.runtimes[].code_sha`` is captured
    *before* the pull, so a finished update's plan always looks stale and must not
    retrigger a restart; consult it only for an unfinished receipt.

    See #95294.
    """
    from hermes_cli.update_cmd import _current_checkout_sha
    try:
        from hermes_cli.update_receipt import read_latest_receipt
        receipt = read_latest_receipt()
    except Exception:
        receipt = None
    if not isinstance(receipt, dict):
        return False
    expected_sha = expected_sha or _current_checkout_sha()
    if not expected_sha:
        return False

    def _sha_mismatch(code_sha) -> bool:
        return bool(code_sha) and str(code_sha) != str(expected_sha)

    fleet = receipt.get("fleet")
    if isinstance(fleet, list) and fleet:
        return any(
            isinstance(entry, dict)
            and (entry.get("state") == "stale" or _sha_mismatch(entry.get("code_sha")))
            for entry in fleet
        )

    if not _receipt_looks_unfinished(receipt):
        return False

    # Guard (#98022): when the receipt is from an interrupted run but the
    # code that was *pulled* in that run (receipt["post_update"]["sha"])
    # already matches the current checkout, the fleet is already on the
    # right code and no restart is needed. Without this guard an interrupted
    # receipt triggers a catchup restart on every subsequent `hermes update`
    # call because `plan.runtimes[].code_sha` always contains the *pre-pull*
    # SHAs which will never equal the post-pull checkout.
    post_update = receipt.get("post_update")
    if isinstance(post_update, dict):
        pulled_sha = post_update.get("sha") or post_update.get("code_sha")
        if pulled_sha and str(pulled_sha) == str(expected_sha):
            return False

    plan = receipt.get("plan")
    if not isinstance(plan, dict):
        return False
    return any(
        isinstance(runtime, dict) and _sha_mismatch(runtime.get("code_sha"))
        for runtime in plan.get("runtimes") or []
    )


def _pending_fleet_restart_needed() -> bool:
    """True when a prior pull still owes the fleet a restart.

    See #95294.
    """
    with suppress(OSError):
        if _fleet_restart_pending_marker_path().is_file():
            return True
    return _receipt_reports_stale_runtime()


def _warn_pending_fleet_restart(*, startup: bool = False) -> None:
    """Print the specific interrupted-update fleet-restart warning."""
    stream = sys.stderr if startup else sys.stdout
    print("⚠ A previous `hermes update` pulled new code but did not restart running gateways.", file=stream)
    print("  Gateways may still be serving pre-update modules (mixed sys.modules).", file=stream)
    if startup:
        print("  Run `hermes update` or `hermes gateway restart`.", file=stream)


def _warn_pending_fleet_restart_on_startup() -> None:
    """Cheap CLI-startup hint. Never restarts; never raises."""
    with suppress(Exception):
        if _pending_fleet_restart_needed():
            _warn_pending_fleet_restart(startup=True)


def _systemd_gateway_unit_listings(on_list_timeout=None):
    """Yield ``(scope, scope_cmd, list-units CompletedProcess)`` per systemd scope that answered.

    A missing systemctl skips the scope silently; a listing timeout skips it after
    ``on_list_timeout(scope, exc)`` (when given) so the other scope is still processed.
    """
    for scope, scope_cmd in _SYSTEMD_SCOPES:
        try:
            result = _systemctl(scope_cmd + _LIST_GATEWAY_UNITS, timeout=10)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired as exc:
            if on_list_timeout is not None:
                on_list_timeout(scope, exc)
            continue
        yield scope, scope_cmd, result


def _needs_sudo(scope: str) -> bool:
    return (
        scope == "system"
        and hasattr(os, "geteuid")
        and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
    )


def _restart_systemd_gateway_units_best_effort(failed: list) -> None:
    """Best-effort ``systemctl restart`` of every hermes-gateway/serve unit."""
    for scope, scope_cmd, result in _systemd_gateway_unit_listings():
        if result.returncode != 0:
            continue

        def process_unit(svc_name: str, _scope=scope, _cmd=scope_cmd) -> None:
            restart_cmd = list(_cmd) + ["--no-ask-password", "restart", svc_name]
            if _needs_sudo(_scope):
                restart_cmd = ["sudo", "-n"] + restart_cmd
            _systemctl(restart_cmd, timeout=30)

        _for_each_systemd_gateway_unit(
            result.stdout,
            process_unit=process_unit,
            on_unit_timeout=lambda svc_name, exc: failed.append(svc_name),
        )


def _run_pending_fleet_restart() -> bool:
    """Catch-up restart for gateways left on pre-update code. Never raises.

    True when the restart completed or nothing was running; False if incomplete.

    See #95294.
    """
    from hermes_cli.update_cmd import _m
    print("→ Restarting gateways left on pre-update code...")
    with suppress(Exception):
        _m()._purge_stale_hermes_modules()
    # Warn if legacy Hermes gateway unit files are still installed. When both hermes.service (from a
    # pre-rename install) and the current hermes-gateway.service are enabled, they SIGTERM-fight for the
    # same bot token (see PR #11909). Flagging here means every `hermes update` surfaces the issue until the
    # user migrates.
    try:
        from hermes_cli.gateway import (
            find_gateway_pids, is_macos, is_windows, kill_gateway_processes, supports_systemd_services,
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
        # --- Systemd services (Linux) --- Discover all hermes-gateway* units (default + profiles) plus
        # hermes-serve* units (the Desktop app's backend, #83438).
        if supports_systemd_services():
            _restart_systemd_gateway_units_best_effort(failed)
        # --- Launchd services (macOS) --- Restart EVERY ai.hermes.gateway* LaunchAgent, not only the
        # invoking profile's — parity with the systemd branch above (#41403). Per-label TimeoutExpired
        # isolation happens inside.
        if is_macos():
            try:
                _restart_macos_launchd_gateways([], failed, 45.0)
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
        try:
            leftover = list(find_gateway_pids(all_profiles=True))
        except Exception:
            leftover = list(pids or [])
        if leftover:
            with _best_effort('Pending fleet restart: PID stop failed: %s'):
                kill_gateway_processes(all_profiles=True)
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
        if failed:
            _warn_incomplete_gateway_fleet_restart(failed)
            return False
        print("  ✓ Pending fleet restart completed.")
        return True
    except Exception as exc:
        try:
            surviving = list(find_gateway_pids(all_profiles=True))
        except Exception:
            surviving = pids
        _warn_gateway_restart_phase_aborted(exc, surviving)
        return False


def _apply_pending_fleet_restart_catchup() -> None:
    """On an already-up-to-date ``hermes update``, finish a skipped restart.

    No-op when nothing is pending; exits 1 on incomplete catch-up so automation
    does not treat the fleet as healthy.
    """
    from hermes_cli.update_cmd import _run_pending_fleet_restart
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


def _systemctl(cmd: list, *, timeout: float):
    """Run a systemctl (or sudo systemctl) invocation, capturing utf-8 text with a timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _systemctl_reset_and_restart(manage_cmd: list, svc_name: str):
    """``reset-failed`` then ``restart``: a unit parked in failed state by systemd's own
    auto-restart can wedge a plain ``restart`` against RestartSec backoff and stay dead."""
    _systemctl(manage_cmd + ["reset-failed", svc_name], timeout=10)
    return _systemctl(manage_cmd + ["restart", svc_name], timeout=15)


def _is_hermes_gateway_unit(unit: str) -> bool:
    """Exact base unit or hyphenated profile family only: ``startswith("hermes-serve")``
    would accept ``hermes-server.service``."""
    return (
        # list-units is already pattern-filtered, but keep the name gate so a stray non-gateway/serve line
        # cannot enter the restart path. See #83595.
        unit == "hermes-gateway.service"
        or unit.startswith("hermes-gateway-")
        or unit == "hermes-serve.service"
        or unit.startswith("hermes-serve-")
    )


def _for_each_systemd_gateway_unit(list_units_stdout: str, *, process_unit, on_unit_timeout) -> None:
    """Process each hermes-gateway*/hermes-serve* unit from ``systemctl list-units``.

    ``TimeoutExpired`` from ``process_unit`` is isolated per unit via ``on_unit_timeout``
    so one wedged systemctl call cannot abort the rest of the fleet.

    See #68523.
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service") or not _is_hermes_gateway_unit(unit):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)


def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` runs ``gateway/run.py`` (the handler); SIGUSR1 would just
    kill ``hermes-serve*`` and burn the drain budget, so those go straight to the blunt
    restart. Same exact/hyphenated shape as ``_for_each_systemd_gateway_unit`` so a
    near-prefix unit like ``hermes-gatewayd`` is never signalled.

    See #83438.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    from hermes_cli.gateway import is_macos
    if not failed_units:
        return
    ordered = list(dict.fromkeys(failed_units))  # de-dup, discovery order
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    if is_macos():
        # A label lands here when launchd wasn't supervising a live process after
        # the restart — likely deregistered, which `launchctl kickstart` can't revive.
        # See #88848.
        print("  Listed services may be deregistered from launchd, or still")
        print("  running pre-update code (mixed sys.modules). Recover with:")
        print("    hermes gateway status")
        print("    launchctl list | grep <label>")
        print("    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<label>.plist")
        return
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    if any(not name.startswith("ai.hermes.") for name in ordered):
        print("    systemctl --user restart <unit>   # user-scope")
        print("    sudo systemctl restart <unit>     # system-scope")
    if any(name.startswith("ai.hermes.") for name in ordered):
        print("    launchctl kickstart -k gui/$UID/<label>   # macOS (or user/$UID)")


def _restart_launchd_gateway_after_update(*, supervision_verify: bool = True) -> tuple[list, list]:
    """Restart the invoking profile's launchd gateway after an update.

    No ``launchctl list`` gating: a booted-out job (plist present, definition
    deregistered) fails it, and it can exit non-zero while the job is alive — gating on it
    silently skipped the restart yet printed "Update complete!". When the plist exists
    ``launchd_restart()`` always runs; every failure path is loud with a manual recovery
    command. Returns ``(restarted_labels, failed_labels)``; with ``supervision_verify``
    success also requires a fresh supervised PID ("the call returned" is not "supervised").

    74973 (salvage #75021 by @jeff-mettel): the restart used to be gated on ``launchctl list <label>``
    exiting 0. A *booted-out* job — plist present, definition deregistered from launchd (crashed helper,
    manual bootout, failed prior update) — fails that check, so the whole branch silently skipped: no
    restart, no message, ``KeepAlive`` unable to revive a definition launchd no longer knows, and the update
    still printed "Update complete!".
    See #88848.
    """
    from hermes_cli.gateway import (
        get_launchd_label, get_launchd_plist_path, launchd_restart, wait_for_launchd_gateway_supervision,
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
        # A plist exists, so a gateway is SUPPOSED to be supervised; a broken/wedged
        # launchctl is not proof nothing needs restarting. Count it, tell the operator.
        print(
            # The old code `pass`ed here (#74973's second silent variant); count it and tell the operator.
            "  ⚠ Could not restart the gateway "
            f"({e.__class__.__name__}: {e}).\n"
            "    Recover manually: hermes gateway restart"
        )
        return [], [current_label]

    if not supervision_verify:
        return [current_label], []

    # launchd_restart() returning only means "restart REQUESTED" (async). A helper dying
    # before first bootstrap, or a bootstrap exiting 0 without registering (macOS 26.6.1),
    # would otherwise reach "Update complete!" unsupervised. Verified domain-agnostically:
    # domain locate fails on macOS-26 per-user domains.
    # launchd_restart() returning is only "restart REQUESTED" — the self-restart branch hands work to the
    # running gateway, a plist reload to a detached helper; both asynchronous. See #88848.
    if wait_for_launchd_gateway_supervision(label=current_label):
        return [current_label], []
    print(
        f"  ✗ {current_label} restarted but launchd is not supervising it.\n"
        "    Check logs, then: hermes gateway restart"
    )
    return [], [current_label]


def _restart_macos_launchd_gateways(restarted_services: list, failed_or_stale_units: list, drain_budget: float) -> None:
    """Restart every launchd-managed gateway after an update (macOS).

    The pull is shared across profiles, so every ``ai.hermes.gateway*`` LaunchAgent
    must reload it or siblings stay on pre-update ``sys.modules`` (systemd parity).
    Invoking profile uses ``launchd_restart()``; siblings get the same drain-first
    sequence with their domain (``gui/<uid>`` vs ``user/<uid>``) resolved per label so
    none is kickstarted in the wrong domain. ``TimeoutExpired`` is isolated per label.

    See #41403.
    The invoking profile keeps the existing ``launchd_restart()`` treatment (self-restart request → graceful
    drain → kickstart). ``subprocess.TimeoutExpired`` is isolated per label so one wedged launchctl call
    cannot leave the rest of the fleet on old code (#68523).
    """
    from hermes_cli.gateway import (
        get_launchd_label, launchd_gateway_labels_for_install, _graceful_restart_via_sigusr1, _launchd_kickstart,
        _locate_launchd_gateway_service, _wait_for_launchd_service_pid,
    )
    _restarted, _failed = _restart_launchd_gateway_after_update(supervision_verify=True)
    restarted_services.extend(_restarted)
    failed_or_stale_units.extend(_failed)
    current_label = get_launchd_label()

    for label in launchd_gateway_labels_for_install():
        if label == current_label:
            continue
        try:
            # Locate = liveness + domain in one probe; kickstart and fresh-PID checks
            # reuse that domain so a sibling is never probed in one and restarted in another.
            domain, old_pid = _locate_launchd_gateway_service(label)
            if domain is None:
                continue  # installed but not bootstrapped — nothing running old code here
            graceful_ok = False
            if old_pid is not None and old_pid > 0:
                print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
                graceful_ok = _graceful_restart_via_sigusr1(old_pid, drain_timeout=drain_budget)
            if graceful_ok and _wait_for_launchd_service_pid(label, old_pid=old_pid, timeout=10.0, domain=domain):
                # KeepAlive already respawned it on new code — a kickstart would kill it.
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
            if _wait_for_launchd_service_pid(label, old_pid=old_pid, timeout=15.0, domain=domain):
                restarted_services.append(label)
            else:
                failed_or_stale_units.append(label)
                print(
                    f"  ✗ {label} failed to come back after restart.\n"
                    f"    Check logs, then: launchctl kickstart -k {domain}/{label}"
                )
        except subprocess.TimeoutExpired:
            failed_or_stale_units.append(label)
            print(f"  ⚠ launchctl timed out restarting {label}; continuing with remaining gateways")


def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    ``None`` when undeterminable (notably ``hermes_cli.gateway`` no longer importing
    under the replaced checkout). Callers treat ``None`` and non-empty as "assume
    stale"; only a positive empty result proves nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids
        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label (systemd/launchd/s6 shapes) to a profile.

    Never substring-match: ``foo`` must not claim ``hermes-gateway-foobar.service``.
    """
    name = str(service).removesuffix(".service")
    if profile == "default":
        return name in {"hermes-gateway", "ai.hermes.gateway", "gateway", "gateway-default"}
    return name in {f"hermes-gateway-{profile}", f"ai.hermes.gateway-{profile}", f"gateway-{profile}"}


_MANUAL_GATEWAY_SKIP_REASON = (
    "manual gateway has no supervisor relaunch authority; left running for explicit operator restart"
)
_DESKTOP_SERVE_SKIP_REASON = (
    "desktop app owns and respawns this serve backend;"
    " the recovery pass must not restart it out from under its supervisor"
)
# NOT a claim that no supervisor exists: a systemd-launched serve sets neither HERMES_SPAWN
# nor HERMES_PARENT_PID ("manual-serve"). Unit-backed serves are recovered by the fresh
# child's systemd pass; survivors reported by _surviving_pre_update_serve_runtimes.
_SERVE_SKIP_REASON = (
    "no per-profile relaunch command reaches a serve/dashboard runtime; recovered by the fresh"
    " systemd unit pass when it owns a hermes-serve* unit, else left running for explicit"
    " operator restart"
)


def _gateway_recovery_partition(plan, *, skip_profiles: set[str] | None = None) -> tuple[dict[str, str], list[dict]]:
    """Partition pre-update runtimes into fresh-restart candidates and skips.

    Uses only the pre-checkout inventory: re-importing ``hermes_cli.gateway`` in the
    failing interpreter is what raises the original ``ImportError``. Returns
    ``(candidates, skipped)``: profile → supervisor for supervised gateways the fresh
    process may restart; every other inventoried runtime with an explicit reason so
    nothing vanishes silently. Skipped serve/dashboard is NOT unrecoverable: the fresh
    child's ``hermes-serve*`` systemd pass enumerates units from systemd; leftovers are
    caught by :func:`_surviving_pre_update_serve_runtimes`.
    """
    skip_profiles = skip_profiles or set()
    candidates: dict[str, str] = {}
    skipped: list[dict] = []
    with _best_effort('Could not prepare fresh gateway restart profiles: %s'):
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
                    continue
                reason = _MANUAL_GATEWAY_SKIP_REASON
            elif kind in ("serve", "dashboard"):
                reason = _DESKTOP_SERVE_SKIP_REASON if supervisor == "desktop" else _SERVE_SKIP_REASON
            else:
                continue
            skipped.append({"profile": profile, "kind": str(kind), "supervisor": str(supervisor), "reason": reason})
    return candidates, skipped


def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    Previously a blanket debug-logged ``except Exception`` erased every drain/restart
    line, so "Update complete!" exited 0 while the gateway kept serving pre-update
    modules and died on the next turn with an ImportError.

    Issue #78574: the gateway auto-restart phase was wrapped in a blanket ``except Exception`` that only
    logged at debug level, so an early failure (e.g. importing ``hermes_cli.gateway`` from the freshly
    pulled checkout) erased every drain/restart line from the update output.
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


def _drain_or_signal_gateway_for_update(pid: int, drain_budget: float, label: str) -> bool:
    """Three-way triage (shared by systemd and bare-process paths) for handing a
    running gateway over to new code. Returns True when signalled/stopped.

    1. Gateway is an ancestor of this process (auto-update cron inside the gateway
       tree): waiting is circular (gateway waits on in-flight work → cron session
       waits on update → update waits on gateway) and the 1800s force-drain cap burns.
       So fire-and-forget: signal restart and return; it completes once THIS process exits.
    2. Event loop provably wedged: SIGUSR1 can never drain it; bounded SIGTERM→SIGKILL.
    3. Live out-of-tree gateway: graceful SIGUSR1 drain up to ``drain_budget``.

    The wedged-loop probe cannot break it: the cron session posts activity every ~180s (process-tool poll
    return), so it is "actively waiting forever" and never marked wedged — the gateway burns the full
    force-drain cap (1800s) before killing its own updater's session. See #86684.
    """
    from hermes_cli.gateway import (
        GATEWAY_LOOP_WEDGED, _escalate_wedged_gateway, _graceful_restart_via_sigusr1,
        _is_pid_ancestor_of_current_process, _request_gateway_self_restart, probe_gateway_loop_liveness,
    )
    if _is_pid_ancestor_of_current_process(pid):
        print(
            f"  → {label}: update is running inside this gateway's "
            "process tree — signalling restart and letting the gateway "
            "drain itself (avoids the cron-update deadlock, #100179)"
        )
        return _request_gateway_self_restart(pid)
    if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        print(f"  ⚠ {label}: gateway event loop is unresponsive — skipping drain, forcing a bounded stop...")
        _escalate_wedged_gateway(pid)
        return True
    print(f"  → {label}: draining (up to {int(drain_budget)}s)...")
    return _graceful_restart_via_sigusr1(pid, drain_timeout=drain_budget)


def _resolve_manage_cmd(cache: dict, scope_: str, scope_cmd_: list, svc_name_: str):
    """Resolve the command prefix for manage-units verbs (None ⇒ no privilege path).

    Manage-units verbs on a *system* service trigger a polkit prompt for non-root
    users, which flashes and dies inside our captured 10-15s subprocess. Root → plain
    systemctl; else ``sudo -n`` blanket probe, then a targeted ``reset-failed`` probe
    so a least-privilege sudoers entry scoped to hermes-gateway* qualifies (idempotent
    no-op we run before every privileged restart anyway). On None the caller must SKIP
    the restart (without draining first!). ``--no-ask-password`` prevents polkit hangs.
    """
    if scope_ in cache:
        return cache[scope_]
    cmd = scope_cmd_ + ["--no-ask-password"]
    if _needs_sudo(scope_):
        sudo_cmd = ["sudo", "-n"] + cmd
        try:
            sudo_ok = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5).returncode == 0
            if not sudo_ok:
                # Blanket sudo refused — a targeted NOPASSWD sudoers entry may still work.
                sudo_ok = subprocess.run(
                    sudo_cmd + ["reset-failed", svc_name_], capture_output=True, timeout=5
                ).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            sudo_ok = False
        cmd = sudo_cmd if sudo_ok else None
    cache[scope_] = cmd
    return cmd


def _restart_one_systemd_gateway_unit(
    svc_name: str, *, scope: str, scope_cmd: list, drain_budget: float, _manage_cmd_cache: dict,
    restarted_services: list, failed_or_stale_units: list,
) -> None:
    """Restart one active systemd gateway/serve unit: graceful SIGUSR1 drain, then forced restart.

    Appends settled names to ``restarted_services`` and failures to ``failed_or_stale_units``.
    """
    check = _systemctl(scope_cmd + ["is-active", svc_name], timeout=5)
    if check.stdout.strip() != "active":
        return

    # None ⇒ no non-interactive privilege path; avoid manage-units verbs
    # entirely or polkit prompts inside the captured subprocess.
    _manage_cmd = _resolve_manage_cmd(_manage_cmd_cache, scope, scope_cmd, svc_name)

    # Graceful SIGUSR1 first so in-flight runs drain: handler → request_restart(via_service=True)
    # → drain → exit, Restart=always respawns. hermes-serve has no handler → blunt restart below.
    _main_pid = 0
    if _service_unit_supports_graceful_sigusr1_restart(svc_name):
        try:
            _show = _systemctl(scope_cmd + ["show", svc_name, "--property=MainPID", "--value"], timeout=5)
            _main_pid = int((_show.stdout or "").strip() or 0)
        except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
            _main_pid = 0

    # Three-way triage (ancestor / wedged / graceful drain).
    _graceful_ok = _main_pid > 0 and _drain_or_signal_gateway_for_update(_main_pid, drain_budget, svc_name)

    if _graceful_ok:
        # ``Restart=always`` respawns only after RestartSec (60s in our unit; dead time for a
        # voluntary restart). ``reset-failed`` + ``start`` skips it (~1-3s); if RestartSec already
        # elapsed, ``start`` is a no-op and we fall through to the poll. Needs manage-units
        # privileges; without them auto-restart still fires after RestartSec.
        if _manage_cmd is not None:
            _systemctl(_manage_cmd + ["reset-failed", svc_name], timeout=10)
            _systemctl(_manage_cmd + ["start", svc_name], timeout=15)
            if _wait_for_service_active(scope_cmd, svc_name, timeout=10.0):
                restarted_services.append(svc_name)
                return
        # Passive poll: auto-restart fires after RestartSec regardless of
        # privileges — primary when _manage_cmd is None, fallback otherwise.
        _restart_sec = _service_restart_sec(scope_cmd, svc_name, default=0.0)
        if _manage_cmd is None and _restart_sec > 5.0:
            print(
                f"  → {svc_name}: waiting for systemd "
                f"auto-restart (~{int(_restart_sec)}s; "
                "no root for an immediate restart)..."
            )
        if _wait_for_service_active(scope_cmd, svc_name, timeout=max(10.0, _restart_sec + 10.0)):
            restarted_services.append(svc_name)
            return
        # Exited but not respawned (older unit without Restart=on-failure /
        # RestartForceExitStatus=75); fall through to forced restart.
        print(f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart")

    # Forcing needs manage-units privileges; without a non-interactive path
    # polkit would prompt inside the captured subprocess — skip, instruct.
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

    # Blunt restart — only when the graceful path failed (no SIGUSR1 wiring, drain over
    # budget, restart-policy mismatch). Mirrors `hermes gateway restart` (`systemd_restart()`).
    restart = _systemctl_reset_and_restart(_manage_cmd, svc_name)
    if restart.returncode != 0:
        failed_or_stale_units.append(svc_name)
        print(f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}")
        return
    # restart returns 0 even if the new process crashes at once — verify.
    if _wait_for_service_active(scope_cmd, svc_name, timeout=10.0):
        restarted_services.append(svc_name)
        return
    # Retry once — transient startup failures (stale module cache,
    # import race) often clear; reset-failed so the retry isn't blocked.
    print(f"  ⚠ {svc_name} died after restart, retrying...")
    _systemctl_reset_and_restart(_manage_cmd, svc_name)
    if _wait_for_service_active(scope_cmd, svc_name, timeout=10.0):
        restarted_services.append(svc_name)
        print(f"  ✓ {svc_name} recovered on retry")
        return
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


def _restart_systemd_gateway_units(restarted_services, failed_or_stale_units, restarted_scoped_units, drain_budget):
    """Restart every active hermes-gateway*/hermes-serve* systemd unit (user + system).

    Settled units → ``restarted_services`` (bare) and ``restarted_scoped_units``
    (``scope/name``); failures → ``failed_or_stale_units``. Per-unit timeouts isolated.
    """
    from hermes_cli.gateway import supports_systemd_services, _ensure_user_systemd_env
    if not supports_systemd_services():
        return
    _manage_cmd_cache: dict = {}
    with suppress(Exception):
        _ensure_user_systemd_env()

    def _on_list_timeout(scope: str, exc: subprocess.TimeoutExpired) -> None:
        # Discovery timeout — skip this scope, keep the other.
        print(
            f"  ⚠ systemctl timed out listing {scope}-scope "
            f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
            f"Check the gateway with: hermes gateway status"
        )

    def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
        # Isolate to this unit; a scope-wide handler used to abort every
        # later gateway and leave the fleet on mixed code.
        failed_or_stale_units.append(svc_name)
        print(
            # See #68523.
            f"  ⚠ systemctl timed out restarting {svc_name} "
            f"({exc.cmd if exc.cmd else 'unknown command'}); "
            f"continuing with remaining gateways"
        )

    for scope, scope_cmd, result in _systemd_gateway_unit_listings(_on_list_timeout):
        # Scope-qualify this scope's additions before the next scope can add a
        # same-named unit; ``finally`` so a mid-scope abort keeps settled units.
        _scope_mark = len(restarted_services)
        try:
            _for_each_systemd_gateway_unit(
                result.stdout,
                process_unit=lambda svc_name: _restart_one_systemd_gateway_unit(
                    svc_name,
                    scope=scope,
                    scope_cmd=scope_cmd,
                    drain_budget=drain_budget,
                    _manage_cmd_cache=_manage_cmd_cache,
                    restarted_services=restarted_services,
                    failed_or_stale_units=failed_or_stale_units,
                ),
                on_unit_timeout=_on_unit_timeout,
            )
        finally:
            restarted_scoped_units.update(f"{scope}/{name}" for name in restarted_services[_scope_mark:])


@dataclass
class _GatewayRestartOutcome:
    """Restart-phase bookkeeping. ``restarted_services`` keeps bare unit names (fleet
    probe, receipt, summary read it); ``incomplete`` ⇒ a gateway may still be stale."""

    incomplete: bool
    phase_errors: list
    pre_restart_gateway_pids: "list | None"
    restarted_services: list
    failed_or_stale_units: list
    relaunched_profiles: list
    externally_supervised_profiles: list
    killed_pids: set
    #: Gateways stopped with NO successor (no profile mapping / relaunch could not be armed);
    #: the summary tells the user to restart them by hand, so the fleet probe must not expect
    #: a row for them.
    stopped_unmapped_pids: set = field(default_factory=set)

    def fleet_probe_signals(self) -> tuple:
        """``(pre_restart_pids, killed_pids)`` with the unmapped stops removed — the signals that
        legitimately predict a fleet-matrix row."""
        pre = self.pre_restart_gateway_pids
        if pre is not None:
            pre = [pid for pid in pre if pid not in self.stopped_unmapped_pids]
        return pre, self.killed_pids - self.stopped_unmapped_pids

    def record_receipt(self, **extra) -> None:
        """Best-effort ``record_gateway_restart`` from the current bookkeeping."""
        with suppress(Exception):
            from hermes_cli.update_receipt import record_gateway_restart
            record_gateway_restart(
                restarted_services=self.restarted_services, relaunched_profiles=self.relaunched_profiles,
                externally_supervised_profiles=self.externally_supervised_profiles,
                killed_pids=sorted(self.killed_pids), failed_units=self.failed_or_stale_units,
                incomplete=self.incomplete, **extra,
            )


def _restart_manual_gateways(out: _GatewayRestartOutcome, _drain_budget) -> None:
    """Drain/stop every manual (non-service) gateway and print the restart summary.

    Mutates ``out`` in place; raises so the caller's abort recovery fires.
    """
    import signal as _signal
    from hermes_cli.gateway import (
        find_gateway_pids, find_profile_gateway_processes, _prepare_profile_gateway_update_restart, _get_service_pids,
        _wait_for_gateway_exit,
    )
    # Exclude just-restarted service PIDs so we don't kill what systemd/launchd spawned.
    service_pids = _get_service_pids(all_profiles=True)
    manual_pids = find_gateway_pids(exclude_pids=service_pids, all_profiles=True)
    profile_processes = {
        proc.pid: proc
        for proc in find_profile_gateway_processes(exclude_pids=service_pids)
        if proc.pid in manual_pids
    }
    # Profile gateways we couldn't arm a relaunch for must NOT keep running stale:
    # the unmapped sweep below stops them and lists them under "Restart manually".
    # These must NOT be left running: their modules are the pre-update ones and every lazy import from here
    # on mixes versions against the new code on disk (#88654). Handing them to the unmapped sweep below
    # stops them and surfaces them in the "Stopped N manual gateway process(es) / Restart manually" summary,
    # which is the contract already used for gateways with no profile mapping.
    unrestartable_pids = set()
    for pid, proc in profile_processes.items():
        restart_mode = _prepare_profile_gateway_update_restart(proc.profile, pid)
        if restart_mode is None:
            # A bare ``continue`` here left it serving stale modules with no signal.
            print(
                f"  ⚠ {proc.profile}: could not arm an automatic "
                f"gateway restart for PID {pid} — stopping it instead "
                "so it cannot keep running pre-update code"
            )
            unrestartable_pids.add(pid)
            continue
        # SIGUSR1 drain first, SIGTERM fallback if unsupported/over budget — the watcher
        # relaunches either way. The helper announces its choice first because a silent
        # full-budget wait reads as a hung update.
        if not _drain_or_signal_gateway_for_update(pid, _drain_budget, proc.profile):
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, _signal.SIGTERM)
        # Wait ≤5s for exit: Telegram keeps the old getUpdates session ~30s; a new gateway
        # inside that window gets a 409 (_handle_polling_conflict retries, but a brief
        # wait avoids it on fast machines).
        _wait_for_gateway_exit(timeout=5.0, force_after=None)
        out.killed_pids.add(pid)
        if restart_mode == "external-supervisor":
            out.externally_supervised_profiles.append(proc.profile)
        else:
            out.relaunched_profiles.append(proc.profile)

    for pid in manual_pids:
        if pid in profile_processes and pid not in unrestartable_pids:
            continue
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, _signal.SIGTERM)
            out.killed_pids.add(pid)
            out.stopped_unmapped_pids.add(pid)

    if out.restarted_services or out.killed_pids:
        print()
        for svc in out.restarted_services:
            print(f"  ✓ Restarted {svc}")
        if out.relaunched_profiles:
            print(f"  ✓ Restarting manual gateway profile(s): {', '.join(out.relaunched_profiles)}")
        if out.externally_supervised_profiles:
            names = ", ".join(out.externally_supervised_profiles)
            print(f"  ✓ Handed gateway profile(s) back to their external supervisor: {names}")
        unmapped_count = (len(out.killed_pids) - len(out.relaunched_profiles) - len(out.externally_supervised_profiles))
        if unmapped_count:
            print(f"  → Stopped {unmapped_count} manual gateway process(es)")
            print("    Restart manually: hermes gateway run")
            if unmapped_count > 1:
                print("    (or: hermes -p <profile> gateway run  for each profile)")


def _force_kill_stuck_gateways(killed_pids) -> None:
    """Survivor sweep: gateways ignoring SIGTERM (stuck drain, blocked I/O, zombie) never
    exit, so the watcher never respawns and ImportErrors persist. Give graceful paths a
    moment, then SIGKILL remaining pre-update PIDs."""
    with _best_effort('Post-restart survivor sweep failed: %s'):
        from hermes_cli.gateway import find_gateway_pids, _get_service_pids
        # --- Post-restart survivor sweep ----------------------------- Issue #17648: some gateways ignore
        # SIGTERM (stuck drain, blocked I/O, PID dead but zombie). The detached profile watchers wait 120s
        # for the old PID to exit — if it never does, no respawn happens and the user keeps hitting
        # ImportError against a stale sys.modules.
        _time.sleep(3.0)
        _surviving = find_gateway_pids(exclude_pids=_get_service_pids(all_profiles=True), all_profiles=True)
        # Only PIDs we already tried to kill; newer ones are left alone.
        _stuck = [pid for pid in _surviving if pid in killed_pids]
        if _stuck:
            print()
            print(f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing")
            from gateway.status import get_process_start_time, terminate_pid
            for pid in _stuck:
                with suppress(ProcessLookupError, PermissionError, OSError):
                    # taskkill /T /F on Windows (no SIGKILL there), SIGKILL on POSIX.
                    terminate_pid(pid, force=True, expected_start_time=get_process_start_time(pid))
            # Let the OS reap so watchers see the exit and respawn.
            _time.sleep(1.5)


def _recover_after_restart_phase_abort(
    e, _pre_update_plan, out: _GatewayRestartOutcome, *, gateway_mode, restarted_scoped_units
) -> None:
    """Phase-abort recovery: fresh-child restart + fail-closed verdict; updates ``out`` in place."""
    from hermes_cli.update_cmd import (
        _abort_recovery_is_complete, _recover_gateway_restart_after_abort, _surviving_pre_update_serve_runtimes,
        _warn_stale_serve_runtimes, _write_gateway_update_exit_code,
    )
    logger.debug("Gateway restart during update failed: %s", e)
    out.phase_errors.append(str(e))
    # Restart output never printed: assume stale unless provably no gateway runs.
    # Empty ``_surviving`` proves safety only if nothing ran beforehand; a gone
    # pre-restart gateway was stopped without verified replacement → fail closed.
    # An exception escaping the whole phase means the drain/restart output the user relies on never printed.
    # Don't let that pass for a clean update: surface it and treat the fleet as stale unless we can
    # positively prove no gateway is running (#78574). A positive-empty ``_surviving`` is only
    # proof-of-safety when nothing was running before we touched anything. If a gateway was discovered
    # pre-restart and none survive now, it was stopped and its replacement was never verified — the same
    # fail-open contract this fix closes — so we must still fail closed on ``[]``.
    _surviving = _surviving_gateway_pids_after_failed_restart()
    _planned_gateway_profiles = {
        runtime.profile
        for runtime in getattr(_pre_update_plan, "runtimes", ()) or ()
        if getattr(runtime, "kind", None) == "gateway"
        and isinstance(getattr(runtime, "profile", None), str)
    }
    _already_restarted_profiles = set(out.relaunched_profiles) | set(out.externally_supervised_profiles)
    _already_restarted_profiles.update(
        profile
        for profile in _planned_gateway_profiles
        if any(_gateway_service_matches_profile(profile, service) for service in out.restarted_services)
    )
    _recovery_result = _recover_gateway_restart_after_abort(
        _pre_update_plan, gateway_mode=gateway_mode, skip_profiles=_already_restarted_profiles,
        skip_units=set(restarted_scoped_units),
    )
    _serve_units_failed = list((_recovery_result.get("serve_units") or {}).get("failed") or [])
    # Deliberately NOT merged into ``restarted_services`` (gateway vocabulary feeding the
    # fleet probe); serve coverage lives in the recovery result/receipt. A serve/dashboard
    # still the SAME pre-update process is live on old code (unreachable by `gateway
    # restart`): recovery may not claim success while one remains, and must never kill
    # one (manual/Desktop serves have no relaunch authority).
    _stale_runtime_rows = _surviving_pre_update_serve_runtimes(_pre_update_plan)
    _recovery_result["stale_runtimes"] = _stale_runtime_rows
    # Only systemd-VERIFIED outcomes claim coverage; a relaunch that merely exited 0
    # ("relaunch_attempted") was never observed and must not clear incomplete.
    _recovery_verified = set(_recovery_result.get("verified") or [])
    out.relaunched_profiles.extend(
        profile for profile in sorted(_recovery_verified) if profile not in out.relaunched_profiles
    )
    if _abort_recovery_is_complete(
        planned_gateway_profiles=_planned_gateway_profiles,
        covered_gateway_profiles=_already_restarted_profiles | _recovery_verified,
        recovery_result=_recovery_result,
        stale_runtime_rows=_stale_runtime_rows,
    ):
        # Fresh child is terminal; the fleet-version matrix stays the authoritative
        # read-back before success is declared.
        out.incomplete = False
    elif (
        _restart_phase_failure_is_incomplete(_surviving, out.pre_restart_gateway_pids)
        or _stale_runtime_rows
        or _serve_units_failed
    ):
        out.incomplete = True
        _warn_gateway_restart_phase_aborted(e, _surviving)
        _warn_stale_serve_runtimes(_stale_runtime_rows)
        if gateway_mode:
            _write_gateway_update_exit_code(False)
    out.record_receipt(phase_error=str(e), fresh_recovery=_recovery_result)


def _restart_gateway_fleet_after_update(_pre_update_plan, gateway_mode: bool):
    """Restart every running gateway (systemd, launchd, manual) onto the pulled code.

    Never raises: a phase abort runs fresh-child recovery and fails closed unless
    every planned gateway is verifiably covered.
    """
    from hermes_cli.update_cmd import _m, _write_gateway_update_exit_code
    # All bookkeeping is declared before the try so abort recovery and fleet reconciliation
    # can read it even if the phase raises early. ``pre_restart_gateway_pids`` stays empty
    # until we are about to stop/drain, so an early exception has nothing to fail closed on,
    # while a failure after stopping a discovered gateway fails closed on an empty survivor probe.
    out = _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[], failed_or_stale_units=[],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )
    # Scope-qualified twin (``user/hermes-serve`` vs ``system/hermes-serve`` are different
    # processes; abort recovery needs WHICH settled). Bare names stay in
    # ``restarted_services`` for the fleet probe, receipt and summary.
    # Snapshot of gateways running before we touch anything. Stays empty until we successfully import the
    # probe and are about to stop/drain — so an exception raised before we touch any gateway keeps this
    # empty (nothing to fail closed on), while a failure after we have stopped a discovered gateway lets the
    # handler fail closed on an empty survivor probe rather than reporting a clean update (#78574).
    # Declared outside the restart try/except below (and never reset to None) so it's always safe to read
    # afterwards even if that block raises before reaching its own restart bookkeeping — needed to forward
    # already-restarted units to ``_finish_dashboard_update_cleanup`` (review on #83595).
    restarted_scoped_units: set = set()

    # Purge stale cached Hermes modules FIRST: the import below loads new gateway
    # source into this pre-update interpreter, and a cached sibling missing a
    # symbol the new source expects would ImportError and abort the whole phase.
    _m()._purge_stale_hermes_modules()
    try:
        # Every gateway helper the phase needs is imported up front so a broken gateway
        # module aborts into recovery BEFORE any unit is touched.
        from hermes_cli.gateway import (  # noqa: F401
            is_macos,
            find_gateway_pids,
            find_profile_gateway_processes,
            _prepare_profile_gateway_update_restart,
            _get_service_pids,
            _wait_for_gateway_exit,
        )
        # Drain budget covers ``restart_after_turn_timeout`` and stop()'s
        # ``restart_drain_timeout`` so a gateway waiting on a turn isn't hard-killed;
        # units without SIGUSR1 wiring just time out into ``systemctl restart``.
        try:
            from hermes_cli.gateway import _get_restart_exit_wait_budget
            _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
        except Exception:
            _drain_budget = 45.0

        # Snapshot before any stop/drain so an empty survivor probe reads as "stopped
        # and never came back", not "nothing was running"; None fails closed.
        try:
            out.pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            out.pre_restart_gateway_pids = None

        _restart_systemd_gateway_units(
            out.restarted_services, out.failed_or_stale_units, restarted_scoped_units, _drain_budget
        )

        # macOS: EVERY ai.hermes.gateway* LaunchAgent (systemd parity).
        if is_macos():
            with suppress(FileNotFoundError, ImportError):
                _restart_macos_launchd_gateways(out.restarted_services, out.failed_or_stale_units, _drain_budget)

        _restart_manual_gateways(out, _drain_budget)

        if out.failed_or_stale_units:
            out.incomplete = True
            if gateway_mode:
                _write_gateway_update_exit_code(False)
        _warn_incomplete_gateway_fleet_restart(out.failed_or_stale_units)
        out.record_receipt()
        _force_kill_stuck_gateways(out.killed_pids)

    except Exception as e:
        _recover_after_restart_phase_abort(
            e, _pre_update_plan, out, gateway_mode=gateway_mode, restarted_scoped_units=restarted_scoped_units
        )

    return out


def _print_legacy_units_warning() -> None:
    """Legacy hermes.service fights hermes-gateway.service over the bot token; warn on
    every update until migrated."""
    from hermes_cli.gateway import (has_legacy_hermes_units, _find_legacy_hermes_units, supports_systemd_services)
    if not (supports_systemd_services() and has_legacy_hermes_units()):
        return
    print()
    print("⚠ Legacy Hermes gateway unit(s) detected:")
    for name, path, is_sys in _find_legacy_hermes_units():
        scope = "system" if is_sys else "user"
        print(f"    {path}  ({scope} scope)")
    print()
    print("  These pre-rename units (hermes.service) fight the current")
    print("  hermes-gateway.service for the bot token and cause SIGTERM")
    print("  flap loops. Remove them with:")
    print()
    print("    hermes gateway migrate-legacy")
    print()
    print("  (add `sudo` if any are in system scope)")


def _collect_fleet_snapshot(restart, rows_expected: bool) -> list:
    """Fleet version rows, polled over a bounded settle window when runtimes are expected.

    Gateways need time to rewrite gateway_state.json; Windows resumes DETACHED (~10s boot),
    so a single 2s sleep reported "no rows" on healthy resumes. A "down" row may be a
    detached replacement still booting: poll until none remain or the deadline passes.
    Pre-restart PIDs make a gateway stopped WITHOUT verified replacement a DOWN row (exit 1)
    instead of no row at all.
    """
    from hermes_cli.update_receipt import collect_fleet_versions
    if not rows_expected:
        return collect_fleet_versions(pre_restart_pids=restart.pre_restart_gateway_pids)
    _fleet_deadline = _time.monotonic() + 30.0
    while True:
        _time.sleep(2.0)
        snapshot = collect_fleet_versions(pre_restart_pids=restart.pre_restart_gateway_pids)
        if snapshot and not any(row.get("state") == "down" for row in snapshot):
            return snapshot
        if _time.monotonic() >= _fleet_deadline:
            return snapshot


def _verify_fleet_after_update(restart, *, _pre_update_plan, _windows_gateway_resume, node_failures, update_complete):
    """Post-restart verification: legacy-unit warning, dashboard cleanup, stale serve
    probe, fleet version matrix, plan-vs-execution reconciliation, receipt finalize.

    Exits 1 (leaving ``fleet_restart_pending`` for the next catch-up) when any gateway
    may still be stale; otherwise clears the marker.
    """
    from hermes_cli.update_cmd import (
        _finish_dashboard_update_cleanup, _m, _surviving_pre_update_serve_runtimes, _warn_stale_serve_runtimes,
    )
    with _best_effort('Legacy unit check during update failed: %s'):
        _print_legacy_units_warning()

    # Restart a managed dashboard via systemd or stop stale manual ones (raw-killing
    # a systemd-owned PID reads as clean stop and leaves the Cloudflare origin dead).
    # Failed Node refresh leaves it untouched; already-restarted units aren't redone.
    _finish_dashboard_update_cleanup(node_failures, already_restarted_units=set(restart.restarted_services))

    # Success-path twin of the abort-recovery probe: the restart phase only touches
    # units, so a unit-less `hermes serve` keeps stale sys.modules. Runs AFTER
    # dashboard cleanup so a respawned manual dashboard isn't a survivor. Rows feed
    # reconciliation (survivor → exit 1); ``None`` = probe failed, stays fail-closed.
    # Check if any pre-update serve/dashboard runtimes survived on pre-update code generations (#100479).
    # This is the SUCCESS-path twin of the abort-recovery probe above: the restart phase only restarts
    # units, so an sshd-spawned `serve --isolated` or a manual `hermes serve` (no unit) is left running its
    # pre-update sys.modules graph — and its cron ticker keeps firing agent jobs that ImportError on every
    # symbol added in the pulled range. The rows also feed the plan-vs-execution reconciliation below, so a
    # survivor is escalated (exit 1) instead of merely printed.
    _stale_serve_rows: "list | None" = None
    with _best_effort('Failed to check for surviving serve runtimes: %s'):
        _stale_serve_rows = _surviving_pre_update_serve_runtimes(_pre_update_plan)
        if _stale_serve_rows:
            _warn_stale_serve_runtimes(_stale_serve_rows)

    print()
    print("Tip: You can now select a provider and model:")
    print("  hermes model              # Select provider and model")

    # Compare every live gateway's stamped code_sha against the fresh checkout
    # instead of assuming the restart phase worked.
    # Phase 1 (#91277): post-update fleet version verification.
    _fleet_snapshot: list = []
    with _best_effort('Fleet version verification failed: %s'):
        from hermes_cli.update_receipt import print_fleet_version_matrix
        # Cross-platform "rows expected" signal: (restarted_services or killed_pids)
        # never fires on Windows (pause/resume populates neither), so a healthy
        # resumed gateway yielded zero rows and exit 0.
        # See #93406.
        # A gateway stopped WITHOUT a successor ("Restart manually") publishes no row by design,
        # so it must not count as an expected one — otherwise an update whose only live gateways
        # were unmapped exits 1 with "no rows" after correctly stopping them.
        _pre_restart, _killed = restart.fleet_probe_signals()
        _fleet_rows_expected = _m()._fleet_probe_expected_runtimes(
            _pre_update_plan, _pre_restart, _windows_gateway_resume, restart.restarted_services, _killed,
        )
        _fleet_snapshot = _collect_fleet_snapshot(restart, _fleet_rows_expected)
        if print_fleet_version_matrix(_fleet_snapshot):
            restart.incomplete = True
        elif not _fleet_snapshot and _fleet_rows_expected:
            # collect_fleet_versions() swallows every failure, so zero rows with
            # expected runtimes is indistinguishable from health — fail (partial, exit 1).
            print(
                # Fleet probe returned zero rows even though at least one gateway runtime was (or may have
                # been) live pre-update — POSIX restart bookkeeping, the pre-restart PID snapshot, the
                # pre-update plan inventory, or the Windows pause/resume token all count as that signal.
                # Every failure path inside collect_fleet_versions() is swallowed via logger.debug(), so an
                # empty list is indistinguishable from a healthy fleet in the current output. Treat it as
                # verification failure so the receipt records "partial" and the exit code is 1 (#93406).
                "\n⚠ Fleet version check returned no rows even though"
                " gateway runtimes were expected — verification incomplete."
            )
            restart.incomplete = True

    # Every runtime the PLAN saw must appear in restart bookkeeping; an
    # unaccounted one is a silent miss and escalates like a STALE/DOWN row.
    with _best_effort('Runtime-outcome reconciliation failed: %s'):
        # An unaccounted runtime is the silent-miss class (a platform branch re-discovered its own targets
        # and skipped one the inventory knew about) — escalate it exactly like a STALE/DOWN fleet row. See
        # #91277.
        if _pre_update_plan is not None and _pre_update_plan.runtimes:
            from hermes_cli.update_inventory import (match_runtime_outcomes, report_unaccounted_runtimes)
            _runtime_outcomes = match_runtime_outcomes(
                _pre_update_plan,
                restarted_services=restart.restarted_services,
                relaunched_profiles=restart.relaunched_profiles,
                externally_supervised_profiles=restart.externally_supervised_profiles,
                killed_pids=restart.killed_pids,
                failed_units=restart.failed_or_stale_units,
                # Serve/dashboard reconcile by incarnation liveness, not unit names.
                # See #100479.
                stale_serve_pids=(
                    {row.get("pid") for row in _stale_serve_rows}
                    if _stale_serve_rows is not None
                    else None
                ),
            )
            if report_unaccounted_runtimes(_runtime_outcomes):
                restart.incomplete = True
            with suppress(Exception):
                import hermes_cli.update_receipt as _ur
                if _ur._current is not None:
                    _ur._current.data["runtime_outcomes"] = _runtime_outcomes

    with _best_effort('Update receipt finalize failed: %s'):
        from hermes_cli.update_receipt import finalize_update_receipt
        _receipt_path = finalize_update_receipt(
            "partial" if restart.incomplete or not update_complete else "success",
            fleet=_fleet_snapshot,
        )
        if _receipt_path is not None:
            logger.info("Update receipt written: %s", _receipt_path)

    if restart.incomplete:
        # Code updated but a gateway may still run stale modules: fail so automation
        # doesn't treat the fleet as healthy; leave the pending marker for catch-up.
        sys.exit(1)
    _clear_fleet_restart_pending_marker()


def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped restart-phase exception must fail the update.

    Fail closed unless provably safe: ``surviving`` None (unprobeable) or non-empty →
    stale. ``[]`` proves safety ONLY if nothing ran beforehand; a pre-restart gateway
    (``pre_restart_pids`` non-empty or None) now gone was stopped unverified.

    * ``surviving is None`` — the survivor probe could not determine state (typically the freshly-pulled
    ``hermes_cli.gateway`` no longer imports, one of the ways the phase aborts). That is proof-of-safety
    ONLY when nothing was running before we touched anything. If a gateway was discovered pre-restart
    (``pre_restart_pids`` non-empty, or ``None`` meaning the pre-state could not be read), it was stopped
    without a verified replacement, so we still fail closed (#78574).
    """
    if surviving is None or surviving:
        return True
    return pre_restart_pids is None or bool(pre_restart_pids)


def _fleet_probe_expected_runtimes(
    pre_update_plan, pre_restart_pids, windows_resume_token, restarted_services, killed_pids,
) -> bool:
    """Whether the post-update fleet probe should have produced rows.

    ``collect_fleet_versions()`` swallows every failure and an empty matrix prints as
    healthy, so zero rows is only proof-of-safety when NOTHING says a gateway existed
    pre-update. Signals: ``restarted_services``/``killed_pids``; ``pre_restart_pids``
    non-empty or None (same contract as ``_restart_phase_failure_is_incomplete``); plan
    inventoried ≥1 ``kind == "gateway"`` runtime. ``windows_resume_token`` is deliberately EXCLUDED: it is
    pause/resume bookkeeping, not an inventory, and its entries don't map to probe rows
    (``unmapped`` Scheduled-Task gateways never publish gateway_state.json; a paused
    profile resumes DETACHED). Counting it made every Windows update that paused a
    gateway exit 1 after a long silent wait; a live pre-update Windows gateway is already
    covered by ``pre_restart_pids`` and the plan. The same condition gates the settle sleep.

    See #93406.
    See #78574.
    See #93406.
    """
    del windows_resume_token  # excluded on purpose — see docstring
    # See #93406.
    if restarted_services or killed_pids:
        return True
    if pre_restart_pids is None or pre_restart_pids:
        return True
    with suppress(Exception):
        # Gateway-kind only: serve/dashboard plan records never publish a gateway_state.json
        # row, so a dashboard-only plan cannot ground a rows-expected verdict (#97332).
        if pre_update_plan is not None and any(
            getattr(runtime, "kind", None) == "gateway" for runtime in pre_update_plan.runtimes
        ):
            return True
    return False


def _wait_for_service_active(scope_cmd_: list, svc_name_: str, timeout: float = 10.0) -> bool:
    """Poll ``systemctl is-active`` (0.5s) up to ``timeout``: the Stopped -> Started
    transition isn't instantaneous, so a one-shot check falsely reports down."""
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        with suppress(FileNotFoundError, subprocess.TimeoutExpired):
            _verify = _systemctl(scope_cmd_ + ["is-active", svc_name_], timeout=5)
            if _verify.stdout.strip() == "active":
                return True
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)


_RESTART_SEC_UNITS = (("ms", 0.001), ("us", 0.000001), ("min", 60.0), ("s", 1.0))


def _service_restart_sec(scope_cmd_: list, svc_name_: str, default: float = 0.0) -> float:
    """Read the unit's ``RestartUSec`` in seconds. ``is-active`` pollers must wait
    >= RestartSec + slack or they give up *during* the cooldown and misreport."""
    try:
        _show = _systemctl(scope_cmd_ + ["show", svc_name_, "--property=RestartUSec", "--value"], timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # Values like "30s", "100ms", "1min 30s", "infinity"; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in _RESTART_SEC_UNITS:
            if part.endswith(_suf):
                with suppress(ValueError):
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                break
    return total if matched else default
