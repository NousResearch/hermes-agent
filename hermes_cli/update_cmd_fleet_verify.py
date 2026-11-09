"""Post-restart fleet verification for ``hermes update`` (``update_cmd_fleet`` sibling).

After the restart phase: compare every live gateway's stamped code against the checkout, reconcile
the pre-update plan against what the restart did, keep the restart owed (a ``gateway_restart``
follow-up, never a failed update — contract C3) or clear the obligation, and finalize the receipt.
Restart-phase helpers stay in ``update_cmd_fleet`` and are read through it so their patches apply.
"""

import json
import logging
import subprocess
import time as _time
from contextlib import suppress

from hermes_cli.update_cmd_common import _best_effort

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

# A supervisor can report a restarted unit active before the gateway finishes its
# bootstrap and publishes ``gateway_state.json``. Keep the readiness poll bounded,
# but allow the default systemd startup budget plus status-publication slack.
_FLEET_PROBE_SETTLE_TIMEOUT_SECONDS = 120.0


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
    Pre-restart PIDs make a gateway stopped WITHOUT verified replacement a DOWN row (owed restart)
    instead of no row at all. An ``unknown`` row whose pid is NOT a pre-restart pid is a successor
    that has not published its code identity yet (a relaunched gateway can sit ~10s between process
    start and its first runtime-status write, #112634) — keep polling; at the deadline it is flagged
    ``identity_pending`` so the matrix does not call it a pre-stamping gateway.
    """
    from hermes_cli.update_receipt import collect_fleet_versions
    pending = getattr(restart, "self_restart_pending_pids", None) or None
    if not rows_expected:
        return collect_fleet_versions(
            pre_restart_pids=restart.pre_restart_gateway_pids, self_restart_pending=pending)
    pre_pids = restart.pre_restart_gateway_pids
    _fleet_deadline = _time.monotonic() + _FLEET_PROBE_SETTLE_TIMEOUT_SECONDS
    while True:
        _time.sleep(2.0)
        snapshot = collect_fleet_versions(pre_restart_pids=pre_pids, self_restart_pending=pending)
        unstamped = [row for row in snapshot if _fleet_row_identity_pending(row, pre_pids)]
        if snapshot and not unstamped and not any(row.get("state") == "down" for row in snapshot):
            return snapshot
        if _time.monotonic() >= _fleet_deadline or _restarted_units_gone(
                getattr(restart, "restarted_scoped_units", ())):
            for row in unstamped:
                row["identity_pending"] = True
            return snapshot


def _fleet_row_identity_pending(row: dict, pre_restart_pids) -> bool:
    """An ``unknown`` row with no sha from a pid that did not exist at update start: a relaunched
    gateway still booting, not a gateway that predates version stamping. A surviving pre-restart pid
    (or no pid snapshot at all) is settled as-is — waiting cannot change what it publishes."""
    if row.get("state") != "unknown" or row.get("code_sha"):
        return False
    if pre_restart_pids is None:
        return False
    return row.get("pid") not in {int(p) for p in pre_restart_pids if isinstance(p, int)}


def _restarted_units_gone(scoped_units) -> bool:
    """True when every restarted systemd unit is LOADED in its scope and neither active nor
    activating: the successor died, nothing will publish a state stamp, so the settle poll should fail
    closed now instead of at the deadline. Anything inconclusive keeps waiting: no units, systemctl
    missing/slow, or ``LoadState=not-found`` — a unit name asked in a scope that does not own it
    answers ``inactive`` exactly like a dead unit (#112466), so only a loaded unit can prove death."""
    from hermes_cli import update_cmd_fleet as fleet
    if not scoped_units:
        return False
    scope_cmds = dict(fleet._SYSTEMD_SCOPES)
    for scoped in scoped_units:
        scope, _, name = scoped.partition("/")
        try:
            stdout = fleet._systemctl(scope_cmds[scope] + ["show", "-p", "LoadState,ActiveState", name], timeout=5).stdout
        except (KeyError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        props = dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)
        if props.get("LoadState") != "loaded":
            return False
        if props.get("ActiveState") in ("active", "activating", "reloading"):
            return False
    return True


def _live_gateway_pids_from_fleet(fleet_rows: list) -> dict:
    """Profile -> gateway PIDs alive after the restart phase, from the fleet snapshot.

    Incarnation evidence for gateway reconciliation (``match_runtime_outcomes``'s
    ``live_gateway_pids``): the plan identifies a gateway by the profile it SERVES while the restart
    bookkeeping names the SERVICE, and a service can serve a profile its name does not encode — the
    profile-scoped name matcher then never credits the planned runtime. A ``down`` row reports the
    PRE-restart PID (nothing replaced it), so it is never a successor; rows without a usable
    profile/PID are skipped.
    """
    live: dict = {}
    for row in fleet_rows:
        pid, profile = row.get("pid"), row.get("profile")
        if profile and isinstance(pid, int) and row.get("state") != "down":
            live.setdefault(str(profile), set()).add(pid)
    return live


def _gateway_unit_runs_one_of(unit: str, pids: set) -> bool:
    """Every scope that has ``hermes-gateway*`` *unit* loaded runs it as one of *pids*.

    The restart bookkeeping keeps bare unit names, so it cannot say which scope failed: a loaded twin
    in the other scope that is not proven keeps the failure. Anything unreadable proves nothing.
    """
    from hermes_cli import update_cmd_fleet as fleet
    if not pids or not (unit == "hermes-gateway" or unit.startswith("hermes-gateway-")):
        return False
    loaded = False
    for _scope, scope_cmd in fleet._SYSTEMD_SCOPES:
        try:
            stdout = fleet._systemctl(scope_cmd + ["show", "-p", "LoadState,MainPID", unit], timeout=5).stdout
            props = dict(line.split("=", 1) for line in (stdout or "").splitlines() if "=" in line)
            if props.get("LoadState") != "loaded":
                continue
            loaded = True
            if int(props.get("MainPID") or 0) not in pids:
                return False
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return False
    return loaded


def _settle_units_proven_current(restart, fleet_rows: list) -> bool:
    """Credit each failed gateway unit whose own MainPID the fleet probe shows serving this checkout.

    A restart call can time out while systemd still finishes the job (a unit whose stop takes
    minutes), so the restart phase's "failed" can be a gateway that came up on the new code moments
    later. Proof is the unit's own process: a ``current`` row whose pid is the unit's MainPID and was
    not running when the restart phase began. Another profile's current row never vouches for it.
    Returns True when a unit was credited; the phase's own incompleteness is then recomputed. The
    caller saves the restart result once verification is done: a later check can still owe a restart.
    """
    pre = restart.pre_restart_gateway_pids
    if pre is None or not restart.failed_or_stale_units:
        return False
    fresh = {row.get("pid") for row in fleet_rows if row.get("state") == "current"} - set(pre)
    proven = [unit for unit in dict.fromkeys(restart.failed_or_stale_units) if _gateway_unit_runs_one_of(unit, fresh)]
    if not proven:
        return False
    restart.failed_or_stale_units[:] = [u for u in restart.failed_or_stale_units if u not in proven]
    restart.unrestarted_units[:] = [u for u in restart.unrestarted_units if u not in proven]
    restart.restarted_services.extend(u for u in proven if u not in restart.restarted_services)
    for unit in proven:
        print(f"  ✓ {unit} finished restarting onto the new code after the restart phase stopped waiting")
    restart.incomplete = bool(restart.failed_or_stale_units or restart.phase_errors)
    if not restart.incomplete:
        # The restart phase withheld this stamp; every other profile's CLI reads it.
        from hermes_cli import update_cmd_fleet as fleet
        from hermes_cli.update_host_obligation import mark_host_restart_completed
        mark_host_restart_completed(fleet._restart_identity_sha())
    return True


def _record_owed_gateway_inventory(plan, stopped_unmapped_pids=()) -> None:
    """Name the pre-update gateways on the obligation once their restart is owed.

    The pull arms the obligation before the restart phase knows what it owes. An inventory-less
    record on a host whose gateway then died at boot is settled by the gateway-less discharge
    (nothing live, nothing named), which would silence the owed-restart warning that replaces a
    failed update under contract C3. A named inventory keeps it until those gateways run HEAD.
    A gateway stopped with no known profile is recorded as an unmapped row for the same reason: it
    has no ``gateway_state.json`` the gateway-less probe could see, so absence would settle it.
    A readable inventory (a same-SHA catch-up keeps the standing record) only GROWS: rows this run
    newly owes are appended, so a mapped successor cannot discharge a newly stopped unmapped one.
    """
    from hermes_cli import update_cmd_fleet as fleet
    armed = fleet._fleet_restart_obligation_armed()
    fields = fleet._obligation_fields() if armed else {}
    if fields is None:
        return  # unreadable terms stay fail-closed
    recorded: list = []
    if fields.get("inventory") not in (None, "", "null"):
        try:
            inventory = json.loads(fields["inventory"])
            fleet._marker_owed_gateways(inventory)  # validates; malformed/unsupported stays untouched
        except ValueError:
            return
        recorded = inventory["runtimes"]
    from dataclasses import asdict
    rows = [
        asdict(runtime) for runtime in getattr(plan, "runtimes", ()) or ()
        if getattr(runtime, "kind", None) == "gateway"
        and isinstance(getattr(runtime, "profile", None), str)
        and runtime.profile.strip() and runtime.profile != "unknown"
    ] + [{"kind": "gateway", "profile": None, "pid": pid, "stopped_unmapped": True}
         for pid in sorted(stopped_unmapped_pids)]
    rows = [row for row in rows if row not in recorded]
    if rows:
        # Not armed any more (a pre-restart probe settled an inventory-less record): re-arm, the
        # restart is owed. The SHA is the code the fleet must be proven current on.
        sha = fields.get("expected_sha", "") if armed else (fleet._current_checkout_sha() or "")
        fleet._write_fleet_restart_pending_marker(expected_sha=sha, runtimes=recorded + rows)


def _named_gateways_still_owed() -> bool:
    """True when the armed obligation NAMES gateways (recorded by an earlier owed restart) and the
    live fleet still does not prove them current. An inventory-less record keeps its old rules."""
    from hermes_cli import update_cmd_fleet as fleet
    if not fleet._fleet_restart_obligation_armed():
        return False
    fields = fleet._obligation_fields()
    if fields is None or fields.get("inventory") in (None, "", "null"):
        return False
    return not fleet._marker_only_restart_obsolete()


def _is_serve_unit(name: object) -> bool:
    """A serve/dashboard unit by the fleet loop's own name gate (``hermes-serve*``/``hermes-dashboard*``,
    bare or ``.service``). Anything else it records as failed (gateway units, launchd labels,
    listing-failure markers) is gateway evidence."""
    from hermes_cli.update_cmd_fleet import _is_serve_or_dashboard_unit

    return _is_serve_or_dashboard_unit(f"{str(name).removesuffix('.service')}.service")


def _restart_phase_owes_gateway(restart) -> bool:
    """Did the restart phase itself leave a gateway owed? Its incompleteness counts against the
    gateways unless every failure it recorded is a serve/dashboard unit (#141021)."""
    if not restart.incomplete:
        return False
    failed = list(restart.failed_or_stale_units)
    return bool(restart.phase_errors) or not failed or not all(map(_is_serve_unit, failed))


def _split_unaccounted(outcomes) -> tuple[bool, set]:
    """``(gateway missed, serve/dashboard PIDs missed)`` among the unaccounted runtime outcomes."""
    from hermes_cli.update_inventory import _SERVE_KINDS

    missed = [o for o in outcomes if o.get("outcome") == "unaccounted"]
    serve_pids = {o["pid"] for o in missed if o.get("kind") in _SERVE_KINDS and isinstance(o.get("pid"), int)}
    return any(o.get("kind") not in _SERVE_KINDS for o in missed), serve_pids


# The Desktop hand-offs (scripts/desktop-update/posix.sh, windows.ps1) drop their "Run hermes gateway
# restart" hint only when every gateway_restart reason starts with this prefix, so every serve-only
# part below starts with it and the gateway part always comes first (#141071).
SERVE_ONLY_REASON_PREFIX = "serve/dashboard "
_GATEWAY_OWED_REASON = "gateways may still run pre-update code or were stopped without a successor"
_GATEWAY_RECOVERY = "; recover with `hermes gateway restart`"


def _owed_restart_reason(restart, stopped, *, gateway_owed: bool, stale_serves: set, down_serves: set) -> str:
    """The ``gateway_restart`` follow-up's reason, naming what is actually owed (#141021).

    The step id stays ``gateway_restart`` (Desktop hand-offs parse it); only the text changes. A run
    whose gateways all verified current but whose dashboard unit could not be restarted must not tell
    the user to run `hermes gateway restart`, which never touches that unit."""
    parts = []
    if gateway_owed:
        parts.append(
            _GATEWAY_OWED_REASON
            + (f" (stopped PIDs {', '.join(map(str, stopped))})" if stopped else "")
            + _GATEWAY_RECOVERY)
    units = sorted({str(u).removesuffix(".service") for u in restart.failed_or_stale_units if _is_serve_unit(u)})
    if units:
        parts.append(
            f"{SERVE_ONLY_REASON_PREFIX}unit(s) {', '.join(units)} were not restarted onto the new code;"
            " see the systemctl advice above")
    if down_serves:
        parts.append(
            f"{SERVE_ONLY_REASON_PREFIX}process(es) {', '.join(map(str, sorted(down_serves)))} were stopped during the"
            " update and not brought back; relaunch `hermes serve` / `hermes dashboard`")
    if stale_serves - down_serves:
        parts.append(
            f"{SERVE_ONLY_REASON_PREFIX}process(es) {', '.join(map(str, sorted(stale_serves - down_serves)))} still run"
            " pre-update code; restart them as advised above")
    # Nothing classified (an unknown incompleteness): keep the historical, conservative wording.
    return "; ".join(parts) or (_GATEWAY_OWED_REASON + _GATEWAY_RECOVERY)


def _verify_fleet_after_update(restart, *, _pre_update_plan, _windows_gateway_resume, update_complete):
    """Post-restart verification: legacy-unit warning, dashboard cleanup, stale serve
    probe, fleet version matrix, plan-vs-execution reconciliation, receipt finalize.

    Never fails the committed update (contract C3): when any gateway may still be stale it
    records a ``gateway_restart`` follow-up and leaves ``fleet_restart_pending`` armed for the
    next catch-up and the startup warning; otherwise clears the marker.
    """
    from hermes_cli import update_cmd_fleet as fleet
    from hermes_cli.update_cmd import (
        _m, _surviving_pre_update_serve_runtimes, _warn_stale_serve_runtimes,
    )
    from hermes_cli.update_cmd_maint import _refresh_dashboard_after_update
    with _best_effort('Legacy unit check during update failed: %s'):
        _print_legacy_units_warning()

    # Restart a managed dashboard via systemd or stop stale manual ones (raw-killing
    # a systemd-owned PID reads as clean stop and leaves the Cloudflare origin dead).
    # Which family is owed decides what the follow-up tells the user to run (#141021). Read before the
    # dashboard pass can flip ``incomplete``.
    _gateway_owed = _restart_phase_owes_gateway(restart)
    # Units the fleet loop already attempted aren't redone — restarted ones, and failed ones too: the
    # loop already reported each failure (with its systemctl advice), and a second attempt only
    # duplicates it or, for a system unit without root, adds a polkit/tty error (#141021).
    # A dashboard it stopped and could not bring back is a promised restart that did not happen.
    _dashboards_down = _refresh_dashboard_after_update(
        already_restarted_units=set(restart.restarted_services) | set(restart.failed_or_stale_units))
    if _dashboards_down:
        restart.incomplete = True
    _stale_serves_owed: set = set()

    # Success-path twin of the abort-recovery probe: the restart phase only touches
    # units, so a unit-less `hermes serve` keeps stale sys.modules. Runs AFTER
    # dashboard cleanup so a respawned manual dashboard isn't a survivor. Rows feed
    # reconciliation (survivor → owed restart); ``None`` = probe failed, stays fail-closed.
    # Check if any pre-update serve/dashboard runtimes survived on pre-update code generations (#100479).
    # This is the SUCCESS-path twin of the abort-recovery probe above: the restart phase only restarts
    # units, so an sshd-spawned `serve --isolated` or a manual `hermes serve` (no unit) is left running its
    # pre-update sys.modules graph — and its cron ticker keeps firing agent jobs that ImportError on every
    # symbol added in the pulled range. The rows also feed the plan-vs-execution reconciliation below, so a
    # survivor is escalated (an owed gateway_restart follow-up) instead of merely printed.
    _stale_serve_rows: list | None = None
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
    _units_settled = False
    with _best_effort('Fleet version verification failed: %s'):
        from hermes_cli.update_receipt import print_fleet_version_matrix
        # Cross-platform "rows expected" signal: (restarted_services or killed_pids)
        # never fires on Windows (pause/resume populates neither), so a healthy
        # resumed gateway yielded zero rows and exit 0.
        # See #93406.
        # A gateway stopped WITHOUT a successor ("Restart manually") publishes no row by design,
        # so it must not count as an expected one — otherwise an update whose only live gateways
        # were unmapped reports "no rows" after correctly stopping them.
        _pre_restart, _killed = restart.fleet_probe_signals()
        _fleet_rows_expected = _m()._fleet_probe_expected_runtimes(
            _pre_update_plan, _pre_restart, _windows_gateway_resume, restart.restarted_services, _killed,
        )
        _fleet_snapshot = _collect_fleet_snapshot(restart, _fleet_rows_expected)
        _units_settled = _settle_units_proven_current(restart, _fleet_snapshot)
        if _units_settled:
            _gateway_owed = _restart_phase_owes_gateway(restart)
            restart.incomplete = restart.incomplete or bool(_dashboards_down)
        if print_fleet_version_matrix(_fleet_snapshot):
            restart.incomplete = _gateway_owed = True
            # A proven-stale survivor must not keep running (its ticker yields every tick and
            # nothing else restarts it, #117275): hand it to the drain-first restart path.
            from hermes_cli.update_cmd_stale_survivors import signal_stale_fleet_survivors
            signal_stale_fleet_survivors(_fleet_snapshot, restart, fleet._gateway_drain_budget())
        elif not _fleet_snapshot and _fleet_rows_expected:
            # collect_fleet_versions() swallows every failure, so zero rows with
            # expected runtimes is indistinguishable from health — keep the restart owed.
            print(
                # Fleet probe returned zero rows even though at least one gateway runtime was (or may have
                # been) live pre-update — POSIX restart bookkeeping, the pre-restart PID snapshot, the
                # pre-update plan inventory, or the Windows pause/resume token all count as that signal.
                # Every failure path inside collect_fleet_versions() is swallowed via logger.debug(), so an
                # empty list is indistinguishable from a healthy fleet in the current output. Treat it as
                # verification failure so the receipt carries a gateway_restart follow-up (#93406).
                "\n⚠ Fleet version check returned no rows even though"
                " gateway runtimes were expected — verification incomplete."
            )
            restart.incomplete = _gateway_owed = True
    # Printed after the probe, not by the restart phase: only units it did not prove current.
    fleet._warn_incomplete_gateway_fleet_restart(getattr(restart, "unrestarted_units", None) or [])

    # Every runtime the PLAN saw must appear in restart bookkeeping; an
    # unaccounted one is a silent miss and escalates like a STALE/DOWN row.
    with _best_effort('Runtime-outcome reconciliation failed: %s'):
        # An unaccounted runtime is the silent-miss class (a platform branch re-discovered its own targets
        # and skipped one the inventory knew about) — escalate it exactly like a STALE/DOWN fleet row. See
        # #91277.
        if _pre_update_plan is not None and _pre_update_plan.runtimes:
            from hermes_cli.update_inventory import (match_runtime_outcomes, report_unaccounted_runtimes)
            from hermes_cli.update_receipt import row_is_external
            # Gateway incarnation evidence, from the post-restart fleet snapshot collected above: a
            # service can serve a profile its own name does not encode (root-home launchd label +
            # sticky active profile, hashed custom HERMES_HOME), so the bookkeeping's service names
            # alone cannot credit the planned runtime. A profile the probe produced no row for simply
            # has no evidence — that runtime stays on the name-matching path (and logs it).
            _live_gateway_pids = _live_gateway_pids_from_fleet(_fleet_snapshot)
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
                failed_respawn_pids=_dashboards_down,
                # A symlinked profile served by another install's checkout (#120240).
                external_gateway_pids={row.get("pid") for row in _fleet_snapshot if row_is_external(row)},
                live_gateway_pids=_live_gateway_pids,
            )
            from dataclasses import asdict
            from hermes_cli.update_serve_obligations import defer_manual_serve

            for runtime, outcome in zip(_pre_update_plan.runtimes, _runtime_outcomes):
                if outcome["outcome"] == "unaccounted" and defer_manual_serve(asdict(runtime), require_alive=True):
                    outcome["outcome"] = "deferred"
            if report_unaccounted_runtimes(_runtime_outcomes):
                restart.incomplete = True
                _missed_gateway, _missed_serves = _split_unaccounted(_runtime_outcomes)
                _gateway_owed |= _missed_gateway
                _stale_serves_owed |= _missed_serves
            with suppress(Exception):
                import hermes_cli.update_receipt as _ur
                _active = _ur._current.get()
                if _active is not None:
                    _active.data["runtime_outcomes"] = _runtime_outcomes

    if not restart.incomplete:
        with _best_effort('Owed-gateway check failed: %s'):
            # Only on an otherwise-complete run: the check can settle (clear) the host obligation,
            # which an incomplete run must not do mid-verification. An incomplete run keeps the
            # obligation armed anyway, and the startup warning names any older owed gateway.
            if _named_gateways_still_owed():
                # A gateway a previous run could not restart is still not serving HEAD (nothing
                # live to restart this time): keep owing it rather than clearing the obligation.
                print("  ⚠ A gateway owed a restart by a previous update is still not serving this checkout.")
                restart.incomplete = _gateway_owed = True
    if getattr(restart, "stopped_unmapped_pids", None):
        # A gateway stopped with no successor is owed a restart, not "accounted for": keep the
        # fleet obligation armed so the startup warning names it until someone restarts it.
        restart.incomplete = _gateway_owed = True
    if restart.incomplete:
        # Code is committed (contract C3): a gateway that may still run stale modules is a
        # follow-up, never a failed update. The fleet obligation stays armed, every CLI start
        # warns about it, and the next `hermes update` retries the restart.
        from hermes_cli.update_receipt import record_followup
        stopped = sorted(getattr(restart, "stopped_unmapped_pids", None) or ())
        record_followup("gateway_restart", _owed_restart_reason(
            restart, stopped, gateway_owed=_gateway_owed, stale_serves=_stale_serves_owed,
            down_serves=set(_dashboards_down or ())))
        with _best_effort('Fleet restart inventory not recorded: %s'):
            _record_owed_gateway_inventory(_pre_update_plan, stopped)
    if _units_settled:
        # Credited units changed the restart result; save it with the final verdict so it agrees with
        # the follow-up above.
        restart.record_receipt()
    with _best_effort('Update receipt finalize failed: %s'):
        from hermes_cli.update_receipt import finalize_update_receipt
        # A False ``update_complete`` (unsafe SQLite runtime, owed tail work) never fails the
        # run: maintenance already reported it as a follow-up. It only vetoes migration below.
        _receipt_path = finalize_update_receipt("success", fleet=_fleet_snapshot)
        if _receipt_path is not None:
            logger.info("Update receipt written: %s", _receipt_path)

    if restart.incomplete:
        return
    fleet._clear_fleet_restart_pending_marker()
    if not update_complete:
        # Fleet caught up, but the selected runtime's SQLite is unsafe or the tail is still owed
        # (both reported by the completion): leave the topology alone until they are repaired.
        return
    # Fleet is healthy on the new code: fold per-profile gateways into one multiplexer when nothing
    # blocks it (deterministic; never prompts), else print the blockers and the one-liner to run later.
    try:
        from hermes_cli.gateway_migrate import maybe_auto_migrate_after_update
        maybe_auto_migrate_after_update()
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- a SystemExit here must not fail a committed update
        logger.warning('Multiplex auto-migration after update failed: %s', exc)
        print(f"  ⚠ Gateway multiplex migration did not finish: {exc} (run `hermes gateway migrate` later)")
