"""Fleet-version settlement and terminal receipt logic for ``hermes update``.

The public/patchable names remain re-exported by ``update_cmd_fleet``. Helpers
late-bind that facade when production invokes them, preserving its test seams.
"""

import logging
import subprocess
import sys
import time as _time
from contextlib import suppress

from hermes_cli.update_cmd_common import _best_effort

logger = logging.getLogger("hermes_cli.update_cmd")


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
    instead of no row at all. An ``unknown`` row whose pid is NOT a pre-restart pid is a successor
    that has not published its code identity yet (a relaunched gateway can sit ~10s between process
    start and its first runtime-status write, #112634) — keep polling; at the deadline it is flagged
    ``identity_pending`` so the matrix does not call it a pre-stamping gateway.
    """
    from hermes_cli import update_cmd_fleet as _fleet
    from hermes_cli.update_receipt import collect_fleet_versions
    pending = getattr(restart, "self_restart_pending_pids", None) or None
    if not rows_expected:
        return collect_fleet_versions(
            pre_restart_pids=restart.pre_restart_gateway_pids, self_restart_pending=pending)
    pre_pids = restart.pre_restart_gateway_pids
    _fleet_deadline = _time.monotonic() + _fleet._FLEET_PROBE_SETTLE_TIMEOUT_SECONDS
    while True:
        _time.sleep(2.0)
        snapshot = collect_fleet_versions(pre_restart_pids=pre_pids, self_restart_pending=pending)
        unstamped = [row for row in snapshot if _fleet._fleet_row_identity_pending(row, pre_pids)]
        if snapshot and not unstamped and not any(row.get("state") == "down" for row in snapshot):
            return snapshot
        if _time.monotonic() >= _fleet_deadline or _fleet._restarted_units_gone(
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
    from hermes_cli import update_cmd_fleet as _fleet
    if not scoped_units:
        return False
    scope_cmds = dict(_fleet._SYSTEMD_SCOPES)
    for scoped in scoped_units:
        scope, _, name = scoped.partition("/")
        try:
            stdout = _fleet._systemctl(scope_cmds[scope] + ["show", "-p", "LoadState,ActiveState", name], timeout=5).stdout
        except (KeyError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        props = dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)
        if props.get("LoadState") != "loaded":
            return False
        if props.get("ActiveState") in ("active", "activating", "reloading"):
            return False
    return True


def _verify_fleet_after_update(restart, *, _pre_update_plan, _windows_gateway_resume, node_failures, update_complete, pinned=False):
    """Post-restart verification: legacy-unit warning, dashboard cleanup, stale serve
    probe, fleet version matrix, plan-vs-execution reconciliation, receipt finalize.

    Exits 1 (leaving ``fleet_restart_pending`` for the next catch-up) when any gateway
    may still be stale; otherwise clears the marker.
    """
    from hermes_cli import update_cmd_fleet as _fleet
    from hermes_cli.update_cmd import (
        _finish_dashboard_update_cleanup, _m, _surviving_pre_update_serve_runtimes, _warn_stale_serve_runtimes,
    )
    with _best_effort('Legacy unit check during update failed: %s'):
        _fleet._print_legacy_units_warning()

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
        _fleet_snapshot = _fleet._collect_fleet_snapshot(restart, _fleet_rows_expected)
        if print_fleet_version_matrix(_fleet_snapshot):
            restart.incomplete = True
            # A proven-stale survivor must not keep running (its ticker yields every tick and
            # nothing else restarts it, #117275): hand it to the drain-first restart path.
            from hermes_cli.update_cmd_stale_survivors import signal_stale_fleet_survivors
            signal_stale_fleet_survivors(_fleet_snapshot, restart, _fleet._gateway_drain_budget())
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
            from dataclasses import asdict
            from hermes_cli.update_serve_obligations import defer_manual_serve

            for runtime, outcome in zip(_pre_update_plan.runtimes, _runtime_outcomes):
                if outcome["outcome"] == "unaccounted" and defer_manual_serve(asdict(runtime), require_alive=True):
                    outcome["outcome"] = "deferred"
            if report_unaccounted_runtimes(_runtime_outcomes):
                restart.incomplete = True
            with suppress(Exception):
                import hermes_cli.update_receipt as _ur
                if _ur._current is not None:
                    _ur._current.data["runtime_outcomes"] = _runtime_outcomes

    partial = restart.incomplete or not update_complete or (pinned and bool(node_failures))
    with _best_effort('Update receipt finalize failed: %s'):
        from hermes_cli.update_receipt import finalize_update_receipt
        _receipt_path = finalize_update_receipt(
            "partial" if partial else "success",
            fleet=_fleet_snapshot,
        )
        if _receipt_path is not None:
            logger.info("Update receipt written: %s", _receipt_path)

    if restart.incomplete:
        # Code updated but a gateway may still run stale modules: fail so automation
        # doesn't treat the fleet as healthy; leave the pending marker for catch-up.
        sys.exit(1)
    _fleet._clear_fleet_restart_pending_marker()
    if pinned and partial:
        # The fleet is current, but the pinned update still has unfinished build or
        # maintenance work. Its terminal receipt and process exit must agree.
        sys.exit(1)
    # Fleet is healthy on the new code: fold per-profile gateways into one multiplexer when nothing
    # blocks it (deterministic; never prompts), else print the blockers and the one-liner to run later.
    with _best_effort('Multiplex auto-migration after update failed: %s'):
        from hermes_cli.gateway_migrate import maybe_auto_migrate_after_update
        maybe_auto_migrate_after_update()
