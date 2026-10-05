"""Post-checkout completion shared by current and historical update callers."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def finish_update(*, root, assume_yes, gateway_mode, pre_update_snapshot_id,
                  had_desktop_app_before_update, pre_update_version,
                  plan, windows_resume, followups=None) -> None:
    """Finish the selected checkout; never fetch, switch branches or restore a stash.

    Same contract as the current completion (C3): the code is committed, so a failed build,
    maintenance step or fleet restart/verification is owed work, never a failed update.
    ``followups`` carries the steps that already failed (the product build); while any is owed
    the install stamp is withheld, the source-update tail stays pending for the next launch, and
    the gateway topology is left alone. An unsafe SQLite runtime is reported by maintenance and
    vetoes migration only.
    """
    from hermes_cli.update_cmd import (
        _run_post_update_maintenance,
        _restart_gateway_fleet_after_update, _verify_fleet_after_update,
        _write_gateway_update_exit_code, _resume_windows_gateways_and_merge_outcome,
    )
    from hermes_cli import update_receipt
    from hermes_cli.update_receipt import TAIL_FOLLOWUPS, record_build_stage, record_followup

    owed = followups if followups is not None else []
    runtime_safe = False
    try:
        runtime_safe = _run_post_update_maintenance(
            assume_yes=assume_yes, gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            had_desktop_app_before_update=had_desktop_app_before_update,
            pre_update_version=pre_update_version, followups=owed,
        )
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- the code is committed; retry later
        reason = str(exc) or type(exc).__name__
        record_followup("maintenance", reason)
        owed.append(("maintenance", reason))
    tail_owed = any(step in TAIL_FOLLOWUPS for step, _ in owed)
    record_build_stage(owed)
    if not tail_owed:
        from hermes_cli.source_stamp import write_source_stamp
        from hermes_cli.venv_sync import clear_completion

        try:
            write_source_stamp(Path(root))
        except (OSError, ValueError) as exc:
            print(f"⚠ Source update completed, but the install stamp could not be written: {exc}",
                  file=sys.stderr)
        clear_completion(Path(root))
    # Restart can kill this process's gateway cgroup; record its result first. The code is
    # committed, so the watcher sees success; owed work is on the receipt.
    if gateway_mode:
        _write_gateway_update_exit_code(True)
    run = update_receipt._current.get()
    update_id = run.data.get("update_id") if run is not None else None
    try:
        restarted = _restart_gateway_fleet_after_update(plan, gateway_mode)
        _resume_windows_gateways_and_merge_outcome(restarted, windows_resume, gateway_mode)
        _verify_fleet_after_update(restarted, _pre_update_plan=plan, _windows_gateway_resume=windows_resume,
                                  update_complete=bool(runtime_safe) and not tail_owed)
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- the code is committed (C3): the fleet obligation stays armed
        reason = str(exc) or type(exc).__name__
        record_followup("gateway_restart", reason)
        if update_receipt._current.get() is None and update_id:  # verification already finalized the run
            update_receipt.amend_terminal_followup(update_id, "gateway_restart", reason)


def _restore_plan(data):
    if not data:
        return None
    from dataclasses import fields
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    values = {field.name: data[field.name] for field in fields(UpdatePlan) if field.name in data}
    names = {field.name for field in fields(RuntimeRecord)}
    values["runtimes"] = [RuntimeRecord(**{key: value for key, value in row.items() if key in names})
                          for row in data.get("runtimes", [])]
    return UpdatePlan(**values)


def main(context: Path, result: Path) -> int:
    request = json.loads(context.read_text(encoding="utf-8-sig"))
    root = Path(request["root"])
    sys.path.insert(0, str(root))
    from hermes_cli import update_receipt

    token = request.get("windows_resume")
    resume = None
    restarting = request.get("restart_update", False)
    cli_started = False
    begun = False
    code = 1
    try:
        if not restarting:
            update_receipt.begin_update_receipt(previous=request.get("receipt"), correlation_id=request["update_id"])
            begun = update_receipt._current.get() is not None
            if not begun:
                raise RuntimeError("cannot create takeover receipt")
            from pm import receipt
            receipt.accept_worker_receipt(request.get("pm_receipt"), request["update_id"])
            update_receipt.record_step("historical_takeover", True, "continuing in the selected interpreter")
        # Preparation has already committed the dependency generation and
        # selected store Python. Bootstrap must run before any app imports;
        # its ordinary currency check is now a no-op, not another update.
        sys.argv = list(request["argv"]) if restarting else [str(root / "hermes"), "update"]
        import hermes_bootstrap  # noqa: F401
        # Import failures are update failures too: keep the original receipt
        # open before importing the application graph from the new checkout.
        from hermes_cli import main as cli
        from hermes_cli.source_build import build_update_products
        from hermes_cli.update_lock import UpdateLock, describe_holder
        from hermes_cli.update_cmd_windows import _resume_windows_gateways_after_update
        from hermes_cli.venv_sync import arm_completion

        cli.PROJECT_ROOT = root
        if restarting:
            # A known pre-pull capture has no completion worklist yet. The
            # current CLI owns its plan, receipt and lock; don't start a nested
            # receipt or execute the finish-only path against the old checkout.
            cli_started = True
            cli.main()
        else:
            plan = _restore_plan(request.get("plan"))
            with UpdateLock() as lock:
                if not lock.acquired and lock.holder is not None:
                    print(describe_holder(lock.holder), file=sys.stderr)
                    code = 2
                    return code
                resume = _resume_windows_gateways_after_update
                desktop = request.get("desktop")
                if desktop is None:
                    # Historical hooks can precede Desktop detection. Resolve
                    # only that unknown state, in the freshly bootstrapped app.
                    from hermes_cli.main_desktop import _desktop_dist_exists, _desktop_packaged_executable

                    desktop_dir = root / "apps" / "desktop"
                    desktop = (_desktop_packaged_executable(desktop_dir) is not None
                               or _desktop_dist_exists(desktop_dir))
                # The tail is owed from here until it finishes: a failure or a kill leaves the
                # pending marker, so the next launch finishes it (same as the current completion).
                try:
                    arm_completion(root)
                except OSError as exc:  # stale dependencies still trigger the next launch's sync
                    print(f"  ⚠ Could not record the owed source-update tail: {exc}")
                owed: list[tuple[str, str]] = []
                try:
                    build_update_products(root, desktop=desktop)
                except (Exception, SystemExit) as exc:  # health: allow BLE001 -- committed code: owed, not failed
                    reason = str(exc) or type(exc).__name__
                    update_receipt.record_followup("build", reason)
                    owed.append(("build", reason))
                finish_update(
                    root=root, assume_yes=request["assume_yes"], gateway_mode=request["gateway_mode"],
                    pre_update_snapshot_id=request.get("pre_update_snapshot_id"),
                    had_desktop_app_before_update=desktop,
                    pre_update_version=request.get("pre_update_version"),
                    plan=plan, windows_resume=token, followups=owed,
                )
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        update_receipt.record_step("historical_completion", False, str(exc))
        print(f"Update completion failed: {exc}", file=sys.stderr, flush=True)
    finally:
        terminal = update_receipt.finalized_receipt(request["update_id"]) or {}
        if code and request.get("gateway_mode"):
            # Even an application import failure must wake the gateway watcher. A run that already
            # closed as a success (an interrupt after verification) stays 0 (review regression 3).
            from hermes_constants import get_hermes_home
            from hermes_cli.runtime_state import _atomic_bytes

            _atomic_bytes(get_hermes_home() / ".update_exit_code",
                          b"0" if terminal.get("outcome") == "success" else b"1")
        if restarting and not cli_started:
            # Startup failed before the replacement command could own a
            # receipt. Preserve the original handoff, just like preparation.
            update_receipt.begin_update_receipt(previous=request.get("receipt"), correlation_id=request["update_id"])
            begun = update_receipt._current.get() is not None
        attempted = False
        if resume is not None:
            try:
                resume(token)
            except Exception as exc:  # noqa: BLE001 — a restart failure is owed, never the exit status (C3)
                attempted = True
                reason = f"Windows gateway recovery failed: {exc}"
                update_receipt.record_followup("windows_resume", reason)
                if update_receipt._current.get() is None:  # the run already finalized its receipt
                    update_receipt.amend_terminal_followup(request["update_id"], "windows_resume", reason)
        handled = cli_started or (begun and update_receipt._current.get() is None)
        written = update_receipt.finalize_pending_update_receipt(code, "historical takeover completion")
        result.write_text(json.dumps({
            # An attempt that failed is reported here: the historical parent must not replay it.
            "resume_handled": attempted or not token or not token.get("resume_needed"),
            "receipt_handled": handled or written is not None,
        }), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
