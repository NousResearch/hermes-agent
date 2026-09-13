"""Post-checkout completion shared by current and historical update callers."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def finish_update(*, assume_yes, gateway_mode, pre_update_snapshot_id,
                  had_desktop_app_before_update, pre_update_version,
                  plan, windows_resume) -> None:
    """Finish the selected checkout; never fetch, switch branches or restore a stash."""
    from hermes_cli.update_cmd import (
        _run_post_update_maintenance,
        _restart_gateway_fleet_after_update, _verify_fleet_after_update,
        _write_gateway_update_exit_code, _resume_windows_gateways_and_merge_outcome,
    )

    complete = _run_post_update_maintenance(
        assume_yes=assume_yes, gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
        had_desktop_app_before_update=had_desktop_app_before_update,
        pre_update_version=pre_update_version,
    )
    # Restart can kill this process's gateway cgroup; record its result first.
    if gateway_mode:
        _write_gateway_update_exit_code(complete)
    restarted = _restart_gateway_fleet_after_update(plan, gateway_mode)
    _resume_windows_gateways_and_merge_outcome(restarted, windows_resume, gateway_mode)
    _verify_fleet_after_update(restarted, _pre_update_plan=plan,
                              _windows_gateway_resume=windows_resume, update_complete=complete)


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
    request = json.loads(context.read_text(encoding="utf-8"))
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
                build_update_products(root, desktop=request["desktop"])
                finish_update(
                    assume_yes=request["assume_yes"], gateway_mode=request["gateway_mode"],
                    pre_update_snapshot_id=request.get("pre_update_snapshot_id"),
                    had_desktop_app_before_update=request["desktop"],
                    pre_update_version=request.get("pre_update_version"),
                    plan=plan, windows_resume=token,
                )
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        update_receipt.record_step("historical_completion", False, str(exc))
        print(f"Update completion failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if code and request.get("gateway_mode"):
            # Even an application import failure must wake the gateway watcher.
            from hermes_constants import get_hermes_home
            from hermes_cli.runtime_state import _atomic_bytes

            _atomic_bytes(get_hermes_home() / ".update_exit_code", b"1")
        if restarting and not cli_started:
            # Startup failed before the replacement command could own a
            # receipt. Preserve the original handoff, just like preparation.
            update_receipt.begin_update_receipt(previous=request.get("receipt"), correlation_id=request["update_id"])
            begun = update_receipt._current.get() is not None
        if resume is not None:
            try:
                resume(token)
            except Exception as exc:
                code = 1
                print(f"Gateway recovery failed: {exc}", file=sys.stderr, flush=True)
        handled = cli_started or (begun and update_receipt._current.get() is None)
        written = update_receipt.finalize_pending_update_receipt(code, "historical takeover completion")
        result.write_text(json.dumps({
            "resume_handled": not token or not token.get("resume_needed"),
            "receipt_handled": handled or written is not None,
        }), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
