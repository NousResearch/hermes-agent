"""Absolute-path bootstrap for historical updater takeover; no app imports yet."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def prepare(request: dict) -> tuple[Path, dict[str, str]]:
    """Provision the new graph before entering its application interpreter."""
    root = Path(request["root"])
    sys.path.insert(0, str(root))
    # The old shim's UI has been frozen since the pull; from here the new
    # tree can publish stages (and pop the panel the old shim couldn't).
    from hermes_cli.update_stage import ensure_panel, publish_stage

    ensure_panel(root)
    publish_stage("Updating Python dependencies (PM)")
    from pm import receipt
    from pm.client import ensure_tools_for_sync, sync_venv, venv_is_current
    from pm.environments import activation_environment, install_state_dir, runtime_facts_path
    from hermes_cli._launchers import resolve_store_python
    from hermes_cli.venv_sync import (
        arm_completion, collect_superseded_generations, publish_launchers, refuse_foreign_owned_venv)

    # The historical updater already moved the tree: owe the tail and the fleet restart before the
    # first slow step, exactly like a current updater's commit point, so a kill from here on leaves
    # both for the next launch / `hermes update` instead of nothing.
    refuse_foreign_owned_venv(root)
    arm_completion(root)
    _arm_fleet_obligation(root)
    correlation = request["update_id"]
    with receipt.worker_context(correlation):
        # A pre-PM installation has no required-tool facts. A current Python
        # generation alone does not prove its Node/Git/tool closure is ready.
        ensure_tools_for_sync()
        from pm.extras import legacy_selection

        extras = legacy_selection(root) if not runtime_facts_path(root).is_file() else None
        repair_marker = install_state_dir(root) / ".repair-incomplete"
        # Repair preserves the old stamp. Changed source inputs instead need
        # an ordinary sync, which builds and validates a fresh generation too.
        repair = repair_marker.is_file() and venv_is_current(project_root=root)
        # An update never fails because of a plugin: misfits are disabled and reported.
        sync_venv(None if repair else extras, explicit=True, project_root=root, repair=repair,
                  evict_incompatible_plugins=not repair)
        collect_superseded_generations(root)
        request["pm_receipt"] = receipt.last_for_update(correlation)
    publish_launchers(root)
    repair_marker.unlink(missing_ok=True)
    for name in (".update-incomplete", ".lazy-refresh-incomplete"):
        (root / name).unlink(missing_ok=True)
    python = resolve_store_python(root)
    if python is None:
        raise RuntimeError("updated installation has no managed interpreter")
    return python, activation_environment(root)


def _arm_fleet_obligation(root: Path) -> None:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8",
                          stdin=subprocess.DEVNULL, timeout=60)
    sha = head.stdout.strip() if head.returncode == 0 else ""
    if not sha:  # an SHA-less record names no code the fleet could be proven current on
        return
    # This runs in the HISTORICAL interpreter, before PM syncs the new dependencies: only stdlib-only
    # modules here (``update_cmd_fleet``'s writer imports ``update_cmd`` -> config -> ruamel, which a
    # release older than ruamel does not have).
    from hermes_cli.update_host_obligation import write_host_obligation

    if not write_host_obligation(expected_sha=sha):
        # Same fallback as ``update_cmd_fleet._write_fleet_restart_pending_marker``: the per-home
        # breadcrumb every reader still honours.
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "fleet_restart_pending").write_text(
            f"started={time.time()}\npid={os.getpid()}\nexpected_sha={sha}\n", encoding="utf-8")


def _record_failure(request: dict, result: Path, code: int, detail: str) -> None:
    from hermes_cli import update_receipt
    from hermes_constants import get_hermes_home

    update_receipt.record_step("historical_takeover", False, detail)
    saved = update_receipt.finalize_pending_update_receipt(code, detail)
    if request.get("gateway_mode"):
        (get_hermes_home() / ".update_exit_code").write_text(f"{code}\n", encoding="utf-8")
    result.write_text(json.dumps({"receipt_handled": saved is not None, "resume_handled": False}), encoding="utf-8")


def main() -> int:
    context, result = map(Path, sys.argv[1:3])
    request = json.loads(context.read_text(encoding="utf-8-sig"))
    sys.path.insert(0, request["root"])
    if "stopped_serves" in request:
        # Historical atexit cleanup may run after the update's result is fixed.
        # It must reuse that installation, never start a second update/repair.
        from hermes_cli._launchers import resolve_store_python
        from pm.environments import activation_environment

        root = Path(request["root"])
        python = resolve_store_python(root)
        if python is None:
            print("Cannot resume stopped backends: no selected Hermes interpreter", file=sys.stderr)
            return 1
        return subprocess.run(
            [str(python), "-I", "-B", "-X", "utf8", str(root / "hermes_cli/update_serve_resume.py"),
             str(context), str(result)], cwd=root, env=activation_environment(root),
        ).returncode
    from hermes_cli import update_receipt
    from hermes_cli.update_lock import UpdateLock, describe_holder

    lock = UpdateLock()
    if not lock.acquire():
        print(describe_holder(lock.holder), file=sys.stderr)
        return 2
    old_receipt = request.get("receipt") or {}
    update_receipt.begin_update_receipt(previous=old_receipt, correlation_id=old_receipt.get("update_id"))
    request["update_id"] = update_receipt.current_correlation_id()
    try:
        python, env = prepare(request)
        request["receipt"] = update_receipt._current.get().data
        context.write_text(json.dumps(request), encoding="utf-8")
        # This file is new too. Direct execution bypasses normal launch-time
        # update liveness checks while the waiting parent still holds its lock.
        command = [str(python), "-I", "-B", "-X", "utf8", str(Path(request["root"]) / "hermes_cli/update_finish.py"),
                   str(context), str(result)]
        code = subprocess.run(command, cwd=request["root"], env=env).returncode
        if code != 0 and not result.is_file():
            _record_failure(request, result, code, f"completion child exited {code} without acknowledgement")
        return code
    except Exception as exc:
        print(f"Update preparation failed: {exc}", file=sys.stderr, flush=True)
        _record_failure(request, result, 1, f"historical takeover preparation failed: {exc}")
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
