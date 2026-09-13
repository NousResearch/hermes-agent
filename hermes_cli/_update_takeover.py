"""Absolute-path bootstrap for historical updater takeover; no app imports yet."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def prepare(request: dict) -> tuple[Path, dict[str, str]]:
    """Provision the new graph before entering its application interpreter."""
    root = Path(request["root"])
    sys.path.insert(0, str(root))
    from pm import paths, receipt
    from pm.client import ensure, sync_venv, venv_is_current
    from pm.lock import Lockfile
    from pm.registry import get_package
    from hermes_cli.runtime_paths import activation_environment, install_state_dir, runtime_facts_path
    from hermes_cli._launchers import resolve_store_python
    from hermes_cli.venv_sync import publish_launchers

    correlation = request["update_id"]
    with receipt.worker_context(correlation):
        lock = Lockfile(paths.lockfile_path())
        # A pre-PM installation has no required-tool facts. A current Python
        # generation alone does not prove its Node/Git/tool closure is ready.
        for name in lock.names():
            if not get_package(name).optional or name == "python":
                ensure(name, explicit=True)
        extras = ["all"] if not runtime_facts_path(root).is_file() else None
        repair_marker = install_state_dir(root) / ".repair-incomplete"
        # Repair preserves the old stamp. Changed source inputs instead need
        # an ordinary sync, which builds and validates a fresh generation too.
        repair = repair_marker.is_file() and venv_is_current(project_root=root)
        sync_venv(None if repair else extras, explicit=True, project_root=root, repair=repair)
        request["pm_receipt"] = receipt.last_for_update(correlation)
    publish_launchers(root)
    repair_marker.unlink(missing_ok=True)
    for name in (".update-incomplete", ".lazy-refresh-incomplete"):
        (root / name).unlink(missing_ok=True)
    python = resolve_store_python(root)
    if python is None:
        raise RuntimeError("updated installation has no managed interpreter")
    return python, activation_environment(root)


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
    request = json.loads(context.read_text(encoding="utf-8"))
    sys.path.insert(0, request["root"])
    if "stopped_serves" in request:
        # Historical atexit cleanup may run after the update's result is fixed.
        # It must reuse that installation, never start a second update/repair.
        from hermes_cli._launchers import resolve_store_python
        from hermes_cli.runtime_paths import activation_environment

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
