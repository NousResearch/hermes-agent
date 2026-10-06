"""Keep the whole source updater outside a launching systemd service's cgroup."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import uuid


def _has_terminal_update_receipt(directory: Path, action_id: str) -> bool:
    """An exact update archive, not a shared latest/PM pointer, owns the result."""
    for archive in directory.glob(f"update_*_{action_id}.json"):
        try:
            data = json.loads(archive.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("kind") or data.get("update_id") != action_id:
            continue
        if data.get("finished_at") and data.get("outcome") in {"success", "partial", "failed", "refused"}:
            return True
    return False


def isolate_update_process() -> None:
    """Launch before receipt/lock creation, retaining argv, cwd and inherited stdio.

    setsid only changes the session, not systemd ownership. Moving just the
    completion leaves its waiting updater parent (and lock/output) killable.
    A lock-free observer retains responsibility for scope-launch failures.
    """
    if sys.platform != "linux" or not os.environ.get("INVOCATION_ID"):
        return
    # INVOCATION_ID is inherited even after scope entry; consult the closest
    # real unit to avoid recursively relaunching, not an environment sentinel.
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    for line in cgroup.splitlines():
        for part in reversed(line.split(":", 2)[-1].split("/")):
            if part.endswith(".scope"):
                return
            if part.endswith(".service"):
                break
        else:
            continue
        break
    else:
        return
    from tools.process_registry import (
        _build_systemd_scope_argv, _systemd_run_user_scope_available, systemd_user_bus_env,
    )

    command = list(sys.orig_argv)
    scoped = (_build_systemd_scope_argv(command, unit_suffix=f"update-{uuid.uuid4().hex}")
              if _systemd_run_user_scope_available() else command)
    if scoped == command:
        from hermes_cli.update_receipt import begin_update_receipt, finalize_pending_update_receipt

        message = ("Cannot start update safely from this systemd service: a restart-safe user scope "
                   "is unavailable. Run hermes update from an external shell, or enable the service "
                   "user's systemd session bus (loginctl enable-linger <user>).")
        print(f"✗ {message}", file=sys.stderr)
        action_id = os.environ.get("HERMES_ACTION_ID", "")
        correlation_id = action_id if re.fullmatch(r"[0-9a-f]{32}", action_id) else None
        begin_update_receipt(correlation_id=correlation_id)
        finalize_pending_update_receipt(1, message)
        raise SystemExit(1)
    env = systemd_user_bus_env()
    # Before 254 scopes executed argv literally and rejected this option. Unknown
    # versions retain the disabling flag: failure is safer than expanded argv.
    literal_legacy_scope = False
    try:
        version = subprocess.run([scoped[0], "--version"], capture_output=True,
                                 text=True, timeout=3, env=env)
        match = re.match(r"systemd ([0-9]+)(?:\s|$)", version.stdout)
        literal_legacy_scope = (version.returncode == 0 and match is not None
                                and int(match[1]) < 254)
    except (OSError, subprocess.SubprocessError):
        pass
    if not literal_legacy_scope:
        scoped.insert(scoped.index("--"), "--expand-environment=no")
    # This observer owns only launcher failure, never the updater lock/pipeline.
    # The actual updater (including its completion parent/output) lives in the
    # scope. Restarting the old service may kill this waiter without affecting it.
    # Capture receipt APIs/home before launching: the child may swap the checkout.
    from hermes_cli.update_receipt import begin_update_receipt, finalize_pending_update_receipt
    from hermes_constants import get_hermes_home

    action_id = env.get("HERMES_ACTION_ID", "")
    correlation_id = action_id if re.fullmatch(r"[0-9a-f]{32}", action_id) else uuid.uuid4().hex
    env["HERMES_ACTION_ID"] = correlation_id
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    try:
        exit_code = subprocess.Popen(scoped, env=env).wait()
    except OSError as exc:
        exit_code = 1
        message = f"Cannot start update safely: scope execution failed: {exc}. Run hermes update from an external shell."
    else:
        exit_code = exit_code if exit_code >= 0 else 128 - exit_code
        if exit_code == 0:
            raise SystemExit(0)
        message = (f"Cannot start update safely: scope launcher exited {exit_code} without a terminal "
                   "update receipt. Run hermes update from an external shell.")
    if _has_terminal_update_receipt(receipt_dir, correlation_id):
        raise SystemExit(exit_code)
    print(f"✗ {message}", file=sys.stderr)
    begin_update_receipt(correlation_id=correlation_id)
    finalize_pending_update_receipt(exit_code, message)
    raise SystemExit(exit_code)
