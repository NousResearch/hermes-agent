"""Profile delete/rename: remove the OLD name's supervisor registrations that live outside
the user systemd / launchd arms in ``profiles._cleanup_gateway_service``.

Every registration is keyed on the profile name and starts ``--profile <old>`` with
``HERMES_HOME`` pinned to a directory that the delete removed or the rename moved, so a
survivor crash-loops (KeepAlive / Restart= / Task Scheduler retry) at the next boot or login.
Callers bind the profile's home (``set_hermes_home_override``) before calling so the
name-deriving helpers resolve THAT profile, never the ambient one.
"""
from __future__ import annotations

import os
import subprocess


def remove_system_systemd_unit() -> bool:
    """Remove ``/etc/systemd/system/hermes-gateway-<profile>.service`` when root; otherwise
    name the leftover and the exact command, because the profile dir the ``gateway uninstall
    --system`` verb would need to resolve the name from is about to vanish."""
    from hermes_cli import gateway
    svc_name = gateway.get_service_name()
    unit = gateway._SYSTEM_UNIT_DIR / f"{svc_name}.service"
    if not unit.exists():
        return False
    if os.geteuid() != 0:  # windows-footgun: ok — called only from the Linux arm of profiles._cleanup_gateway_service
        print(f"⚠ System service {svc_name} remains at {unit}; it will restart the removed profile at boot.")
        print(f"  Remove it with: sudo systemctl disable --now {svc_name} && sudo rm {unit} && sudo systemctl daemon-reload")
        return False
    subprocess.run(["systemctl", "disable", svc_name], capture_output=True, check=False, timeout=30)
    stopped = subprocess.run(["systemctl", "stop", svc_name], capture_output=True, check=False, timeout=30)
    unit.unlink(missing_ok=True)
    subprocess.run(["systemctl", "daemon-reload"], capture_output=True, check=False, timeout=30)
    if getattr(stopped, "returncode", 0) != 0:
        # The unit file is gone (it must not resurrect the removed profile at boot), but the
        # gateway it supervised may still be running under the old name.
        print(f"⚠ System service {svc_name} unit removed, but `systemctl stop` exited {stopped.returncode}; "
              f"its gateway may still be running. Stop it with: sudo systemctl stop {svc_name}")
        return True
    print(f"✓ System service {svc_name} removed")
    return True


def remove_windows_task() -> bool:
    """Delete the profile's Scheduled Task and Startup-folder login item. Direct ``schtasks
    /Delete`` rather than ``gateway_windows.uninstall()``: that verb may open a UAC prompt,
    which a dashboard DELETE or a scripted rename must never block on."""
    from hermes_cli import gateway_windows as gw
    task_name = gw.get_task_name()
    removed = False
    if gw.is_task_registered():
        code, _out, err = gw._exec_schtasks(["/Delete", "/F", "/TN", task_name])
        if code == 0:
            removed = True
            print(f"✓ Removed Scheduled Task {task_name!r}")
        else:
            print(f"⚠ Scheduled Task {task_name!r} remains (schtasks code {code}): {err.strip()}")
    for path in (gw.get_startup_entry_path(), gw._legacy_startup_entry_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed = True
        print(f"✓ Removed Windows login item: {path}")
    return removed
