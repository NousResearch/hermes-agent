"""Systemd gateway install/start/stop/uninstall lifecycle."""
from __future__ import annotations

import subprocess
from pathlib import Path

from gateway import (
    service_identity,
    signal_restart,
    systemd_identity,
    systemd_legacy,
    systemd_linger,
    systemd_runtime,
    systemd_unit_render,
    systemd_unit_state,
)
from hermes_constants import get_hermes_home


def prepare_action(
    action: str,
    system: bool,
    *,
    require_installed: bool = True,
    preflight: bool = False,
) -> bool:
    system = systemd_runtime.select_scope(system)
    if system:
        systemd_identity.require_root(action)
    elif preflight:
        systemd_runtime.preflight_user()
    if require_installed and not systemd_identity.unit_path(system).exists():
        print("✗ Gateway service is not installed")
        raise SystemExit(1)
    return system


def unit_belongs_to_current_home(system: bool = False) -> bool:
    systemd_runtime.sync_home_from_unit(system)
    pinned = service_identity.hermes_home_pinned_by_unit(systemd_identity.unit_path(system))
    if pinned is None:
        return True
    return Path(pinned).expanduser().resolve() == get_hermes_home().resolve()


def install(
    *,
    force: bool = False,
    system: bool = False,
    run_as_user: str | None = None,
    enable_on_startup: bool = True,
    remove_legacy: bool = False,
) -> None:
    if system:
        systemd_identity.require_root("install")
    if remove_legacy:
        systemd_legacy.remove_units()

    path = systemd_identity.unit_path(system)
    if path.exists():
        systemd_runtime.sync_home_from_unit(system)
    if path.exists() and not force:
        if not systemd_unit_state.unit_is_current(system):
            systemd_unit_state.refresh_if_needed(system)
            if enable_on_startup:
                systemd_runtime.run_systemctl(
                    ["enable", service_identity.service_name()],
                    system=system,
                    check=True,
                    timeout=30,
                )
        configured = systemd_identity.read_unit_user(path) if system else None
        if configured:
            systemd_linger.ensure_system_service_linger(configured)
        elif not system:
            systemd_linger.ensure_linger_enabled()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    definition = systemd_unit_render.generate_systemd_unit(system=system, run_as_user=run_as_user)
    if systemd_unit_state.refuse_temp_home_write(definition, "systemd unit"):
        return
    path.write_text(definition, encoding="utf-8")
    systemd_runtime.run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    if enable_on_startup:
        systemd_runtime.run_systemctl(
            ["enable", service_identity.service_name()],
            system=system,
            check=True,
            timeout=30,
        )
    configured = systemd_identity.read_unit_user(path) if system else None
    if configured:
        systemd_linger.ensure_system_service_linger(configured)
    elif not system:
        systemd_linger.ensure_linger_enabled()


def uninstall(system: bool = False) -> None:
    system = prepare_action("uninstall", system, require_installed=False)
    if not unit_belongs_to_current_home(system):
        return
    systemd_runtime.run_systemctl(
        ["stop", service_identity.service_name()],
        system=system,
        check=False,
        timeout=90,
    )
    systemd_runtime.run_systemctl(
        ["disable", service_identity.service_name()],
        system=system,
        check=False,
        timeout=30,
    )
    path = systemd_identity.unit_path(system)
    if path.exists():
        path.unlink()
    systemd_runtime.run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)


def start(system: bool = False) -> None:
    system = prepare_action("start", system, preflight=True)
    systemd_unit_state.refresh_if_needed(system)
    systemd_runtime.run_systemctl(
        ["start", service_identity.service_name()],
        system=system,
        check=True,
        timeout=30,
    )


def stop(system: bool = False) -> None:
    system = prepare_action("stop", system)
    systemd_runtime.sync_home_from_unit(system)
    signal_restart._mark_planned_stop()
    try:
        systemd_runtime.run_systemctl(
            ["stop", service_identity.service_name()],
            system=system,
            check=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return


def service_call(verb: str, system: bool = False) -> None:
    if verb == "start":
        start(system)
    elif verb == "stop":
        stop(system)
    elif verb == "restart":
        from gateway.systemd_restart import systemd_restart
        systemd_restart(system)
    elif verb == "uninstall":
        uninstall(system)
    else:
        raise ValueError(f"Unsupported systemd service verb: {verb}")
