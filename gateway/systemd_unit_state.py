"""Installed systemd unit comparison and refresh."""
from __future__ import annotations

from pathlib import Path

from gateway import systemd_identity, systemd_runtime, systemd_unit_render
from gateway.service_definition import (
    normalize_service_definition,
    refuse_temp_home_write,
    temp_home_in_definition,
)


def unit_is_current(system: bool = False) -> bool:
    systemd_runtime.sync_home_from_unit(system)
    path = systemd_identity.unit_path(system)
    if not path.exists():
        return False
    installed = path.read_text(encoding="utf-8")
    expected_user = systemd_identity.read_unit_user(path) if system else None
    expected = systemd_unit_render.generate_systemd_unit(system=system, run_as_user=expected_user)

    def norm(text: str) -> str:
        return normalize_service_definition(
            systemd_unit_render.strip_optional_systemd_directives(text)
        )

    return norm(installed) == norm(expected)


def retire_replace_dropin(system: bool = False) -> bool:
    path = systemd_identity.unit_path(system)
    dropin = path.parent / f"{path.name}.d" / "20-replace.conf"
    try:
        text = dropin.read_text(encoding="utf-8")
    except OSError:
        return False
    if not all(
        token in text
        for token in (
            "Added to end the gateway respawn storm",
            "--replace",
            "ExecStart=",
        )
    ):
        return False
    dropin.unlink()
    return True


def refresh_if_needed(system: bool = False) -> bool:
    path = systemd_identity.unit_path(system)
    if not path.exists():
        return False

    current = unit_is_current(system)
    if retire_replace_dropin(system):
        systemd_runtime.run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
        print(
            "↻ Removed the stale Hermes --replace drop-in from the gateway "
            f"{systemd_runtime.scope_label(system)} service"
        )
        if current:
            return True
    elif current:
        return False

    expected_user = systemd_identity.read_unit_user(path) if system else None
    new_unit = systemd_unit_render.generate_systemd_unit(system=system, run_as_user=expected_user)

    if not system and any(
        marker in new_unit
        for marker in ("/pytest-of-", '/hermes_test"', "/hermes_test/")
    ):
        return False
    if refuse_temp_home_write(new_unit, "systemd unit"):
        return False

    path.write_text(new_unit, encoding="utf-8")
    systemd_runtime.run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(
        f"↻ Updated gateway {systemd_runtime.scope_label(system)} service definition "
        "to match the current Hermes install"
    )
    return True
