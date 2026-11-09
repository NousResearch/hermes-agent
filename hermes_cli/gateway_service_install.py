"""Persisted service definition refreshes for systemd and launchd."""
from __future__ import annotations


def _gateway():
    from hermes_cli import gateway
    return gateway


def _retire_hermes_replace_dropin(system: bool = False) -> bool:
    gw = _gateway()
    unit_path = gw.get_systemd_unit_path(system=system)
    dropin = unit_path.parent / f"{unit_path.name}.d" / "20-replace.conf"
    try:
        text = dropin.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    if not all(token in text for token in ("Added to end the gateway respawn storm", "--replace", "ExecStart=")):
        return False
    dropin.unlink()
    return True


def refresh_systemd_unit_if_needed(system: bool = False) -> bool:
    """Refresh only a recognized serving unit owned by this Hermes home."""
    gw = _gateway()
    unit_path = gw.get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False
    from hermes_cli.gateway_service_mode import systemd_definition_mode
    if systemd_definition_mode(unit_path) != "serving":
        return False
    from hermes_cli.gateway_service_owner import unit_belongs_to_caller
    if not unit_belongs_to_caller(unit_path, system, "rewrite"):
        return False

    current = gw.systemd_unit_is_current(system=system)
    if _retire_hermes_replace_dropin(system=system):
        gw._run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
        print(f"↻ Removed the stale Hermes --replace drop-in from the gateway {gw._service_scope_label(system)} service")
        if current:
            return True
    elif current:
        return False

    expected_user = gw._read_systemd_user_from_unit(unit_path) if system else None
    new_unit = gw.generate_systemd_unit(system=system, run_as_user=expected_user)
    if not system and any(m in new_unit for m in ("/pytest-of-", '/hermes_test"', "/hermes_test/")):
        return False
    from hermes_cli import gateway_service_owner
    if gateway_service_owner.refuse_temp_home_service_write(new_unit, "systemd unit"):
        return False
    gw._prepare_service_launcher(system=system, run_as_user=expected_user)
    unit_path.write_text(new_unit, encoding="utf-8")
    gw._run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"↻ Updated gateway {gw._service_scope_label(system)} service definition to match the current Hermes install")
    return True
