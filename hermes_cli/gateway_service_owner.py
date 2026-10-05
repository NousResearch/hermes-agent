"""Who may write a gateway service definition (systemd unit, launchd plist).

A service definition pins the ``HERMES_HOME`` its gateway runs with. Service names are derived from the
home (bare / profile name / path hash), and two homes can still resolve to the same name: a custom root's
``<root>/profiles/<name>`` takes the same ``hermes-gateway-<name>`` unit as ``~/.hermes/profiles/<name>``.
Every gateway boot refreshes "its" unit, so a scratch or E2E gateway started from such a home rewrote the
real install's unit to point at the scratch directory, and the real gateway failed on its next restart.

The rule: a definition that pins a home is written only by a process acting for that home. Uninstall
already refused a unit pinning another home (``_systemd_unit_belongs_to_current_home``); the writers now
refuse the same way. ``hermes gateway install --force-unit-path`` is the explicit repoint.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path


def pinned_home(definition_path: Path) -> str | None:
    """``HERMES_HOME`` pinned by the systemd unit or launchd plist at *definition_path*, or None."""
    if definition_path.suffix == ".plist":
        try:
            with definition_path.open("rb") as fh:
                data = plistlib.load(fh)
        except Exception:  # unreadable / not a plist (hand-edited, truncated): pins nobody
            return None
        env = data.get("EnvironmentVariables") if isinstance(data, dict) else None
        value = env.get("HERMES_HOME") if isinstance(env, dict) else None
        return (str(value).strip() or None) if value is not None else None
    from hermes_cli import gateway as gw
    return gw._hermes_home_pinned_by_unit(definition_path)


def _resolve(raw: str | Path) -> Path | None:
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def service_home_for_unit(unit_path: Path, system: bool) -> Path:
    """The ``HERMES_HOME`` this process acts for when it writes the systemd unit at *unit_path*.

    User scope: its own home. System scope: an explicit ``HERMES_HOME`` remapped to the unit's ``User=``
    account the way ``generate_systemd_unit`` pins it; with none (``sudo`` strips it, HOME=/root), the
    unit's own pinned home, which is the only thing naming the install being operated on. Pure: it never
    adopts the unit's home into ``os.environ``, so a check on this value compares caller and unit.
    """
    from hermes_cli import gateway as gw

    if not system:
        return gw.get_hermes_home()
    if not os.environ.get("HERMES_HOME", "").strip():
        pinned = pinned_home(unit_path) if unit_path.exists() else None
        return Path(pinned).expanduser() if pinned else gw.get_hermes_home()
    user = gw._read_systemd_user_from_unit(unit_path)
    if not user:
        return gw.get_hermes_home()
    try:
        home_dir = gw._system_service_identity(run_as_user=user)[2]
    except ValueError:  # unknown User=: generate_systemd_unit refuses the same way
        return gw.get_hermes_home()
    return Path(gw._hermes_home_for_target_user(home_dir))


def definition_belongs_to_home(definition_path: Path, home: Path, action: str) -> bool:
    """False (with guidance printed) when the existing definition pins a ``HERMES_HOME`` other than *home*.

    A missing definition, or one that pins no home (hand-written, pre-pinning), belongs to nobody else.
    """
    if not definition_path.exists():
        return True
    raw = pinned_home(definition_path)
    if raw is None or _resolve(raw) == _resolve(home):
        return True
    print(f"✗ Refusing to {action} {definition_path}: it runs HERMES_HOME={raw}, "
          f"but this process has HERMES_HOME={home}.")
    print("  That file is another install's gateway. Use that install, or pass --force-unit-path")
    print("  to `hermes gateway install` if you really mean to repoint it at this home.")
    return False


def _installed_definition_pins(home: Path) -> bool:
    """True when a definition at this home's service path already pins *home* (a re-install)."""
    from hermes_cli import gateway as gw

    paths = [gw.get_systemd_unit_path(system=False), gw.get_systemd_unit_path(system=True)]
    if gw.is_macos():
        paths.append(gw.get_launchd_plist_path())
    for path in paths:
        raw = pinned_home(path) if path.exists() else None
        if raw is not None and _resolve(raw) == home:
            return True
    return False


def home_may_install_service(home: Path) -> bool:
    """True for a home that may own a host gateway service: inside the platform-default Hermes tree
    (``~/.hermes`` and its profiles, the sudo invoker's included), the bare-name owner, or a home whose
    installed definition already pins it."""
    from hermes_cli import gateway as gw

    resolved = _resolve(home)
    if resolved is None:
        return False
    if any(resolved == root or root in resolved.parents for root in gw._native_service_homes()):
        return True
    return gw._home_owns_bare_service_name(resolved) or _installed_definition_pins(resolved)


def refuse_foreign_home_install(home: Path, force_unit_path: bool) -> bool:
    """True (with guidance printed) when ``hermes gateway install`` must not register a service for *home*."""
    if force_unit_path or home_may_install_service(home):
        return False
    print(f"✗ Refusing to install a gateway service for HERMES_HOME={home}.")
    print("  It is outside this account's Hermes tree and no service is installed for it yet, so it looks")
    print("  like a scratch / test / E2E home. A service installed for it would outlive that run and restart")
    print("  a gateway on this host forever. Run that gateway in the foreground (`hermes gateway run`),")
    print("  or pass --force-unit-path if this custom home really is meant to own a host service.")
    return True
