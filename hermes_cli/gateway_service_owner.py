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
import sys
from pathlib import Path
from xml.parsers.expat import ExpatError


def pinned_home(definition_path: Path) -> str | None:
    """``HERMES_HOME`` pinned by the systemd unit or launchd plist at *definition_path*, or None."""
    if definition_path.suffix == ".plist":
        try:
            with definition_path.open("rb") as fh:
                data = plistlib.load(fh)
        except (OSError, ValueError, ExpatError):  # unreadable / not a plist: pins nobody
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


def unit_belongs_to_caller(unit_path: Path, system: bool, action: str, run_as_user: str | None = None) -> bool:
    """``definition_belongs_to_home`` for the systemd unit at *unit_path*, judged against the home this
    process would pin there (``run_as_user``'s for an explicit ``--system --run-as-user`` install)."""
    from hermes_cli import gateway as gw

    home = service_home_for_unit(unit_path, system)
    if system and run_as_user:
        try:
            home = Path(gw._hermes_home_for_target_user(gw._system_service_identity(run_as_user)[2]))
        except ValueError:
            pass  # unknown account: generate_systemd_unit refuses it later with the real error
    return definition_belongs_to_home(unit_path, home, action)


def foreign_systemd_unit_home(system: bool) -> str | None:
    """``HERMES_HOME`` of the unit at this scope when it belongs to ANOTHER install, else None.

    A throwaway ``HOME`` used to resolve the bare ``hermes-gateway`` name (its ``.hermes`` looked native) and
    reach the account's real unit — user scope through the account-anchored unit dir, system scope
    because system units are global. Naming now follows the account home, but two homes can still share
    a name (a custom root's ``profiles/<name>``, a pre-pinning unit), so every systemd verb asks this.
    ``_sync_hermes_home_from_systemd_unit`` would adopt the unit's home and ``stop``/``restart`` would take
    the live gateway down, so it must be evaluated BEFORE that sync.

    Foreign means: the unit pins a home, it differs from this process's home, and this process's home
    is not the account's own default root. The last clause keeps the ``sudo`` remap (#108674): root's
    default home operating a unit that pins the invoking user's home is the operator, not a stranger.
    """
    from hermes_cli import gateway as gw
    unit_home = gw._hermes_home_pinned_by_unit(gw.get_systemd_unit_path(system=system))
    if not unit_home:
        return None
    try:
        unit = Path(unit_home).expanduser().resolve()
        mine = gw.get_hermes_home().resolve()
    except OSError:
        return None
    if unit == mine:
        return None
    from hermes_constants import account_default_hermes_home
    if mine == account_default_hermes_home().resolve():
        return None
    return unit_home


def refuse_foreign_systemd_unit(action: str, system: bool) -> None:
    """Exit non-zero instead of operating another install's unit (see ``foreign_systemd_unit_home``)."""
    from hermes_cli import gateway as gw
    unit_home = foreign_systemd_unit_home(system)
    if unit_home is None:
        return
    gw.print_error(
        f"Refusing to {action} {gw.get_service_name()}: that {gw._service_scope_label(system)} unit runs "
        f"HERMES_HOME={unit_home}, but this process has HERMES_HOME={gw.get_hermes_home()}.\n"
        f"It belongs to another Hermes install. Run this from that install's HERMES_HOME."
    )
    sys.exit(1)


def refuse_foreign_launchd_plist(verb: str) -> None:
    """The launchd half of ``refuse_foreign_systemd_unit``: exit 1 rather than start/stop/restart/remove a
    LaunchAgent pinning another HERMES_HOME. (Windows task names derive from LOCALAPPDATA, not HOME.)"""
    from hermes_cli import gateway as gw
    if not definition_belongs_to_home(gw.get_launchd_plist_path(), gw.get_hermes_home(), verb):
        sys.exit(1)


def service_definition_homes(definition: str) -> list[str]:
    """Every HERMES_HOME a gateway service definition sets: systemd ``Environment=``, launchd
    ``EnvironmentVariables``, the Windows ``gateway.cmd`` ``set`` line and the ``gateway.vbs`` env item."""
    import re
    homes = re.findall(r'HERMES_HOME=([^"\r\n]+)', definition)
    homes += re.findall(r"<key>HERMES_HOME</key>\s*<string>(.*?)</string>", definition, flags=re.DOTALL)
    homes += re.findall(r'Item\("HERMES_HOME"\)\s*=\s*"([^"\r\n]+)"', definition)
    return homes


def service_definition_launchers(definition: str) -> list[str]:
    """The scripts a definition starts, by their quoted absolute paths. The Windows Startup-folder
    fallback and a Scheduled Task's action name no home: they start the home's ``gateway.vbs``, which does."""
    import re
    found = re.findall(r'"((?:[A-Za-z]:[\\/]|\\\\|/)[^"\r\n]*\.(?:vbs|cmd))"', definition, flags=re.IGNORECASE)
    return list(dict.fromkeys(found))


def temp_home_in_service_definition(definition: str) -> str | None:
    """Temp-dir HERMES_HOME baked into a systemd unit / launchd plist, or None. A temp home means a
    test/E2E harness generated it; installing it leaves the gateway "running" but deaf to every platform."""
    import tempfile
    candidates = service_definition_homes(definition)
    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp"), Path("/var/tmp"), Path("/private/tmp"), Path("/private/var/tmp"),  # no-tmp: ok — detects a temp HERMES_HOME in service definitions
    }
    for raw in candidates:
        try:
            resolved = Path(raw.strip().strip('"')).resolve()
        except (OSError, ValueError):
            continue
        if any(resolved == root or root in resolved.parents for root in temp_roots):
            return raw.strip()
    return None


def refuse_temp_home_service_write(definition: str, kind: str) -> bool:
    """Refuse (with guidance) when a service definition carries a temp HERMES_HOME."""
    temp_home = temp_home_in_service_definition(definition)
    if temp_home is None:
        return False
    print(f"✗ Refusing to write the gateway {kind}: HERMES_HOME resolves to a temporary directory ({temp_home}).")
    print(
        "  This usually means a test/E2E environment exported HERMES_HOME. "
        "Unset it (or run from a clean shell) and retry."
    )
    return True
