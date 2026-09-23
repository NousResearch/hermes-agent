"""Non-installing service-manager operations for an explicit canonical profile.

The legacy start helpers refresh units, regenerate plists, or fall back to
unmanaged processes. Client startup must not call those helpers.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from xml.parsers.expat import ExpatError

from gateway.runtime_contract import RuntimeState
from hermes_cli._subprocess_compat import windows_hide_flags


class RuntimeStartError(RuntimeError):
    def __init__(self, reason: str, state: RuntimeState = "inaccessible"):
        super().__init__(reason)
        self.reason, self.state = reason, state


def remaining(deadline: float) -> float:
    budget = deadline - time.monotonic()
    if budget <= 0:
        raise TimeoutError
    return budget


def service_suffix(home: Path) -> str:
    """Host-service suffix for the REQUESTED home, never the initiating client's env.

    Same rule as ``hermes_cli.gateway._profile_suffix``: only the platform-native
    default home owns the bare unit name; ``<root>/profiles/<name>`` yields the
    profile name; any other root (Docker, a temp harness) yields a path hash so a
    custom root can never resolve to the production ``hermes-gateway`` unit.
    """
    from hermes_constants import _get_platform_default_hermes_home
    from hermes_cli.gateway import _profile_name_from_home
    default = _get_platform_default_hermes_home().resolve()
    home = home.resolve()
    if home == default:
        return ""
    root = default
    if not home.is_relative_to(default):
        root = home.parent.parent if home.parent.name == "profiles" else home
    if home != root:
        name = _profile_name_from_home(home, root)
        if name:
            return name
    return hashlib.sha256(str(home).encode()).hexdigest()[:8]


@dataclass(frozen=True)
class ExistingService:
    backend: str
    start_argv: tuple[str, ...]
    running: bool = False


def _run(argv: list[str] | tuple[str, ...], deadline: float, *, encoding: str | None = "utf-8"):
    return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                          text=encoding is not None, encoding=encoding, errors="strict" if encoding else None,
                          creationflags=windows_hide_flags(), timeout=remaining(deadline))


def _verify_binding(verify, *args, **kwargs):
    try:
        return verify(*args, **kwargs)
    except (ValueError, TypeError, KeyError, AttributeError, ExpatError) as exc:
        reason = str(exc)
        raise RuntimeStartError(reason if reason in {
            "profile_mismatch", "service_account_mismatch", "service_disabled"
        } else "service_identity_unverified") from None


def _exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _systemd(home: Path, deadline: float) -> ExistingService | None:
    from hermes_cli import gateway as gw
    from hermes_cli.service_manager import _s6_running
    from hermes_cli.gateway_runtime_service_identity import SYSTEMD_IDENTITY_PROPERTIES, verify_systemd
    if _s6_running():
        raise RuntimeStartError("external_supervisor")
    suffix = service_suffix(home)
    unit = f"hermes-gateway{'-' + suffix if suffix else ''}.service"
    paths = (Path.home() / ".config/systemd/user" / unit, Path("/etc/systemd/system") / unit)
    # On non-systemd hosts there is no manager to own transient units. Installed
    # definitions still count: a broken manager is not permission to bypass one.
    manager_paths = (Path("/run/systemd/system"),
                     Path(f"/run/user/{os.getuid()}/systemd/private"))  # windows-footgun: ok — native systemd only
    if not any(_exists(p) for p in (*paths, *manager_paths)):
        return None
    found = []
    for system in (False, True):
        command = gw._systemctl_cmd(system)
        result = _run([*command, "show", unit, "--no-pager", "--all",
                       "--property=LoadState,ActiveState,SubState,UnitFileState," + ",".join(SYSTEMD_IDENTITY_PROPERTIES)], deadline)
        props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if props.get("LoadState") == "not-found" and not _exists(paths[int(system)]):
            continue
        if result.returncode or props.get("LoadState") != "loaded":
            raise RuntimeStartError("service_manager_unavailable")
        if props.get("UnitFileState") in {"masked", "masked-runtime"}:
            raise RuntimeStartError("service_masked")
        state = props.get("ActiveState")
        if state == "deactivating":
            raise RuntimeStartError("service_draining", "draining")
        if state not in {"inactive", "active", "activating", "reloading", "failed"}:
            raise RuntimeStartError("service_state_unknown")
        if state == "failed":
            raise RuntimeStartError("service_failed")
        environment = _run([*command, "show-environment"], deadline)
        if environment.returncode:
            raise RuntimeStartError("service_identity_unverified")
        try:
            verify_systemd(props, environment.stdout, home, system=system)
        except ValueError as exc:
            reason = str(exc)
            raise RuntimeStartError(reason if reason in {
                "profile_mismatch", "service_account_mismatch"
            } else "service_identity_unverified") from None
        found.append(ExistingService("systemd", tuple([*command, "--no-ask-password", "start", unit]),
                                     state != "inactive"))
    if len(found) > 1:
        raise RuntimeStartError("service_scope_conflict", "conflict")
    return found[0] if found else None


def _launchd(home: Path, deadline: float) -> ExistingService | None:
    import pwd
    suffix = service_suffix(home)
    label = f"ai.hermes.gateway{'-' + suffix if suffix else ''}"
    account = pwd.getpwuid(os.getuid())  # windows-footgun: ok — native launchd only
    account_home = Path(account.pw_dir)
    plist = account_home / "Library/LaunchAgents" / f"{label}.plist"
    from hermes_cli.gateway_runtime_service_identity import verify_launchd_loaded, verify_launchd_plist, read_definition
    installed = _exists(plist)
    domains = [f"gui/{os.getuid()}", f"user/{os.getuid()}"]  # windows-footgun: ok — native launchd only
    found = []
    loaded = []
    for domain in domains:
        result = _run(["launchctl", "print", f"{domain}/{label}"], deadline)
        if result.returncode == 0:
            loaded.append(result.stdout)
            found.append(ExistingService("launchd", ("launchctl", "kickstart", f"{domain}/{label}"),
                                         "state = running" in result.stdout))
            continue
        # launchctl's native absent-service code; other errors retain uncertainty.
        if result.returncode != 113:
            raise RuntimeStartError("service_manager_unavailable")
    if len(found) > 1:
        raise RuntimeStartError("service_scope_conflict", "conflict")
    if found:
        _verify_binding(verify_launchd_loaded, loaded[0], home, uid=os.getuid(), username=account.pw_name)  # windows-footgun: ok — native launchd only
        return found[0]
    if installed:
        definition = _verify_binding(plistlib.loads, read_definition(plist))
        _verify_binding(verify_launchd_plist, definition, label, home)
        # Load ONLY the existing file, never bootout/rewrite or kickstart -k.
        return ExistingService("launchd", ("launchctl", "bootstrap", domains[0], str(plist)))
    return None


def _windows(home: Path, deadline: float) -> ExistingService | None:
    from hermes_cli.gateway_windows import _startup_dir, _schtasks_encoding
    suffix = service_suffix(home)
    name = f"Hermes_Gateway{'_' + suffix if suffix else ''}"
    result = _run(["schtasks.exe", "/Query", "/FO", "CSV", "/NH"], deadline, encoding=_schtasks_encoding())
    if result.returncode:
        raise RuntimeStartError("service_manager_unavailable")
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if any(len(row) != 3 for row in rows if row):
        raise RuntimeStartError("service_state_unknown")
    if any(row and row[0].lstrip("\\") == name for row in rows):
        from hermes_cli.gateway_runtime_service_identity import verify_windows_task
        definition = _run(["schtasks.exe", "/Query", "/TN", name, "/XML"], deadline, encoding=None)
        identity = _run(["whoami.exe", "/USER", "/FO", "CSV", "/NH"], deadline, encoding=_schtasks_encoding())
        if definition.returncode or identity.returncode:
            raise RuntimeStartError("service_identity_unverified")
        accounts = list(csv.reader(io.StringIO(identity.stdout)))
        if len(accounts) != 1 or len(accounts[0]) != 2:
            raise RuntimeStartError("service_identity_unverified")
        _verify_binding(verify_windows_task, definition.stdout, home, *accounts[0])
        return ExistingService("windows", ("schtasks.exe", "/Run", "/TN", name))
    # Startup-folder entries are installed persistence too, but have no independent
    # start supervisor. Do not bypass them with a job-bound unmanaged child.
    if any(_exists(_startup_dir() / f"{name}.{ext}") for ext in ("cmd", "vbs")):
        raise RuntimeStartError("startup_service_requires_login")
    return None


def discover_existing_gateway_service(home: Path, *, deadline: float) -> ExistingService | None:
    backend = {"linux": _systemd, "darwin": _launchd, "win32": _windows}.get(sys.platform)
    if backend is None:
        raise RuntimeStartError("unsupported_supervisor_platform")
    try:
        return backend(home, deadline)
    except subprocess.TimeoutExpired:
        raise TimeoutError from None
    except (OSError, ValueError) as exc:
        raise RuntimeStartError("service_manager_unavailable") from exc


def start_existing_gateway_service(service: ExistingService, *, deadline: float) -> None:
    """Request start once; only live control readiness can establish success."""
    if service.running:
        return
    try:
        result = _run(service.start_argv, deadline, encoding=None)
    except subprocess.TimeoutExpired:
        raise TimeoutError from None
    except OSError as exc:
        raise RuntimeStartError("service_start_failed") from exc
    if result.returncode:
        raise RuntimeStartError("service_start_failed")
