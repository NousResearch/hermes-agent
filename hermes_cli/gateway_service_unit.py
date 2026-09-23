"""Gateway service-unit definitions: systemd unit generation, staleness comparison and refresh.

Extracted from ``hermes_cli/gateway.py``. Bodies read facade helpers through ``_gw()`` (late
binding on ``hermes_cli.gateway``) so the seams tests and callers patch on the facade keep
intercepting the moved code.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


def _systemd_env_line(name: str, value: str) -> str:
    """One ``Environment="NAME=value"`` line: ``\\`` and ``"`` escaped for systemd's quoting, ``%``
    doubled so specifier expansion leaves the value alone."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def _installed_unit_ld_library_path(system: bool) -> str:
    """``LD_LIBRARY_PATH`` baked into the installed unit; ``""`` when absent."""
    return _gw()._unit_environment_value(_gw().get_systemd_unit_path(system=system), "LD_LIBRARY_PATH") or ""


def _ld_library_path_line(system: bool, target_home_dir: str | None = None) -> str:
    """Carry the installer's LD_LIBRARY_PATH into the unit (glibc reads it only at process start, so
    ~/.hermes/.env is too late for CUDA libs — #14613); system units remap caller-home components.

    The installed unit is the fallback source: the unit is regenerated and compared on every
    start/restart/status, and a shell without the export (ssh, cron, ``sudo`` strips ``LD_*``)
    must not be able to "repair" the line away."""
    raw = os.environ.get("LD_LIBRARY_PATH", "") or _gw()._installed_unit_ld_library_path(system)
    components = [p for p in raw.split(":") if p]
    if target_home_dir is not None:
        components = [_gw()._remap_path_for_user(p, target_home_dir) for p in components]
    return _gw()._systemd_env_line("LD_LIBRARY_PATH", ":".join(components)) if components else ""


def _hermes_home_for_target_user(target_home_dir: str) -> str:
    """Remap the current HERMES_HOME (root's, under sudo) to the target user's equivalent:
    ``/root/.hermes[/profiles/x]`` → ``/home/alice/.hermes[/profiles/x]``; custom paths kept as-is."""
    current_hermes_raw = os.environ.get("HERMES_HOME", "").strip()
    current_hermes = Path(current_hermes_raw).expanduser() if current_hermes_raw else _gw().get_hermes_home()
    # Keep paths lexical: resolving a non-existent path can bake a different HERMES_HOME into the unit.
    current_default = Path.home() / ".hermes"
    target_default = Path(target_home_dir) / ".hermes"
    try:
        # Default ~/.hermes or a profile/subdir of it → preserve the relative structure under the target.
        return str(target_default / current_hermes.relative_to(current_default))
    except ValueError:
        return str(current_hermes)  # Completely custom path (not under ~/.hermes) — keep as-is


def _build_service_path_dirs(project_root: Path | None = None) -> list[str]:
    """Build PATH directory list for service units, excluding non-existent dirs."""
    if project_root is None:
        project_root = _gw().PROJECT_ROOT

    def _is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    candidates = []
    venv_bin = project_root / "venv" / "bin"
    if _is_dir(venv_bin):
        candidates.append(str(venv_bin))
    elif sys.prefix != sys.base_prefix:
        candidates.append(str(Path(sys.prefix) / "bin"))

    hermes_home = _gw().get_hermes_home()
    extras = (project_root / "node_modules" / ".bin", hermes_home / "node" / "bin", hermes_home / "node_modules" / ".bin")
    for extra in extras:
        if _is_dir(extra):
            candidates.append(str(extra))
    return candidates


def _stable_service_working_dir() -> str:
    """WorkingDirectory that won't disappear under systemd (HERMES_HOME, else _gw().PROJECT_ROOT). cwd is
    irrelevant to ``-m`` resolution, and a pinned transient checkout rots: systemd fails at CHDIR
    (status=200) before Python loads, so the unit self-heal never runs and Restart=always crash-loops."""
    try:
        home = _gw().get_hermes_home()
        if home and Path(home).is_dir():
            return str(Path(home).resolve())
    except Exception:
        pass
    return str(_gw().PROJECT_ROOT)


def _systemd_watchdog_seconds(hermes_home: str | Path | None = None) -> int:
    """Resolve the managed-overlay-aware watchdog setting for a service home."""
    override_token = reset_home_override = None
    if hermes_home is not None:
        from hermes_constants import (reset_hermes_home_override, set_hermes_home_override)
        override_token = set_hermes_home_override(hermes_home)
        reset_home_override = reset_hermes_home_override
    try:
        config = _gw().load_gateway_config()
        return _gw().coerce_systemd_watchdog_seconds(getattr(config, "systemd_watchdog_seconds", 0))
    except Exception:
        logger.debug("Could not resolve effective systemd watchdog configuration", exc_info=True)
        return 0
    finally:
        if override_token is not None and reset_home_override is not None:
            reset_home_override(override_token)


def _append_node_dir_for_service(path_entries: list[str], hermes_root: Path | None = None) -> None:
    """Append the Node dir a service unit should use: managed ``<hermes_root>/node`` (profile-scoped)
    first — a unit survives reboots, so baking a shell-PATH Node is permanent breakage — else PATH lookup."""
    from hermes_constants import (hermes_managed_node_tree_present, iter_hermes_node_dirs)
    managed_node_present = hermes_managed_node_tree_present(hermes_root)
    for directory in iter_hermes_node_dirs(hermes_root) if managed_node_present else ():
        entry = str(directory)
        try:
            present = directory.is_dir()
        except OSError:
            present = False
        if present and entry not in path_entries:
            path_entries.append(entry)

    # With managed Node present, consulting the invoker's PATH would make a system unit depend on who ran sudo.
    if managed_node_present:
        return

    resolved_node = shutil.which("node")
    if not resolved_node:
        return

    # Use the dir where node is FOUND, not the symlink target (~/.local/bin/node often links into one profile).
    resolved_node_dir = str(Path(resolved_node).parent)
    if resolved_node_dir not in path_entries:
        path_entries.append(resolved_node_dir)


def _service_venv_dir() -> str:
    """VIRTUAL_ENV baked into service definitions: detected venv, else ``_gw().PROJECT_ROOT/venv``."""
    detected_venv = _gw()._detect_venv_dir()
    return str(detected_venv) if detected_venv else str(_gw().PROJECT_ROOT / "venv")


def generate_systemd_unit(system: bool = False, run_as_user: str | None = None) -> str:
    python_path = _gw().get_python_path()
    working_dir = _gw()._stable_service_working_dir()
    venv_dir = _gw()._service_venv_dir()

    path_entries = _gw()._build_service_path_dirs()
    if not system:
        # System units add managed Node once the TARGET user's home is known (not the sudo caller's).
        _gw()._append_node_dir_for_service(path_entries)

    # TimeoutStopSec must cover the full stop budget (cron drain + cleanup) or systemd SIGKILLs mid-drain.
    restart_timeout = _gw().resolve_systemd_timeout_stop_sec(_gw()._get_restart_drain_timeout(), _gw()._get_cron_drain_timeout())

    if system:
        username, group_name, home_dir, uid = _gw()._system_service_identity(run_as_user)
        hermes_home = _gw()._hermes_home_for_target_user(home_dir)
        # Profile arg relative to the TARGET user's ~/.hermes when hermes_home lives under it.
        target_root = Path(home_dir) / ".hermes"
        try:
            Path(hermes_home).resolve().relative_to(target_root.resolve())
            profile_arg = _gw()._profile_arg(hermes_home, default_root=target_root)
        except ValueError:
            profile_arg = _gw()._profile_arg(hermes_home)
        # Remap paths under the calling user's home (/root/) to the target user's so the service can read them.
        python_path = _gw()._remap_path_for_user(python_path, home_dir)
        working_dir = str(hermes_home) if hermes_home else _gw()._remap_path_for_user(working_dir, home_dir)
        venv_dir = _gw()._remap_path_for_user(venv_dir, home_dir)
        path_entries = [_gw()._remap_path_for_user(p, home_dir) for p in path_entries]
        # Managed Node for the TARGET user's tree, prepended so it outranks remapped shell-PATH entries.
        _target_node_entries: list[str] = []
        _gw()._append_node_dir_for_service(_target_node_entries, Path(hermes_home) if hermes_home else None)
        path_entries = [e for e in _target_node_entries if e not in path_entries] + path_entries
        user_home = Path(home_dir)
        identity_lines = f"User={username}\nGroup={group_name}\n"
        # Restart-safe cron/Kanban workers cross `systemd-run --user`, which needs this user's manager;
        # without the ordering the gateway and user@<uid>.service race at boot and the one-shot bus
        # adoption in run_gateway() can miss (#104893).
        ordering_lines = f"After=user@{uid}.service\nWants=user@{uid}.service\n"
        env_lines = (
            f'Environment="HOME={home_dir}"\n'
            f'Environment="USER={username}"\n'
            f'Environment="LOGNAME={username}"\n'
        ) + _gw()._ld_library_path_line(system=True, target_home_dir=home_dir)
        wanted_by = "multi-user.target"
    else:
        hermes_home = str(_gw().get_hermes_home().resolve())
        profile_arg = _gw()._profile_arg(hermes_home)
        user_home = Path.home()
        identity_lines = ordering_lines = ""
        env_lines = _gw()._ld_library_path_line(system=False)
        wanted_by = "default.target"

    watchdog_seconds = _gw()._systemd_watchdog_seconds(hermes_home)
    systemd_type, systemd_watchdog_directives = "simple", ""
    if watchdog_seconds > 0:
        systemd_type, systemd_watchdog_directives = "notify", f"NotifyAccess=main\nWatchdogSec={watchdog_seconds}s\n"
    path_entries.extend(_gw()._build_user_local_paths(user_home, path_entries))
    path_entries.extend(_gw()._build_wsl_interop_paths(path_entries))
    path_entries.extend(["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"])
    sane_path = ":".join(path_entries)
    return f"""[Unit]
Description={_gw().SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
{ordering_lines}StartLimitIntervalSec=0

[Service]
Type={systemd_type}
{systemd_watchdog_directives}{identity_lines}ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run
WorkingDirectory={working_dir}
{env_lines}Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Environment="HERMES_SUPERVISED_CHILD=1"
Restart=always
RestartSec=5
RestartForceExitStatus={_gw().GATEWAY_SERVICE_RESTART_EXIT_CODE}
SuccessExitStatus={_gw().GATEWAY_SERVICE_RESTART_EXIT_CODE}
RestartPreventExitStatus={_gw().GATEWAY_FATAL_CONFIG_EXIT_CODE}
KillMode=mixed
KillSignal=SIGTERM
ExecReload=/bin/kill -USR1 $MAINPID
ExecStop=-{python_path} -m hermes_systemd_planned_stop $MAINPID
ExecStopPost=-{python_path} -m gateway.cgroup_cleanup
TimeoutStopSec={restart_timeout}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy={wanted_by}
"""


def _normalize_service_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


# Directives older systemd silently strips; ignored in stale-checks so such units aren't flagged forever.
_SYSTEMD_OPTIONAL_DIRECTIVES = ("RestartMaxDelaySec", "RestartSteps")


def _strip_optional_systemd_directives(text: str) -> str:
    """Remove systemd directives that older hosts silently drop."""
    filtered = []
    for line in text.splitlines():
        stripped = line.strip()
        is_directive = stripped and not stripped.startswith("#")
        if not (is_directive and stripped.split("=", 1)[0].strip() in _SYSTEMD_OPTIONAL_DIRECTIVES):
            filtered.append(line)
    return "\n".join(filtered)


def _normalize_launchd_plist_for_comparison(text: str) -> str:
    """Normalize plist text for staleness checks, ignoring the PATH payload: the generated PATH is
    captured from the invoking shell and varies across shells."""
    import re
    return re.sub(
        r"(<key>PATH</key>\s*<string>)(.*?)(</string>)", r"\1__HERMES_PATH__\3",
        _gw()._normalize_service_definition(text), flags=re.S,
    )


def systemd_unit_is_current(system: bool = False) -> bool:
    # HERMES_HOME sync chokepoint for every compare/regenerate path: under `sudo … --system` it is often
    # stripped to /root/.hermes, so refresh would rewrite a correct unit and status warn forever.
    # Idempotent; the os.environ mutation persists for later runtime reads (restart's PID/drain).
    _gw()._sync_hermes_home_from_systemd_unit(system=system)

    unit_path = _gw().get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False

    installed = unit_path.read_text(encoding="utf-8")
    expected_user = _gw()._read_systemd_user_from_unit(unit_path) if system else None
    expected = _gw().generate_systemd_unit(system=system, run_as_user=expected_user)
    # Ignore directives older systemd drops (RestartMaxDelaySec, RestartSteps) to avoid a perpetual "outdated" flag.
    norm = lambda text: _gw()._normalize_service_definition(_gw()._strip_optional_systemd_directives(text))  # noqa: E731
    return norm(installed) == norm(expected)


def _temp_home_in_service_definition(definition: str) -> str | None:
    """Temp-dir HERMES_HOME baked into a systemd unit / launchd plist, or None. A temp home means a
    test/E2E harness generated it; installing it leaves the gateway "running" but deaf to every platform."""
    import re
    import tempfile
    candidates = re.findall(r'HERMES_HOME=([^"\n]+)', definition)
    candidates += re.findall(r"<key>HERMES_HOME</key>\s*<string>(.*?)</string>", definition, flags=re.S)
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


def _refuse_temp_home_service_write(definition: str, kind: str) -> bool:
    """Refuse (with guidance) when a service definition carries a temp HERMES_HOME."""
    temp_home = _gw()._temp_home_in_service_definition(definition)
    if temp_home is None:
        return False
    print(f"✗ Refusing to write the gateway {kind}: HERMES_HOME resolves to a temporary directory ({temp_home}).")
    print(
        "  This usually means a test/E2E environment exported HERMES_HOME. "
        "Unset it (or run from a clean shell) and retry."
    )
    return True


def refresh_systemd_unit_if_needed(system: bool = False) -> bool:
    """Rewrite the installed systemd unit when the generated definition has changed."""
    unit_path = _gw().get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False

    # _gw().systemd_unit_is_current is the HERMES_HOME-sync chokepoint; its env mutation persists for the regenerate below.
    if _gw().systemd_unit_is_current(system=system):
        return False

    expected_user = _gw()._read_systemd_user_from_unit(unit_path) if system else None
    new_unit = _gw().generate_systemd_unit(system=system, run_as_user=expected_user)

    # Test safety belt: the user unit path is under Path.home(), which conftest does NOT sandbox, and a
    # pytest-tmp HERMES_HOME baked into the developer's real unit breaks their gateway on next reboot.
    if not system and any(m in new_unit for m in ("/pytest-of-", '/hermes_test"', "/hermes_test/")):
        return False

    # Structural variant: refuse ANY temp-dir HERMES_HOME (manual E2E homes lack the pytest markers).
    if _gw()._refuse_temp_home_service_write(new_unit, "systemd unit"):
        return False

    unit_path.write_text(new_unit, encoding="utf-8")
    _gw()._run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"↻ Updated gateway {_gw()._service_scope_label(system)} service definition to match the current Hermes install")
    return True
