"""Render gateway systemd service definitions."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from gateway.config import coerce_systemd_watchdog_seconds, load_gateway_config
from gateway.restart import (
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    get_cron_drain_timeout,
    get_restart_drain_timeout,
    resolve_systemd_timeout_stop_sec,
)
from gateway.service_definition import normalize_service_definition
from gateway import service_identity
from gateway import service_process, systemd_identity
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"


def systemd_env_line(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def installed_unit_ld_library_path(system: bool) -> str:
    return service_identity.unit_environment_value(systemd_identity.unit_path(system), "LD_LIBRARY_PATH") or ""


def ld_library_path_line(
    system: bool,
    target_home_dir: str | None = None,
) -> str:
    raw = os.environ.get("LD_LIBRARY_PATH", "") or installed_unit_ld_library_path(system)
    components = [p for p in raw.split(":") if p]
    if target_home_dir is not None:
        components = [service_process.remap_path_for_user(p, target_home_dir) for p in components]
    return systemd_env_line("LD_LIBRARY_PATH", ":".join(components)) if components else ""


def systemd_watchdog_seconds(hermes_home: str | Path | None = None) -> int:
    override_token = reset_home_override = None
    if hermes_home is not None:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        override_token = set_hermes_home_override(hermes_home)
        reset_home_override = reset_hermes_home_override
    try:
        config = load_gateway_config()
        return coerce_systemd_watchdog_seconds(
            getattr(config, "systemd_watchdog_seconds", 0)
        )
    except Exception:
        logger.debug(
            "Could not resolve effective systemd watchdog configuration",
            exc_info=True,
        )
        return 0
    finally:
        if override_token is not None and reset_home_override is not None:
            reset_home_override(override_token)


def generate_systemd_unit(
    system: bool = False,
    run_as_user: str | None = None,
) -> str:
    executable = service_process.python_path()
    working_dir = service_process.stable_working_dir()
    venv_dir = service_process.service_venv_dir()
    path_entries = service_process.service_path_dirs()

    if not system:
        service_process.append_node_dir(path_entries)

    restart_timeout = resolve_systemd_timeout_stop_sec(
        get_restart_drain_timeout(),
        get_cron_drain_timeout(),
    )

    if system:
        username, group_name, home_dir, uid = systemd_identity.system_service_identity(run_as_user)
        hermes_home = service_process.hermes_home_for_target_user(home_dir)
        target_root = Path(home_dir) / ".hermes"
        try:
            Path(hermes_home).resolve().relative_to(target_root.resolve())
            profile = service_identity.profile_arg(hermes_home, default_root=target_root)
        except ValueError:
            profile = service_identity.profile_arg(hermes_home)

        executable = service_process.remap_path_for_user(executable, home_dir)
        working_dir = (
            str(hermes_home)
            if hermes_home
            else service_process.remap_path_for_user(working_dir, home_dir)
        )
        venv_dir = service_process.remap_path_for_user(venv_dir, home_dir)
        path_entries = [service_process.remap_path_for_user(p, home_dir) for p in path_entries]
        target_node_entries: list[str] = []
        service_process.append_node_dir(
            target_node_entries,
            Path(hermes_home) if hermes_home else None,
        )
        path_entries = [
            e for e in target_node_entries if e not in path_entries
        ] + path_entries
        user_home = Path(home_dir)
        identity_lines = f"User={username}\nGroup={group_name}\n"
        ordering_lines = f"After=user@{uid}.service\nWants=user@{uid}.service\n"
        env_lines = (
            f'Environment="HOME={home_dir}"\n'
            f'Environment="USER={username}"\n'
            f'Environment="LOGNAME={username}"\n'
        ) + ld_library_path_line(True, target_home_dir=home_dir)
        wanted_by = "multi-user.target"
    else:
        hermes_home = str(get_hermes_home().resolve())
        profile = service_identity.profile_arg(hermes_home)
        user_home = Path.home()
        identity_lines = ordering_lines = ""
        env_lines = ld_library_path_line(False)
        wanted_by = "default.target"

    watchdog = systemd_watchdog_seconds(hermes_home)
    systemd_type = "notify" if watchdog > 0 else "simple"
    watchdog_directives = (
        f"NotifyAccess=main\nWatchdogSec={watchdog}s\n" if watchdog > 0 else ""
    )
    path_entries.extend(service_process.build_user_local_paths(user_home, path_entries))
    path_entries.extend(service_process.build_wsl_interop_paths(path_entries))
    path_entries.extend([
        "/usr/local/sbin", "/usr/local/bin", "/usr/sbin",
        "/usr/bin", "/sbin", "/bin",
    ])
    sane_path = ":".join(path_entries)
    profile_fragment = f" {profile}" if profile else ""

    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
{ordering_lines}StartLimitIntervalSec=0

[Service]
Type={systemd_type}
{watchdog_directives}{identity_lines}ExecStart={executable} -m hermes_cli.main{profile_fragment} gateway run
WorkingDirectory={working_dir}
{env_lines}Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Environment="HERMES_SUPERVISED_CHILD=1"
Restart=always
RestartSec=5
RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}
SuccessExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}
RestartPreventExitStatus={GATEWAY_FATAL_CONFIG_EXIT_CODE}
KillMode=mixed
KillSignal=SIGTERM
ExecReload=/bin/kill -USR1 $MAINPID
ExecStop=-{executable} -m gateway.systemd_stop_mark
ExecStopPost=-{executable} -m gateway.cgroup_cleanup
TimeoutStopSec={restart_timeout}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy={wanted_by}
"""


SYSTEMD_OPTIONAL_DIRECTIVES = ("RestartMaxDelaySec", "RestartSteps")


def strip_optional_systemd_directives(text: str) -> str:
    filtered = []
    for line in text.splitlines():
        stripped = line.strip()
        is_directive = stripped and not stripped.startswith("#")
        if not (
            is_directive
            and stripped.split("=", 1)[0].strip() in SYSTEMD_OPTIONAL_DIRECTIVES
        ):
            filtered.append(line)
    return "\n".join(filtered)
