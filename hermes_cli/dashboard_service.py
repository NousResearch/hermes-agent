"""Durable dashboard service and secure access helpers.

The dashboard is intentionally separate from the messaging gateway, but the
host-service contract should feel the same to operators: profile-scoped
systemd/launchd/Windows service definitions, stable Hermes home anchoring, and
commands that can be invoked from either the CLI or the dashboard admin page.
"""

from __future__ import annotations

import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli.config import get_hermes_home, is_managed, managed_error
from hermes_cli.gateway import (
    PROJECT_ROOT,
    UserSystemdUnavailableError,
    SystemScopeRequiresRootError,
    _build_service_path_dirs,
    _build_user_local_paths,
    _build_wsl_interop_paths,
    _detect_venv_dir,
    _ensure_linger_enabled,
    _hermes_home_for_target_user,
    _launchd_domain,
    _preflight_user_systemd,
    _profile_arg,
    _profile_suffix,
    _read_systemd_user_from_unit,
    _remap_path_for_user,
    _require_root_for_system_service,
    _run_systemctl,
    _service_scope_label,
    _stable_service_working_dir,
    _system_service_identity,
    _sync_hermes_home_from_systemd_unit,
    get_python_path,
    is_container,
    is_macos,
    is_termux,
    is_windows,
    is_wsl,
    supports_systemd_services,
)


SERVICE_BASE = "hermes-dashboard"
SERVICE_DESCRIPTION = "Hermes Agent Dashboard"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9119
DEFAULT_TAILSCALE_HTTPS_PORT = 443


@dataclass(frozen=True)
class DashboardServiceOptions:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    tui: bool = False
    insecure: bool = False
    allowed_hosts: tuple[str, ...] = ()
    public_url: str = ""


def _service_config_path() -> Path:
    return get_hermes_home() / "dashboard-service" / "config.json"


def _options_to_dict(options: DashboardServiceOptions) -> dict[str, Any]:
    return {
        "host": options.host,
        "port": options.port,
        "tui": options.tui,
        "insecure": options.insecure,
        "allowed_hosts": list(options.allowed_hosts),
        "public_url": options.public_url,
    }


def save_service_options(options: DashboardServiceOptions) -> None:
    path = _service_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_options_to_dict(options), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_service_options() -> DashboardServiceOptions | None:
    path = _service_config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return DashboardServiceOptions(
        host=str(raw.get("host") or DEFAULT_HOST),
        port=int(raw.get("port") or DEFAULT_PORT),
        tui=bool(raw.get("tui", False)),
        insecure=bool(raw.get("insecure", False)),
        allowed_hosts=normalize_allowed_hosts(raw.get("allowed_hosts", ())),
        public_url=str(raw.get("public_url") or ""),
    )


def normalize_allowed_hosts(value: Any) -> tuple[str, ...]:
    """Normalize CLI/config/env host allowlist values."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = []
        for item in value:
            if isinstance(item, str):
                raw_items.extend(re.split(r"[,\s]+", item))
            elif item is not None:
                raw_items.append(str(item))
    else:
        raw_items = [str(value)]

    hosts: list[str] = []
    for raw in raw_items:
        item = raw.strip()
        if not item:
            continue
        if "://" in item:
            from urllib.parse import urlparse

            parsed = urlparse(item)
            item = parsed.netloc or parsed.path
        if item.startswith("["):
            close = item.find("]")
            item = item[1:close] if close != -1 else item.strip("[]")
        elif ":" in item:
            item = item.rsplit(":", 1)[0]
        item = item.strip().strip(".").lower()
        if item and item not in hosts:
            hosts.append(item)
    return tuple(hosts)


def options_from_args(args: Any) -> DashboardServiceOptions:
    return DashboardServiceOptions(
        host=getattr(args, "host", DEFAULT_HOST) or DEFAULT_HOST,
        port=int(getattr(args, "port", DEFAULT_PORT) or DEFAULT_PORT),
        tui=bool(getattr(args, "tui", False)),
        insecure=bool(getattr(args, "insecure", False)),
        allowed_hosts=normalize_allowed_hosts(getattr(args, "allowed_hosts", None)),
        public_url=(getattr(args, "public_url", "") or "").strip(),
    )


def get_service_name() -> str:
    suffix = _profile_suffix()
    return f"{SERVICE_BASE}-{suffix}" if suffix else SERVICE_BASE


def get_systemd_unit_path(system: bool = False) -> Path:
    name = get_service_name()
    if system:
        return Path("/etc/systemd/system") / f"{name}.service"
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


def get_launchd_label() -> str:
    suffix = _profile_suffix()
    return f"ai.hermes.dashboard-{suffix}" if suffix else "ai.hermes.dashboard"


def get_launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{get_launchd_label()}.plist"


def _dashboard_cli_args(options: DashboardServiceOptions, hermes_home: str | None = None) -> list[str]:
    profile_arg = _profile_arg(hermes_home)
    args: list[str] = []
    if profile_arg:
        args.extend(shlex.split(profile_arg))
    args.extend(
        [
            "dashboard",
            "--host",
            options.host,
            "--port",
            str(options.port),
            "--no-open",
            "--skip-build",
        ]
    )
    if options.tui:
        args.append("--tui")
    if options.insecure:
        args.append("--insecure")
    if options.allowed_hosts:
        args.extend(["--allowed-hosts", ",".join(options.allowed_hosts)])
    return args


def _service_env_lines(options: DashboardServiceOptions, hermes_home: str) -> list[str]:
    lines = [f'Environment="HERMES_HOME={hermes_home}"']
    if options.public_url:
        lines.append(f'Environment="HERMES_DASHBOARD_PUBLIC_URL={options.public_url}"')
    if options.allowed_hosts:
        lines.append(
            f'Environment="HERMES_DASHBOARD_ALLOWED_HOSTS={",".join(options.allowed_hosts)}"'
        )
    return lines


def generate_systemd_unit(
    options: DashboardServiceOptions | None = None,
    *,
    system: bool = False,
    run_as_user: str | None = None,
) -> str:
    options = options or load_service_options() or DashboardServiceOptions()
    python_path = get_python_path()
    working_dir = _stable_service_working_dir()
    detected_venv = _detect_venv_dir()
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")
    path_entries = _build_service_path_dirs()
    resolved_node = shutil.which("node")
    if resolved_node:
        node_dir = str(Path(resolved_node).resolve().parent)
        if node_dir not in path_entries:
            path_entries.append(node_dir)
    common_bin_paths = [
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]

    if system:
        username, group_name, home_dir = _system_service_identity(run_as_user)
        hermes_home = _hermes_home_for_target_user(home_dir)
        python_path = _remap_path_for_user(python_path, home_dir)
        working_dir = str(hermes_home) if hermes_home else _remap_path_for_user(working_dir, home_dir)
        venv_dir = _remap_path_for_user(venv_dir, home_dir)
        path_entries = [_remap_path_for_user(p, home_dir) for p in path_entries]
        path_entries.extend(_build_user_local_paths(Path(home_dir), path_entries))
        path_entries.extend(_build_wsl_interop_paths(path_entries))
        path_entries.extend(common_bin_paths)
        sane_path = ":".join(path_entries)
        exec_args = " ".join(_dashboard_cli_args(options, hermes_home))
        env_lines = "\n".join(
            [
                f'Environment="HOME={home_dir}"',
                f'Environment="USER={username}"',
                f'Environment="LOGNAME={username}"',
                f'Environment="PATH={sane_path}"',
                f'Environment="VIRTUAL_ENV={venv_dir}"',
                *_service_env_lines(options, hermes_home),
            ]
        )
        return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User={username}
Group={group_name}
ExecStart={python_path} -m hermes_cli.main {exec_args}
WorkingDirectory={working_dir}
{env_lines}
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=60
NoNewPrivileges=true
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

    hermes_home = str(get_hermes_home().resolve())
    path_entries.extend(_build_user_local_paths(Path.home(), path_entries))
    path_entries.extend(_build_wsl_interop_paths(path_entries))
    path_entries.extend(common_bin_paths)
    sane_path = ":".join(path_entries)
    exec_args = " ".join(_dashboard_cli_args(options, hermes_home))
    env_lines = "\n".join(
        [
            f'Environment="PATH={sane_path}"',
            f'Environment="VIRTUAL_ENV={venv_dir}"',
            *_service_env_lines(options, hermes_home),
        ]
    )
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main {exec_args}
WorkingDirectory={working_dir}
{env_lines}
Restart=always
RestartSec=5
RestartMaxDelaySec=300
RestartSteps=5
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=60
NoNewPrivileges=true
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def _normalize_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _normalize_launchd_plist(text: str) -> str:
    return re.sub(
        r"(<key>PATH</key>\s*<string>)(.*?)(</string>)",
        r"\1__HERMES_PATH__\3",
        _normalize_definition(text),
        flags=re.S,
    )


def systemd_unit_is_current(
    options: DashboardServiceOptions | None = None, *, system: bool = False
) -> bool:
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False
    installed = unit_path.read_text(encoding="utf-8")
    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    expected = generate_systemd_unit(options, system=system, run_as_user=expected_user)
    return _normalize_definition(installed) == _normalize_definition(expected)


def _select_systemd_scope(system: bool = False) -> bool:
    if system:
        return True
    return (
        get_systemd_unit_path(system=True).exists()
        and not get_systemd_unit_path(system=False).exists()
    )


def refresh_systemd_unit_if_needed(
    options: DashboardServiceOptions | None = None, *, system: bool = False
) -> bool:
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists() or systemd_unit_is_current(options, system=system):
        return False
    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    new_unit = generate_systemd_unit(options, system=system, run_as_user=expected_user)
    if not system and (
        "/pytest-of-" in new_unit
        or '/hermes_test"' in new_unit
        or "/hermes_test/" in new_unit
    ):
        return False
    unit_path.write_text(new_unit, encoding="utf-8")
    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print("Updated dashboard service definition to match the current Hermes install")
    return True


def systemd_install(
    options: DashboardServiceOptions | None = None,
    *,
    force: bool = False,
    system: bool = False,
    run_as_user: str | None = None,
    enable_on_startup: bool = True,
) -> None:
    options = options or DashboardServiceOptions()
    save_service_options(options)
    if system:
        _require_root_for_system_service("install")
    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists() and not force:
        if not systemd_unit_is_current(options, system=system):
            print(f"Repairing outdated dashboard {_service_scope_label(system)} service at: {unit_path}")
            refresh_systemd_unit_if_needed(options, system=system)
            if enable_on_startup:
                _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print("Dashboard service definition updated")
            return
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(
        generate_systemd_unit(options, system=system, run_as_user=run_as_user),
        encoding="utf-8",
    )
    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    if enable_on_startup:
        _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
    print(f"Dashboard {_service_scope_label(system)} service installed at: {unit_path}")
    if not system:
        _ensure_linger_enabled()


def _require_systemd_service_installed(action: str, *, system: bool = False) -> None:
    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        return
    scope_flag = " --system" if system else ""
    print("Dashboard service is not installed")
    print(f"Run: {'sudo ' if system else ''}hermes dashboard service install{scope_flag}")
    sys.exit(1)


def systemd_uninstall(*, system: bool = False) -> None:
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("uninstall")
    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)
    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"Removed {unit_path}")
    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print("Dashboard service uninstalled")


def systemd_start(
    options: DashboardServiceOptions | None = None, *, system: bool = False
) -> None:
    options = options or load_service_options()
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("start")
    else:
        _preflight_user_systemd()
    _require_systemd_service_installed("start", system=system)
    refresh_systemd_unit_if_needed(options, system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print("Dashboard service started")


def systemd_stop(*, system: bool = False) -> None:
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("stop")
    _require_systemd_service_installed("stop", system=system)
    _sync_hermes_home_from_systemd_unit(system=system)
    _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    print("Dashboard service stopped")


def systemd_restart(
    options: DashboardServiceOptions | None = None, *, system: bool = False
) -> None:
    options = options or load_service_options()
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("restart")
    else:
        _preflight_user_systemd()
    _require_systemd_service_installed("restart", system=system)
    refresh_systemd_unit_if_needed(options, system=system)
    _sync_hermes_home_from_systemd_unit(system=system)
    _run_systemctl(["restart", get_service_name()], system=system, check=True, timeout=90)
    print("Dashboard service restarted")


def _systemd_is_active(system: bool = False) -> bool:
    if not get_systemd_unit_path(system=system).exists():
        return False
    result = _run_systemctl(
        ["is-active", get_service_name()],
        system=system,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() == "active"


def systemd_status(*, deep: bool = False, system: bool = False, full: bool = False) -> None:
    system = _select_systemd_scope(system)
    unit_path = get_systemd_unit_path(system=system)
    print(f"Systemd unit: {unit_path}")
    if not unit_path.exists():
        print("Dashboard service is not installed")
        return
    if not systemd_unit_is_current(system=system):
        print("Installed dashboard service definition is outdated")
        print("Run: hermes dashboard service restart")
    status_cmd = ["status", get_service_name(), "--no-pager"]
    if full:
        status_cmd.append("-l")
    _run_systemctl(status_cmd, system=system, capture_output=False, timeout=10)
    print("Dashboard service is running" if _systemd_is_active(system) else "Dashboard service is stopped")
    if deep:
        subprocess.run(
            (["journalctl"] if system else ["journalctl", "--user"])
            + ["-u", get_service_name(), "-n", "20", "--no-pager"],
            timeout=10,
        )


def generate_launchd_plist(options: DashboardServiceOptions | None = None) -> str:
    options = options or load_service_options() or DashboardServiceOptions()
    python_path = get_python_path()
    working_dir = _stable_service_working_dir()
    hermes_home = str(get_hermes_home().resolve())
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = get_launchd_label()
    detected_venv = _detect_venv_dir()
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")
    priority_dirs = _build_service_path_dirs()
    resolved_node = shutil.which("node")
    if resolved_node:
        node_dir = str(Path(resolved_node).resolve().parent)
        if node_dir not in priority_dirs:
            priority_dirs.append(node_dir)
    sane_path = ":".join(
        dict.fromkeys(priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p])
    )
    prog_args = [python_path, "-m", "hermes_cli.main", *_dashboard_cli_args(options, hermes_home)]
    prog_args_xml = "\n        ".join(f"<string>{html.escape(arg)}</string>" for arg in prog_args)
    public_url = (
        f"\n        <key>HERMES_DASHBOARD_PUBLIC_URL</key>\n        <string>{html.escape(options.public_url)}</string>"
        if options.public_url
        else ""
    )
    allowed_hosts = (
        "\n        <key>HERMES_DASHBOARD_ALLOWED_HOSTS</key>\n"
        f"        <string>{html.escape(','.join(options.allowed_hosts))}</string>"
        if options.allowed_hosts
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{html.escape(label)}</string>

    <key>ProgramArguments</key>
    <array>
        {prog_args_xml}
    </array>

    <key>WorkingDirectory</key>
    <string>{html.escape(working_dir)}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{html.escape(sane_path)}</string>
        <key>VIRTUAL_ENV</key>
        <string>{html.escape(venv_dir)}</string>
        <key>HERMES_HOME</key>
        <string>{html.escape(hermes_home)}</string>{public_url}{allowed_hosts}
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>4096</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{html.escape(str(log_dir / "dashboard.log"))}</string>

    <key>StandardErrorPath</key>
    <string>{html.escape(str(log_dir / "dashboard.error.log"))}</string>
</dict>
</plist>
"""


def launchd_plist_is_current(options: DashboardServiceOptions | None = None) -> bool:
    path = get_launchd_plist_path()
    if not path.exists():
        return False
    return _normalize_launchd_plist(path.read_text(encoding="utf-8")) == _normalize_launchd_plist(
        generate_launchd_plist(options)
    )


def refresh_launchd_plist_if_needed(options: DashboardServiceOptions | None = None) -> bool:
    path = get_launchd_plist_path()
    if not path.exists() or launchd_plist_is_current(options):
        return False
    path.write_text(generate_launchd_plist(options), encoding="utf-8")
    label = get_launchd_label()
    subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=False, timeout=90)
    subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(path)], check=False, timeout=30)
    print("Updated dashboard launchd service definition")
    return True


def launchd_install(options: DashboardServiceOptions | None = None, *, force: bool = False) -> None:
    if options is not None:
        save_service_options(options)
    path = get_launchd_plist_path()
    if path.exists() and not force:
        if not launchd_plist_is_current(options):
            refresh_launchd_plist_if_needed(options)
            print("Dashboard service definition updated")
            return
        print(f"Service already installed at: {path}")
        print("Use --force to reinstall")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_launchd_plist(options), encoding="utf-8")
    subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(path)], check=True, timeout=30)
    print(f"Dashboard launchd service installed at: {path}")


def launchd_uninstall() -> None:
    path = get_launchd_plist_path()
    label = get_launchd_label()
    subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=False, timeout=90)
    if path.exists():
        path.unlink()
        print(f"Removed {path}")
    print("Dashboard service uninstalled")


def launchd_start(options: DashboardServiceOptions | None = None) -> None:
    options = options or load_service_options()
    path = get_launchd_plist_path()
    label = get_launchd_label()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generate_launchd_plist(options), encoding="utf-8")
        subprocess.run(["launchctl", "bootstrap", _launchd_domain(), str(path)], check=True, timeout=30)
    else:
        refresh_launchd_plist_if_needed(options)
    subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)
    print("Dashboard service started")


def launchd_stop() -> None:
    label = get_launchd_label()
    try:
        subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if exc.returncode not in {3, 113}:
            raise
    print("Dashboard service stopped")


def launchd_restart(options: DashboardServiceOptions | None = None) -> None:
    options = options or load_service_options()
    label = get_launchd_label()
    refresh_launchd_plist_if_needed(options)
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"{_launchd_domain()}/{label}"], check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if exc.returncode not in {3, 113}:
            raise
        launchd_start(options)
        return
    print("Dashboard service restarted")


def launchd_status(*, deep: bool = False) -> None:
    path = get_launchd_plist_path()
    label = get_launchd_label()
    print(f"Launchd plist: {path}")
    result = subprocess.run(["launchctl", "list", label], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("Dashboard service is loaded")
        print(result.stdout)
    else:
        print("Dashboard service is not loaded")
    if path.exists() and not launchd_plist_is_current():
        print("Installed dashboard service definition is stale")
    if deep:
        log_file = get_hermes_home() / "logs" / "dashboard.log"
        if log_file.exists():
            subprocess.run(["tail", "-20", str(log_file)], timeout=10)


def get_windows_task_name() -> str:
    suffix = _profile_suffix()
    return f"Hermes_Dashboard_{suffix}" if suffix else "Hermes_Dashboard"


def _windows_script_path() -> Path:
    return get_hermes_home() / "dashboard-service" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', get_windows_task_name())}.cmd"


def _quote_cmd(value: str) -> str:
    if "\r" in value or "\n" in value:
        raise ValueError("refusing to quote value containing newline")
    if not value:
        return '""'
    if not re.search(r'[ \t"]', value):
        return value
    return '"' + value.replace('"', '""') + '"'


def generate_windows_cmd_script(options: DashboardServiceOptions | None = None) -> str:
    options = options or load_service_options() or DashboardServiceOptions()
    hermes_home = str(get_hermes_home().resolve())
    python_path = get_python_path()
    args = [python_path, "-m", "hermes_cli.main", *_dashboard_cli_args(options, hermes_home)]
    lines = [
        "@echo off",
        "rem Hermes Agent Dashboard",
        f"cd /d {_quote_cmd(str(PROJECT_ROOT))}",
        f'set "HERMES_HOME={hermes_home}"',
        "set \"PYTHONIOENCODING=utf-8\"",
    ]
    if options.public_url:
        lines.append(f'set "HERMES_DASHBOARD_PUBLIC_URL={options.public_url}"')
    if options.allowed_hosts:
        lines.append(f'set "HERMES_DASHBOARD_ALLOWED_HOSTS={",".join(options.allowed_hosts)}"')
    lines.append(" ".join(_quote_cmd(a) for a in args))
    lines.append("exit /b 0")
    return "\r\n".join(lines) + "\r\n"


def _write_windows_script(options: DashboardServiceOptions | None = None) -> Path:
    path = _windows_script_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(generate_windows_cmd_script(options), encoding="utf-8", newline="")
    tmp.replace(path)
    return path


def _exec_schtasks(args: list[str]) -> tuple[int, str, str]:
    exe = shutil.which("schtasks")
    if exe is None:
        return 1, "", "schtasks.exe not found"
    proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=20)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def windows_is_installed() -> bool:
    code, _out, _err = _exec_schtasks(["/Query", "/TN", get_windows_task_name()])
    return code == 0


def windows_install(options: DashboardServiceOptions | None = None, *, force: bool = False) -> None:
    if options is not None:
        save_service_options(options)
    script = _write_windows_script(options)
    if force:
        _exec_schtasks(["/Delete", "/F", "/TN", get_windows_task_name()])
    code, out, err = _exec_schtasks(
        [
            "/Create",
            "/F",
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
            "/TN",
            get_windows_task_name(),
            "/TR",
            str(script),
        ]
    )
    if code != 0:
        raise RuntimeError((err or out or "schtasks failed").strip())
    print(f"Dashboard Scheduled Task installed: {get_windows_task_name()}")


def windows_uninstall() -> None:
    _exec_schtasks(["/End", "/TN", get_windows_task_name()])
    code, out, err = _exec_schtasks(["/Delete", "/F", "/TN", get_windows_task_name()])
    if code != 0 and "cannot find" not in (out + err).lower():
        raise RuntimeError((err or out or "schtasks delete failed").strip())
    script = _windows_script_path()
    if script.exists():
        script.unlink()
    print("Dashboard Scheduled Task uninstalled")


def windows_start(options: DashboardServiceOptions | None = None) -> None:
    options = options or load_service_options()
    if not windows_is_installed():
        _write_windows_script(options)
    code, out, err = _exec_schtasks(["/Run", "/TN", get_windows_task_name()])
    if code != 0:
        raise RuntimeError((err or out or "schtasks run failed").strip())
    print("Dashboard Scheduled Task started")


def windows_stop() -> None:
    code, out, err = _exec_schtasks(["/End", "/TN", get_windows_task_name()])
    if code != 0:
        raise RuntimeError((err or out or "schtasks end failed").strip())
    print("Dashboard Scheduled Task stopped")


def windows_status() -> None:
    code, out, err = _exec_schtasks(["/Query", "/TN", get_windows_task_name(), "/V", "/FO", "LIST"])
    if code == 0:
        print(out)
    else:
        print("Dashboard Scheduled Task is not installed")
        if err:
            print(err.strip())


def _installed_manager() -> str:
    if supports_systemd_services() and (
        get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()
    ):
        return "systemd"
    if is_macos() and get_launchd_plist_path().exists():
        return "launchd"
    if is_windows() and windows_is_installed():
        return "windows"
    if supports_systemd_services():
        return "systemd"
    if is_macos():
        return "launchd"
    if is_windows():
        return "windows"
    return "none"


def get_dashboard_service_snapshot(system: bool = False) -> dict[str, Any]:
    manager = _installed_manager()
    installed = False
    running = False
    path = ""
    name = get_service_name()
    scope = None
    if manager == "systemd":
        selected_system = _select_systemd_scope(system)
        path = str(get_systemd_unit_path(system=selected_system))
        installed = Path(path).exists()
        scope = "system" if selected_system else "user"
        if installed:
            try:
                running = _systemd_is_active(selected_system)
            except Exception:
                running = False
    elif manager == "launchd":
        path = str(get_launchd_plist_path())
        installed = Path(path).exists()
        name = get_launchd_label()
        try:
            running = subprocess.run(
                ["launchctl", "list", name],
                capture_output=True,
                text=True,
                timeout=5,
            ).returncode == 0
        except Exception:
            running = False
    elif manager == "windows":
        name = get_windows_task_name()
        path = str(_windows_script_path())
        installed = windows_is_installed()
    return {
        "manager": manager,
        "installed": installed,
        "running": running,
        "name": name,
        "path": path,
        "scope": scope,
    }


def _unsupported_platform() -> None:
    if is_termux():
        print("Dashboard service installation is not supported on Termux.")
        print("Run manually: hermes dashboard")
    elif is_wsl():
        print("WSL detected but systemd is not running.")
        print("Run manually in tmux/screen: hermes dashboard --no-open")
    elif is_container():
        print("Container dashboard supervision is handled by the container runtime/s6.")
    else:
        print("Dashboard service management is not supported on this platform.")
    sys.exit(1)


def _service_command(args: Any) -> None:
    action = getattr(args, "dashboard_service_command", None) or "status"
    options = options_from_args(args) if action in {"install", "unit"} else None
    system = bool(getattr(args, "system", False))

    if action == "unit":
        if supports_systemd_services() or getattr(args, "systemd", False):
            print(generate_systemd_unit(options, system=system, run_as_user=getattr(args, "run_as_user", None)))
        elif is_macos() or getattr(args, "launchd", False):
            print(generate_launchd_plist(options))
        elif is_windows():
            print(generate_windows_cmd_script(options))
        else:
            print(generate_systemd_unit(options, system=system, run_as_user=getattr(args, "run_as_user", None)))
        return

    if action == "install":
        if is_managed():
            managed_error("install dashboard service (managed by NixOS)")
            return
        force = bool(getattr(args, "force", False))
        start_now = bool(getattr(args, "start_now", False))
        enable_on_startup = bool(getattr(args, "start_on_login", True))
        if supports_systemd_services():
            systemd_install(
                options,
                force=force,
                system=system,
                run_as_user=getattr(args, "run_as_user", None),
                enable_on_startup=enable_on_startup,
            )
            if start_now:
                systemd_start(options, system=system)
        elif is_macos():
            launchd_install(options, force=force)
            if start_now:
                launchd_start(options)
        elif is_windows():
            windows_install(options, force=force)
            if start_now:
                windows_start(options)
        else:
            _unsupported_platform()
        return

    if action == "uninstall":
        if supports_systemd_services():
            systemd_uninstall(system=system)
        elif is_macos():
            launchd_uninstall()
        elif is_windows():
            windows_uninstall()
        else:
            _unsupported_platform()
        return

    if action == "start":
        if supports_systemd_services():
            systemd_start(options, system=system)
        elif is_macos():
            launchd_start(options)
        elif is_windows():
            windows_start(options)
        else:
            _unsupported_platform()
        return

    if action == "stop":
        if supports_systemd_services():
            systemd_stop(system=system)
        elif is_macos():
            launchd_stop()
        elif is_windows():
            windows_stop()
        else:
            _unsupported_platform()
        return

    if action == "restart":
        if supports_systemd_services():
            systemd_restart(options, system=system)
        elif is_macos():
            launchd_restart(options)
        elif is_windows():
            windows_stop()
            windows_start(options)
        else:
            _unsupported_platform()
        return

    if action == "status":
        deep = bool(getattr(args, "deep", False))
        full = bool(getattr(args, "full", False))
        if supports_systemd_services() and (
            get_systemd_unit_path(system=False).exists() or get_systemd_unit_path(system=True).exists()
        ):
            systemd_status(deep=deep, system=system, full=full)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep=deep)
        elif is_windows() and windows_is_installed():
            windows_status()
        else:
            snap = get_dashboard_service_snapshot(system=system)
            print("Dashboard service is not installed")
            print(f"Detected manager: {snap['manager']}")
            print("Run: hermes dashboard service install")
        return

    print(f"Unknown dashboard service command: {action}")
    sys.exit(2)


def build_tailscale_serve_command(
    *,
    target: str,
    https_port: int | None = DEFAULT_TAILSCALE_HTTPS_PORT,
    http_port: int | None = None,
    set_path: str = "",
    background: bool = True,
    yes: bool = True,
) -> list[str]:
    cmd = ["tailscale", "serve"]
    if background:
        cmd.append("--bg")
    if yes:
        cmd.append("--yes")
    if set_path:
        cmd.extend(["--set-path", set_path])
    if http_port is not None:
        cmd.append(f"--http={http_port}")
    elif https_port is not None:
        cmd.append(f"--https={https_port}")
    cmd.append(target)
    return cmd


def build_cloudflare_config(
    *,
    tunnel: str,
    credentials_file: str,
    hostname: str,
    service: str,
) -> str:
    return "\n".join(
        [
            f"tunnel: {tunnel}",
            f"credentials-file: {credentials_file}",
            "",
            "ingress:",
            f"  - hostname: {hostname}",
            f"    service: {service}",
            "  - service: http_status:404",
            "",
        ]
    )


def _cloudflared_service_command(action: str) -> list[str] | None:
    if action == "install":
        return ["cloudflared", "service", "install"]
    if action == "uninstall":
        return ["cloudflared", "service", "uninstall"]
    if sys.platform.startswith("linux"):
        return ["systemctl", action, "cloudflared"]
    if is_macos():
        if action == "restart":
            return ["sh", "-c", "launchctl stop com.cloudflare.cloudflared; launchctl start com.cloudflare.cloudflared"]
        return ["launchctl", action, "com.cloudflare.cloudflared"]
    if is_windows():
        mapped = {"start": "start", "stop": "stop", "restart": "restart", "status": "query"}
        return ["sc", mapped.get(action, action), "cloudflared"]
    return None


def _access_command(args: Any) -> None:
    action = getattr(args, "dashboard_access_command", None)
    if action == "tailscale-serve":
        target = getattr(args, "target", None) or f"127.0.0.1:{getattr(args, 'port', DEFAULT_PORT)}"
        http_port = getattr(args, "http", None)
        https_port = None if http_port is not None else int(getattr(args, "https", DEFAULT_TAILSCALE_HTTPS_PORT))
        cmd = build_tailscale_serve_command(
            target=target,
            https_port=https_port,
            http_port=http_port,
            set_path=getattr(args, "set_path", "") or "",
            background=not bool(getattr(args, "foreground", False)),
            yes=not bool(getattr(args, "interactive", False)),
        )
        print(" ".join(shlex.quote(part) for part in cmd))
        if getattr(args, "apply", False):
            subprocess.run(cmd, check=True, timeout=60)
        return

    if action == "cloudflare-config":
        service = getattr(args, "service", None) or f"http://127.0.0.1:{getattr(args, 'port', DEFAULT_PORT)}"
        config = build_cloudflare_config(
            tunnel=getattr(args, "tunnel", ""),
            credentials_file=getattr(args, "credentials_file", ""),
            hostname=getattr(args, "hostname", ""),
            service=service,
        )
        output = getattr(args, "output", None)
        if output:
            path = Path(output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config, encoding="utf-8")
            print(f"Wrote {path}")
        else:
            print(config, end="")
        return

    if action == "cloudflare-service":
        verb = getattr(args, "cloudflare_service_command", None) or "status"
        cmd = _cloudflared_service_command(verb)
        if cmd is None:
            print("cloudflared service management is not supported on this platform")
            sys.exit(1)
        if verb == "status":
            print(" ".join(shlex.quote(part) for part in cmd))
            subprocess.run(cmd, check=False, timeout=30)
        else:
            subprocess.run(cmd, check=True, timeout=90)
        return

    print("Run: hermes dashboard access tailscale-serve|cloudflare-config|cloudflare-service")
    sys.exit(2)


def dashboard_command(args: Any) -> bool:
    """Handle dashboard service/access subcommands.

    Returns True when a subcommand was handled. The regular dashboard server
    startup path should run when this returns False.
    """
    if getattr(args, "dashboard_command", None) == "service":
        try:
            _service_command(args)
        except UserSystemdUnavailableError as exc:
            print("User systemd not reachable:")
            for line in str(exc).splitlines():
                print(f"  {line}")
            sys.exit(1)
        except SystemScopeRequiresRootError as exc:
            print(str(exc))
            sys.exit(1)
        return True
    if getattr(args, "dashboard_command", None) == "access":
        _access_command(args)
        return True
    return False
