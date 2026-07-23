"""Memory provider service lifecycle helpers.

Currently supports Hindsight ``local_external`` on macOS via a user
LaunchAgent. The helper deliberately lives in the CLI layer: external daemon
lifecycle is operational plumbing, not model-tool surface.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home


class UnsupportedMemoryService(RuntimeError):
    """Raised when a provider/mode/platform cannot be service-managed."""


@dataclass(frozen=True)
class HindsightServicePlan:
    provider: str
    mode: str
    api_url: str
    host: str
    port: int
    hermes_home: Path
    config_path: Path
    executable: Path
    env_file: Path
    wrapper_path: Path
    plist_path: Path
    stdout_path: Path
    stderr_path: Path
    label: str


def _profile_name_for_home(hermes_home: Path) -> str:
    """Return a stable Hindsight profile name for a Hermes home."""
    parts = hermes_home.resolve().parts
    if len(parts) >= 2 and parts[-2] == "profiles":
        return parts[-1]
    return "hermes"


def _launchd_label_for_home(hermes_home: Path) -> str:
    profile_name = _profile_name_for_home(hermes_home)
    if profile_name == "hermes":
        return "ai.hermes.hindsight"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in profile_name.lower())
    return f"ai.hermes.hindsight.{safe or 'profile'}"


def _load_hindsight_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise UnsupportedMemoryService(
            f"Hindsight config not found at {config_path}. Run `hermes memory setup hindsight` first."
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise UnsupportedMemoryService(f"Could not read Hindsight config at {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise UnsupportedMemoryService(f"Hindsight config at {config_path} is not a JSON object")
    return data


def _resolve_executable(hermes_home: Path, executable: Path | None = None) -> Path:
    if executable is not None:
        return Path(executable)
    profile_venv_binary = hermes_home / "venvs" / "hindsight" / "bin" / "hindsight-api"
    if profile_venv_binary.exists():
        return profile_venv_binary
    on_path = shutil.which("hindsight-api")
    if on_path:
        return Path(on_path)
    raise UnsupportedMemoryService(
        "Could not find `hindsight-api`. Pass `--executable /path/to/hindsight-api` "
        "or install Hindsight's API package in a dedicated environment."
    )


def build_hindsight_service_plan(
    *,
    hermes_home: Path | None = None,
    executable: Path | None = None,
    env_file: Path | None = None,
) -> HindsightServicePlan:
    """Build a launchd plan for the active profile's Hindsight daemon."""
    hermes_home = Path(hermes_home or get_hermes_home()).expanduser().resolve()
    config_path = hermes_home / "hindsight" / "config.json"
    config = _load_hindsight_config(config_path)
    mode = str(config.get("mode") or "cloud")
    if mode == "local":
        mode = "local_embedded"
    if mode != "local_external":
        raise UnsupportedMemoryService(
            f"`hermes memory service` only manages Hindsight local_external mode; current mode is {mode!r}."
        )

    api_url = str(config.get("api_url") or "http://localhost:8888")
    parsed = urlparse(api_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    profile_name = _profile_name_for_home(hermes_home)
    label = _launchd_label_for_home(hermes_home)
    if env_file is None:
        env_file = Path.home() / ".hindsight" / "profiles" / f"{profile_name}.env"

    return HindsightServicePlan(
        provider="hindsight",
        mode=mode,
        api_url=api_url,
        host=host,
        port=port,
        hermes_home=hermes_home,
        config_path=config_path,
        executable=_resolve_executable(hermes_home, executable),
        env_file=Path(env_file).expanduser(),
        wrapper_path=hermes_home / "services" / "hindsight" / "start-hindsight.sh",
        plist_path=Path.home() / "Library" / "LaunchAgents" / f"{label}.plist",
        stdout_path=hermes_home / "logs" / "hindsight-service.log",
        stderr_path=hermes_home / "logs" / "hindsight-service.err.log",
        label=label,
    )


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_launchd_wrapper(plan: HindsightServicePlan) -> str:
    """Render a launchd wrapper that sources env files without embedding secrets."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME={_shell_single_quote(str(plan.hermes_home))}
CONFIG_PATH="$HERMES_HOME/hindsight/config.json"
HERMES_ENV="$HERMES_HOME/.env"
ENV_FILE={_shell_single_quote(str(plan.env_file))}
HINDSIGHT_API={_shell_single_quote(str(plan.executable))}
LOG_DIR="$HERMES_HOME/logs"

mkdir -p "$LOG_DIR"
printf '[%s] starting Hindsight launchd wrapper\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -x "$HINDSIGHT_API" ]]; then
  printf '[%s] missing executable: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HINDSIGHT_API" >&2
  exit 127
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf '[%s] missing config: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CONFIG_PATH" >&2
  exit 78
fi

export HERMES_HOME
set -a
if [[ -f "$HERMES_ENV" ]]; then
  # shellcheck disable=SC1090
  . "$HERMES_ENV"
fi
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi
set +a

read -r MODE HOST PORT < <(/usr/bin/python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from urllib.parse import urlparse

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
mode = str(cfg.get("mode") or "cloud")
if mode == "local":
    mode = "local_embedded"
api_url = str(cfg.get("api_url") or "http://localhost:8888")
parsed = urlparse(api_url)
host = parsed.hostname or "127.0.0.1"
port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(mode, host, port)
PY
)

if [[ "$MODE" != "local_external" ]]; then
  printf '[%s] refusing to supervise Hindsight: mode is %s, expected local_external\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" >&2
  exit 78
fi

printf '[%s] exec %s --host %s --port %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HINDSIGHT_API" "$HOST" "$PORT"
exec "$HINDSIGHT_API" --host "$HOST" --port "$PORT"
"""


def render_launchd_plist(plan: HindsightServicePlan) -> str:
    """Render a macOS LaunchAgent plist for the Hindsight wrapper."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{escape(plan.label)}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>{escape(str(plan.wrapper_path))}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{escape(str(plan.hermes_home))}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>StandardOutPath</key>
  <string>{escape(str(plan.stdout_path))}</string>
  <key>StandardErrorPath</key>
  <string>{escape(str(plan.stderr_path))}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HERMES_HOME</key>
    <string>{escape(str(plan.hermes_home))}</string>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
"""


def _require_hindsight(provider: str) -> None:
    if provider != "hindsight":
        raise UnsupportedMemoryService(
            f"memory service lifecycle currently supports only 'hindsight', not {provider!r}"
        )


def _require_macos() -> None:
    if platform.system() != "Darwin":
        raise UnsupportedMemoryService(
            "`hermes memory service` currently installs Hindsight services with macOS launchd only."
        )


def _launchd_domain() -> str:
    """Return the current per-user launchd domain after the macOS gate."""
    _require_macos()
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise UnsupportedMemoryService("launchd service management requires a POSIX user id")
    return f"gui/{getuid()}"


def install_hindsight_launchd_service(plan: HindsightServicePlan, *, force: bool = False) -> None:
    """Write the wrapper/plist and bootstrap the user LaunchAgent."""
    domain = _launchd_domain()
    plan.wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    plan.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    plan.plist_path.parent.mkdir(parents=True, exist_ok=True)
    plan.wrapper_path.write_text(render_launchd_wrapper(plan), encoding="utf-8")
    plan.wrapper_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    plan.plist_path.write_text(render_launchd_plist(plan), encoding="utf-8")
    plan.plist_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    service = f"{domain}/{plan.label}"
    if force:
        subprocess.run(["launchctl", "bootout", service], check=False, capture_output=True, text=True)
    subprocess.run(["plutil", "-lint", str(plan.plist_path)], check=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(plan.plist_path)], check=True)
    subprocess.run(["launchctl", "enable", service], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", service], check=True)


def print_hindsight_service_status(plan: HindsightServicePlan) -> None:
    service = f"{_launchd_domain()}/{plan.label}"
    result = subprocess.run(["launchctl", "print", service], capture_output=True, text=True)
    print(f"\nHindsight service: {plan.label}")
    print(f"  Config:  {plan.config_path}")
    print(f"  URL:     {plan.api_url}")
    print(f"  Plist:   {plan.plist_path}")
    print(f"  Wrapper: {plan.wrapper_path}")
    if result.returncode == 0:
        state = "running" if "state = running" in result.stdout else "registered"
        print(f"  launchd: {state} ✓")
    else:
        print("  launchd: not installed ✗")
        if result.stderr.strip():
            print(f"  detail:  {result.stderr.strip()}")
    print()


def restart_hindsight_launchd_service(plan: HindsightServicePlan) -> None:
    service = f"{_launchd_domain()}/{plan.label}"
    subprocess.run(["launchctl", "kickstart", "-k", service], check=True)
    print(f"\n  ✓ Restarted {plan.label}\n")


def show_hindsight_service_logs(plan: HindsightServicePlan) -> None:
    print(f"\nHindsight service logs")
    print(f"  stdout: {plan.stdout_path}")
    print(f"  stderr: {plan.stderr_path}\n")


def memory_service_command(args) -> None:
    provider = getattr(args, "provider", "hindsight")
    try:
        _require_hindsight(provider)
        plan = build_hindsight_service_plan(
            executable=Path(getattr(args, "executable")) if getattr(args, "executable", None) else None,
            env_file=Path(getattr(args, "env_file")) if getattr(args, "env_file", None) else None,
        )
        command = getattr(args, "service_command", "status")
        if command == "install":
            install_hindsight_launchd_service(plan, force=getattr(args, "force", False))
            print(f"\n  ✓ Installed Hindsight service: {plan.label}")
            print(f"  Plist:   {plan.plist_path}")
            print(f"  Wrapper: {plan.wrapper_path}\n")
        elif command == "restart":
            restart_hindsight_launchd_service(plan)
        elif command == "logs":
            show_hindsight_service_logs(plan)
        else:
            print_hindsight_service_status(plan)
    except UnsupportedMemoryService as exc:
        print(f"\n  Cannot manage memory service: {exc}\n")
