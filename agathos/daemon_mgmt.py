"""Daemon management: PID files, launchd integration, lifecycle.

Platform Support:
- macOS: Full launchd service management via launchctl
- Linux: Systemd user service support (planned)
- Windows: Service support via SC or pywin32 (planned)

Non-macOS platforms currently support PID file operations but
service management (install/uninstall/status) is limited.
"""

import os
import sys
import json
import time
import subprocess
import logging
import shutil as _shutil
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("agathos.daemon")

# === PLATFORM DETECTION ===
_IS_WINDOWS = os.name == 'nt' or sys.platform == 'win32'
_IS_MACOS = sys.platform == 'darwin'
_IS_LINUX = sys.platform.startswith('linux')

# === HERMES INTEGRATION ===
# Import Hermes constants with same fallback pattern as argus.py
try:
    from hermes_constants import get_hermes_home
    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

_AGATHOS_HOME = Path(os.path.expanduser("~/hermes"))


def _hermes_path(*parts: str) -> Path:
    """Build a path under HERMES_HOME (~/.hermes)."""
    return _HERMES_HOME.joinpath(*parts)


def _agathos_path(*parts: str) -> Path:
    """Build a path under ARGUS_HOME (~/hermes)."""
    return _AGATHOS_HOME.joinpath(*parts)


# === PID FILE ===
_ARGUS_PID_PATH = _agathos_path("data", "watcher", "agathos.pid")
_ARGUS_KIND = "argus-watcher"


def _get_agathos_pid_path() -> Path:
    """Path to the ARGUS PID file."""
    return _ARGUS_PID_PATH


def _build_agathos_pid_record() -> dict:
    """Build PID record for agathos.pid."""
    return {
        "pid": os.getpid(),
        "kind": _ARGUS_KIND,
        "argv": list(sys.argv),
        "start_time": time.time(),
    }


def write_agathos_pid_file() -> None:
    """Write ARGUS PID file."""
    path = _get_agathos_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_agathos_pid_record()))


def remove_agathos_pid_file() -> None:
    """Remove ARGUS PID file."""
    try:
        _get_agathos_pid_path().unlink(missing_ok=True)
    except Exception:
        pass


def _read_agathos_pid_record() -> Optional[dict]:
    """Read ARGUS PID file, return dict or None."""
    path = _get_agathos_pid_path()
    if not path.exists():
        return None
    raw = path.read_text().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {"pid": int(raw)}
        except ValueError:
            return None


def get_agathos_running_pid() -> Optional[int]:
    """Return PID of running ARGUS instance, or None."""
    record = _read_agathos_pid_record()
    if not record:
        remove_agathos_pid_file()
        return None
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        remove_agathos_pid_file()
        return None

    # Check if process is alive
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        remove_agathos_pid_file()
        return None


def is_agathos_running() -> bool:
    """Check if ARGUS daemon is currently running."""
    return get_agathos_running_pid() is not None


# === LAUNCHD ===
_ARGUS_LAUNCHD_LABEL = "com.hermes.agathos"
_ARGUS_SCRIPT = str(_agathos_path("scripts", "watcher", "argus.py"))


def get_agathos_launchd_label() -> str:
    """Return the launchd service label."""
    return _ARGUS_LAUNCHD_LABEL


def _get_service_directory() -> Path:
    """Get platform-specific service directory.

    Returns:
        Path to service configuration directory:
        - macOS: ~/Library/LaunchAgents (launchd)
        - Linux: ~/.config/systemd/user (systemd)
        - Windows: ~/AppData/Roaming/Agathos (user services)
        - Other: ~/.agathos (fallback)
    """
    if _IS_MACOS:
        return Path.home() / "Library" / "LaunchAgents"
    elif _IS_LINUX:
        return Path.home() / ".config" / "systemd" / "user"
    elif _IS_WINDOWS:
        return Path.home() / "AppData" / "Roaming" / "Agathos"
    return Path.home() / ".agathos"


def _hermes_home_plist_dir() -> Path:
    """Return service directory (macOS: ~/Library/LaunchAgents)."""
    return _get_service_directory()


def get_agathos_launchd_plist_path() -> Path:
    """Return the launchd plist path."""
    if _IS_MACOS:
        return _get_service_directory() / f"{_ARGUS_LAUNCHD_LABEL}.plist"
    elif _IS_LINUX:
        return _get_service_directory() / f"{_ARGUS_LAUNCHD_LABEL}.service"
    elif _IS_WINDOWS:
        return _get_service_directory() / f"{_ARGUS_LAUNCHD_LABEL}.xml"
    return _get_service_directory() / f"{_ARGUS_LAUNCHD_LABEL}.conf"


def generate_agathos_launchd_plist() -> str:
    """Generate launchd plist XML with full PATH, HERMES_HOME, KeepAlive."""
    label = get_agathos_launchd_label()
    script = _ARGUS_SCRIPT
    log_dir = str(_agathos_path("logs", "agathos"))
    hermes_home = str(_HERMES_HOME)

    # Build PATH
    venv_bin = str(_hermes_path("hermes-agent", "venv", "bin"))
    priority_dirs = [venv_bin] if os.path.isdir(venv_bin) else []

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        hermes_dir = str(Path(hermes_bin).resolve().parent)
        if hermes_dir not in priority_dirs:
            priority_dirs.append(hermes_dir)

    # Use os.pathsep for consistency (on macOS this is ':', which is what launchd expects)
    sane_path = os.pathsep.join(
        dict.fromkeys(
            priority_dirs + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        )
    )

    # Detect python
    python = sys.executable or "/usr/bin/python3"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{Path(script).parent}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{sane_path}</string>
        <key>HERMES_HOME</key>
        <string>{hermes_home}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_dir}/argus.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/argus.stderr.log</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>"""


def _is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux."""
    if not _IS_LINUX:
        return False
    # Check for WSL-specific indicators
    if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'):
        return True
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            return 'microsoft' in version or 'wsl' in version
    except Exception:
        pass
    return False


def generate_systemd_service() -> str:
    """Generate systemd user service file for Linux."""
    label = get_agathos_launchd_label()
    script = _ARGUS_SCRIPT
    log_dir = str(_agathos_path("logs", "agathos"))
    hermes_home = str(_HERMES_HOME)

    # Build PATH
    from .venv_utils import get_agathos_venv_paths
    agathos_paths = get_agathos_venv_paths()
    current_path = os.environ.get("PATH", "")
    full_path = os.pathsep.join(dict.fromkeys(agathos_paths + current_path.split(os.pathsep)))

    # Get Python
    python = sys.executable or "/usr/bin/python3"

    return f"""[Unit]
Description=Agathos - Agent Guardian & Health Oversight System
Documentation=https://github.com/NousResearch/hermes-agent
After=network.target

[Service]
Type=simple
ExecStart={python} {script}
WorkingDirectory={Path(script).parent}
Restart=on-failure
RestartSec=10

# Environment
Environment="PATH={full_path}"
Environment="HERMES_HOME={hermes_home}"
Environment="PYTHONUNBUFFERED=1"

# Logging
StandardOutput=append:{log_dir}/agathos.stdout.log
StandardError=append:{log_dir}/agathos.stderr.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=home
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

[Install]
WantedBy=default.target
"""


def agathos_launchd_install() -> bool:
    """Install ARGUS as system service.

    Platform support:
    - macOS: Uses launchd/launchctl
    - Linux: Uses systemd user service
    - Windows: Uses Service Control Manager (planned)
    """
    service_path = get_agathos_launchd_plist_path()

    if _IS_MACOS:
        # Write plist
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(generate_agathos_launchd_plist())
        logger.info("ARGUS plist written to: %s", service_path)

        # Bootstrap via launchctl
        try:
            subprocess.run(
                ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(service_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("ARGUS launchd service bootstrapped")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Failed to bootstrap ARGUS: %s", e.stderr, exc_info=True)
            return False
        except FileNotFoundError:
            logger.error("launchctl not found - is this macOS?")
            return False

    elif _IS_LINUX:
        # Write systemd user service
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(generate_systemd_service())
        logger.info("ARGUS systemd service written to: %s", service_path)

        # Enable and start via systemctl
        try:
            # Enable the service
            subprocess.run(
                ["systemctl", "--user", "enable", str(service_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("ARGUS systemd service enabled")

            # Start the service
            subprocess.run(
                ["systemctl", "--user", "start", _ARGUS_LAUNCHD_LABEL],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("ARGUS systemd service started")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Failed to enable/start ARGUS: %s", e.stderr, exc_info=True)
            logger.info("You may need to run: systemctl --user daemon-reload")
            return False
        except FileNotFoundError:
            logger.error("systemctl not found - is systemd installed?")
            return False

    elif _IS_WINDOWS:
        logger.warning(
            "Windows service installation not yet implemented. "
            "You can still run Agathos manually."
        )
        return False

    else:
        logger.warning(
            "Service installation on %s is not supported. "
            "Supported platforms: macOS (darwin), Linux. "
            "You can still run Agathos manually or via cron.",
            sys.platform
        )
        return False


def agathos_launchd_uninstall() -> bool:
    """Uninstall ARGUS system service.

    Platform support:
    - macOS: Uses launchd/launchctl
    - Linux: Uses systemd user service
    - Windows: Uses Service Control Manager (planned)
    """
    label = get_agathos_launchd_label()
    service_path = get_agathos_launchd_plist_path()

    if _IS_MACOS:
        # Bootout via launchctl
        try:
            subprocess.run(
                ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            pass

    elif _IS_LINUX:
        # Stop and disable via systemctl
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", label],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            pass

        try:
            subprocess.run(
                ["systemctl", "--user", "disable", label],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.debug("ARGUS systemd service disabled")
        except Exception:
            pass

    elif _IS_WINDOWS:
        logger.debug("Windows service uninstall not yet implemented")

    # Remove service file
    service_path.unlink(missing_ok=True)
    logger.info("ARGUS service uninstalled")
    return True


def agathos_service_status() -> dict:
    """Check ARGUS service status with platform-specific details."""
    label = get_agathos_launchd_label()
    service_path = get_agathos_launchd_plist_path()
    running_pid = get_agathos_running_pid()

    status = {
        "label": label,
        "service_path": str(service_path),
        "service_exists": service_path.exists(),
        "pid_file_exists": _get_agathos_pid_path().exists(),
        "running_pid": running_pid,
        "is_running": is_agathos_running(),
        "platform": sys.platform,
        "is_wsl": _is_wsl(),
    }

    # Add platform-specific status
    if _IS_MACOS:
        status["service_type"] = "launchd"
        # Check launchctl status
        try:
            result = subprocess.run(
                ["launchctl", "list", label],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status["service_loaded"] = result.returncode == 0
        except Exception:
            status["service_loaded"] = False

    elif _IS_LINUX:
        status["service_type"] = "systemd"
        status["is_wsl"] = _is_wsl()
        # Check systemd status
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", label],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status["service_active"] = result.returncode == 0
            status["systemd_status"] = result.stdout.strip()
        except Exception:
            status["service_active"] = False
            status["systemd_status"] = "unknown"

    elif _IS_WINDOWS:
        status["service_type"] = "windows"
        status["service_loaded"] = False  # Not implemented

    else:
        status["service_type"] = "unknown"
        status["service_loaded"] = False

    return status


# Backward compatibility alias
agathos_launchd_status = agathos_service_status
