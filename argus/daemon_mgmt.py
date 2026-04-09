"""Daemon management: PID files, launchd integration, lifecycle."""

import os
import sys
import json
import time
import subprocess
import logging
import shutil as _shutil
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("argus.daemon")

# === HERMES INTEGRATION ===
# Import Hermes constants with same fallback pattern as argus.py
try:
    from hermes_constants import get_hermes_home
    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

_ARGUS_HOME = Path(os.path.expanduser("~/hermes"))


def _hermes_path(*parts: str) -> Path:
    """Build a path under HERMES_HOME (~/.hermes)."""
    return _HERMES_HOME.joinpath(*parts)


def _argus_path(*parts: str) -> Path:
    """Build a path under ARGUS_HOME (~/hermes)."""
    return _ARGUS_HOME.joinpath(*parts)


# === PID FILE ===
_ARGUS_PID_PATH = _argus_path("data", "watcher", "argus.pid")
_ARGUS_KIND = "argus-watcher"


def _get_argus_pid_path() -> Path:
    """Path to the ARGUS PID file."""
    return _ARGUS_PID_PATH


def _build_argus_pid_record() -> dict:
    """Build PID record for argus.pid."""
    return {
        "pid": os.getpid(),
        "kind": _ARGUS_KIND,
        "argv": list(sys.argv),
        "start_time": time.time(),
    }


def write_argus_pid_file() -> None:
    """Write ARGUS PID file."""
    path = _get_argus_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_argus_pid_record()))


def remove_argus_pid_file() -> None:
    """Remove ARGUS PID file."""
    try:
        _get_argus_pid_path().unlink(missing_ok=True)
    except Exception:
        pass


def _read_argus_pid_record() -> Optional[dict]:
    """Read ARGUS PID file, return dict or None."""
    path = _get_argus_pid_path()
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


def get_argus_running_pid() -> Optional[int]:
    """Return PID of running ARGUS instance, or None."""
    record = _read_argus_pid_record()
    if not record:
        remove_argus_pid_file()
        return None
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        remove_argus_pid_file()
        return None

    # Check if process is alive
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        remove_argus_pid_file()
        return None


def is_argus_running() -> bool:
    """Check if ARGUS daemon is currently running."""
    return get_argus_running_pid() is not None


# === LAUNCHD ===
_ARGUS_LAUNCHD_LABEL = "com.hermes.argus"
_ARGUS_SCRIPT = str(_argus_path("scripts", "watcher", "argus.py"))


def get_argus_launchd_label() -> str:
    """Return the launchd service label."""
    return _ARGUS_LAUNCHD_LABEL


def _hermes_home_plist_dir() -> Path:
    """Return ~/Library/LaunchAgents (macOS-specific)."""
    return Path.home() / "Library" / "LaunchAgents"


def get_argus_launchd_plist_path() -> Path:
    """Return the launchd plist path."""
    return _hermes_home_plist_dir() / f"{_ARGUS_LAUNCHD_LABEL}.plist"


def generate_argus_launchd_plist() -> str:
    """Generate launchd plist XML with full PATH, HERMES_HOME, KeepAlive."""
    label = get_argus_launchd_label()
    script = _ARGUS_SCRIPT
    log_dir = str(_argus_path("logs", "argus"))
    hermes_home = str(_HERMES_HOME)

    # Build PATH
    venv_bin = str(_hermes_path("hermes-agent", "venv", "bin"))
    priority_dirs = [venv_bin] if os.path.isdir(venv_bin) else []

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        hermes_dir = str(Path(hermes_bin).resolve().parent)
        if hermes_dir not in priority_dirs:
            priority_dirs.append(hermes_dir)

    sane_path = ":".join(
        dict.fromkeys(
            priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p]
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


def argus_launchd_install() -> bool:
    """Install ARGUS as launchd service."""
    plist_path = get_argus_launchd_plist_path()

    # Write plist
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(generate_argus_launchd_plist())
    logger.info("ARGUS plist written to: %s", plist_path)

    # Bootstrap via launchctl
    try:
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
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


def argus_launchd_uninstall() -> bool:
    """Uninstall ARGUS launchd service."""
    label = get_argus_launchd_label()
    plist_path = get_argus_launchd_plist_path()

    # Bootout
    try:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        pass

    # Remove plist
    plist_path.unlink(missing_ok=True)
    logger.info("ARGUS launchd service uninstalled")
    return True


def argus_launchd_status() -> dict:
    """Check ARGUS launchd service status."""
    label = get_argus_launchd_label()
    plist_path = get_argus_launchd_plist_path()

    return {
        "label": label,
        "plist_exists": plist_path.exists(),
        "plist_path": str(plist_path),
        "pid_file_exists": _get_argus_pid_path().exists(),
        "running_pid": get_argus_running_pid(),
        "is_running": is_argus_running(),
    }
