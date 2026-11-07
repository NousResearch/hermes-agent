"""Per-user OpenBSD gateway boot registration via crontab.

OpenBSD has no user service manager. ``@reboot`` starts the invoking user's
gateway. A failed ``crontab -l`` that is not "no crontab" must not be written
back, or the user's other jobs disappear.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Callable

CRON_TAG = "hermes-gateway"
CRON_PATH = "/usr/local/bin:/usr/bin:/bin"


class CrontabReadError(RuntimeError):
    """``crontab -l`` failed for a reason other than a missing crontab."""


def cron_marker(hermes_home: str, profile: str) -> str:
    """Return the comment that identifies one user's gateway line."""
    return f"# {CRON_TAG} home={hermes_home} profile={profile}"


def system_boot_command(username: str, hermes_home: str, command: list[str]) -> list[str]:
    """Boot the gateway as ``username``, not as root.

    OpenBSD ``su`` has no GNU ``-c``. The command is a shell argument after the
    login name: ``su -s /bin/sh user -c '...'``. ``-s /bin/sh`` avoids the
    user's login shell.
    """
    inner = "export HERMES_HOME=" + shlex.quote(hermes_home) + "; exec " + " ".join(shlex.quote(part) for part in command)
    return ["/usr/bin/su", "-s", "/bin/sh", username, "-c", inner]


def missing_crontab(returncode: int, stderr: str) -> bool:
    """True only for crontab's own "no crontab for this user" failure."""
    return returncode != 0 and "no crontab for" in stderr


def render_cron_line(command: list[str], hermes_home: str, profile: str) -> str:
    """One ``@reboot`` line. The command is absolute; cron's default PATH is not used."""
    cmd_dir = str(Path(command[0]).parent) if command else ""
    path = CRON_PATH if not cmd_dir or cmd_dir in CRON_PATH.split(":") else f"{CRON_PATH}:{cmd_dir}"
    quoted = " ".join(shlex.quote(part) for part in command)
    return f"@reboot PATH={path} {quoted} >> {hermes_home}/logs/gateway.log 2>&1 {cron_marker(hermes_home, profile)}"


def merge_crontab(existing: str, line: str, marker: str) -> str:
    """Replace any line carrying ``marker`` and keep every other line."""
    kept = [item for item in existing.splitlines() if marker not in item]
    kept.append(line)
    return "\n".join(kept) + "\n"


def remove_marked_lines(existing: str, marker: str) -> str:
    """Drop lines carrying ``marker``. Other lines stay, including their order."""
    kept = [item for item in existing.splitlines() if marker not in item]
    if not kept:
        return ""
    return "\n".join(kept) + ("\n" if existing.endswith("\n") or kept else "")


def _read_crontab(run: Callable) -> str:
    result = run(["crontab", "-l"], capture_output=True, text=True)
    stderr = result.stderr or ""
    if missing_crontab(result.returncode, stderr):
        return ""
    if result.returncode != 0:
        raise CrontabReadError(stderr.strip() or "crontab -l failed")
    return result.stdout or ""


def _write_crontab(text: str, run: Callable) -> None:
    result = run(["crontab", "-"], input=text, text=True)
    if getattr(result, "returncode", 0) != 0:
        stderr = getattr(result, "stderr", "") or ""
        raise CrontabReadError(stderr.strip() or "crontab write failed")


def install_boot_line(command: list[str], hermes_home: str, profile: str, run: Callable = subprocess.run) -> None:
    """Insert or replace this profile's ``@reboot`` line. Does not write on a bad read."""
    existing = _read_crontab(run)
    marker = cron_marker(hermes_home, profile)
    _write_crontab(merge_crontab(existing, render_cron_line(command, hermes_home, profile), marker), run)


def uninstall_boot_line(hermes_home: str, profile: str, run: Callable = subprocess.run) -> None:
    """Remove this profile's line. Does not write on a bad read."""
    existing = _read_crontab(run)
    _write_crontab(remove_marked_lines(existing, cron_marker(hermes_home, profile)), run)
