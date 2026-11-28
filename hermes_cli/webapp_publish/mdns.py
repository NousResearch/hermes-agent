"""mDNS publishing for the Desktop setup flow's local mode.

Avahi is the supported way to answer for ``<name>.local`` on Linux
(systemd-resolved only announces the machine's own hostname). One avahi
constraint shapes this module: the daemon registers ONE name per local
interface address, so ``/etc/avahi/hosts`` aliases and
``avahi-publish-address`` for the host's OWN IP fail with "Local name
collision" (verified on avahi 0.8). The reliable way to give this machine a
specific mDNS identity is the daemon's own ``host-name=`` setting in
``/etc/avahi/avahi-daemon.conf`` — it then publishes that name with every
local address, persistently across reboots.

Writing the conf needs root; the flow degrades to exact instructions when it
cannot, and tells the user the machine's mDNS identity changes.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_AVAHI_CONF = Path("/etc/avahi/avahi-daemon.conf")


def label_of(name: str) -> str:
    """``hermes-desktop.local`` -> ``hermes-desktop`` (avahi wants the label)."""
    return name.strip().lower().removesuffix(".local")


def current_hostname(conf_file: Path) -> Optional[str]:
    """The explicit ``host-name=`` under [server], or None when unset (the
    daemon then uses the system hostname)."""
    try:
        text = conf_file.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    in_server = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_server = stripped.lower() == "[server]"
            continue
        if in_server and not stripped.startswith("#"):
            match = re.match(r"host-name\s*=\s*(\S+)", stripped)
            if match:
                return match.group(1)
    return None


def set_hostname(conf_file: Path, name: str) -> tuple[bool, Optional[Path]]:
    """Idempotently set ``host-name=<label>`` under [server]. Returns
    ``(changed, backup_path)``; raises OSError when the file is unwritable."""
    label = label_of(name)
    text = conf_file.read_text(encoding="utf-8-sig")
    if current_hostname(conf_file) == label:
        return False, None
    backup = conf_file.with_name(conf_file.name + ".bak-desktop")
    shutil.copy2(conf_file, backup)
    lines = text.splitlines()
    out, in_server, replaced = [], False, False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            # Leaving [server]: insert the setting before the next section
            # when it had no host-name line of its own.
            if in_server and not replaced:
                out.append(f"host-name={label}")
                replaced = True
            in_server = stripped.lower() == "[server]"
            out.append(line)
            continue
        if in_server and re.match(r"\s*#?\s*host-name\s*=", line):
            if not replaced:
                out.append(f"host-name={label}")
                replaced = True
            continue  # drop duplicates/commented originals
        out.append(line)
    if in_server and not replaced:
        out.append(f"host-name={label}")
        replaced = True
    if not replaced:  # no [server] section at all (defensive)
        out += ["", "[server]", f"host-name={label}"]
    conf_file.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True, backup


def avahi_installed() -> bool:
    return shutil.which("avahi-publish-address") is not None


def restart_avahi(*, run: Optional[Callable] = None) -> bool:
    runner = run or subprocess.run
    try:
        result = runner(["systemctl", "restart", "avahi-daemon"],
                        capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("avahi restart failed: %s", exc)
        return False


def resolves(name: str, expected_ip: str, *, run: Optional[Callable] = None) -> bool:
    """Best-effort live check: avahi-resolve-host-name -4 <name> == expected IP."""
    if shutil.which("avahi-resolve-host-name") is None:
        return False
    runner = run or subprocess.run
    try:
        result = runner(["avahi-resolve-host-name", "-4", name],
                        capture_output=True, text=True, timeout=10)
        return expected_ip in (result.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return False


def publish_local_name(name: str, ip: str, *, conf_file: Path = DEFAULT_AVAHI_CONF,
                       run: Optional[Callable] = None) -> dict:
    """Publish ``name`` -> this machine's addresses via the avahi host-name.
    Returns ``published``, ``changed``, ``previous`` (the old explicit
    host-name, if any), ``reason`` when not published, and ``instructions``
    for a sudo fallback. Never raises for permission problems."""
    if not avahi_installed():
        return {
            "published": False, "changed": False, "previous": None,
            "reason": "avahi-utils is not installed.",
            "instructions": [
                "sudo apt-get install -y avahi-daemon avahi-utils",
                f"sudo sed -i '/^\\[server\\]/a host-name={label_of(name)}' {conf_file}",
                "sudo systemctl restart avahi-daemon",
            ],
        }
    previous = current_hostname(conf_file)
    try:
        changed, backup = set_hostname(conf_file, name)
    except OSError as exc:
        return {
            "published": False, "changed": False, "previous": previous, "backup": None,
            "reason": f"Cannot write {conf_file} ({exc.__class__.__name__}).",
            "instructions": [
                f"sudo sed -i '/^\\[server\\]/a host-name={label_of(name)}' {conf_file}",
                "sudo systemctl restart avahi-daemon",
            ],
        }
    restarted = restart_avahi(run=run)
    return {
        "published": restarted, "changed": changed, "previous": previous,
        "backup": str(backup) if backup else None,
        "reason": None if restarted else "avahi-daemon could not be restarted.",
        "instructions": [] if restarted else ["sudo systemctl restart avahi-daemon"],
    }