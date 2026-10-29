"""Reading the systemd unit (or launchd plist) that describes a gateway service.

Service definitions live outside ``~/.hermes`` (``/etc/systemd/system``, ``~/.config/systemd/user``),
so they are not always readable by the user running the command: a system-scope unit installed by
root is commonly mode ``0640 root:root``. Every read here therefore returns ``None`` rather than
raising ``PermissionError``, which lets ``hermes gateway status`` and ``hermes gateway migrate``
report "unknown" instead of dying with a traceback.
"""

from __future__ import annotations

from pathlib import Path


def read_unit_file_text(unit_path: Path) -> str | None:
    """Text of the unit file at *unit_path*; None when it is missing or unreadable.

    A system-scope unit installed by root is not always world-readable (``0640 root:root``
    is what a root umask of 027 produces), so unprivileged commands must treat an
    unreadable unit as unknown instead of dying with ``PermissionError``.
    """
    try:
        return unit_path.read_text(encoding="utf-8-sig")
    except OSError:
        return None


def unit_environment_value(unit_path: Path, name: str) -> str | None:
    """Value of one ``Environment="NAME=…"`` directive in the unit file at *unit_path*, with
    systemd's ``\\"``/``\\\\``/``%%`` quoting undone; None when the file or the key is absent."""
    text = read_unit_file_text(unit_path)
    if text is None:
        return None
    for line in text.splitlines():
        body = line.strip()
        if not body.startswith("Environment="):
            continue
        body = body[len("Environment=") :].strip()
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1].replace('\\"', '"').replace("\\\\", "\\").replace("%%", "%")
        if body.startswith(f"{name}="):
            return body.split("=", 1)[1].strip() or None
    return None


def unit_home_pinned_by_unit(unit_path: Path) -> str | None:
    """``HERMES_HOME`` pinned by the unit file at *unit_path*, or None when absent/unreadable."""
    return unit_environment_value(unit_path, "HERMES_HOME")


def read_systemd_user_from_unit(unit_path: Path) -> str | None:
    text = read_unit_file_text(unit_path)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("User="):
            return line.split("=", 1)[1].strip() or None
    return None
