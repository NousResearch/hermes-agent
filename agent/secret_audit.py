"""Where Czesiek's credentials live on disk and whether other users of the computer could read them.

Presence-only: the audit reports file names, permissions and the *names* of secret-looking ``.env``
entries — never a value. ``fix_permissions`` tightens files to owner-only (0600) and folders to 0700.
Moving secrets into the operating system's keychain is a separate step this module does not take.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_FILES = (".env", "google_token.json", "google_client_secret.json", "auth.json")
_DIRS = ("browser-profile", "chrome-debug")
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD)$", re.IGNORECASE)


def _is_posix() -> bool:
    return os.name == "posix"


def _too_open(path: Path) -> bool:
    return _is_posix() and bool(path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def _env_secret_names(env: Path) -> list[str]:
    names = []
    try:
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and value.strip().strip("'\"") and _SECRET_NAME.search(key.strip()):
                names.append(key.strip())
    except OSError:
        pass
    return sorted(set(names))


def audit() -> dict[str, Any]:
    home = get_hermes_home()
    items = []
    for name in _FILES:
        path = home / name
        if path.is_file():
            items.append({"name": name, "kind": "file", "too_open": _too_open(path)})
    for name in _DIRS:
        path = home / name
        if path.is_dir():
            items.append({"name": name, "kind": "folder", "too_open": _too_open(path)})
    return {
        "items": items,
        "env_secrets": _env_secret_names(home / ".env"),
        "permissions_checked": _is_posix(),
        "storage": "file",
        "needs_fix": any(i["too_open"] for i in items),
    }


def fix_permissions() -> dict[str, Any]:
    home = get_hermes_home()
    for item in audit()["items"]:
        if item["too_open"]:
            (home / item["name"]).chmod(0o600 if item["kind"] == "file" else 0o700)
    return audit()
