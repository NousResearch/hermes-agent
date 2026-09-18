"""Read-only verification at the Windows Desktop handoff receipt boundary."""
import json
from pathlib import Path, PurePosixPath
import re
import struct

from hermes_cli.main_desktop import (
    _HTML_TAG_WITH_URL,
    _MODULE_TAG,
    _desktop_build_needed,
    _desktop_exe_integrity_error,
    _desktop_packaged_executable,
)


def _verify_packaged_entry(resources: Path) -> None:
    """Read ASAR's Pickle header and the entry declared by packaged package.json.

    dist/** is unpacked by electron-builder. No Node install or application
    launch is needed at this boundary; this is not a full dependency audit.
    """
    archive = resources / "app.asar"
    try:
        with archive.open("rb") as stream:
            size, header_size, payload_size, json_size = struct.unpack("<4I", stream.read(16))
            if (size != 4 or header_size != payload_size + 4
                    or payload_size != 4 + ((json_size + 3) // 4) * 4
                    or not 0 < json_size <= 64 * 1024 * 1024
                    or 8 + header_size > archive.stat().st_size):
                raise ValueError("invalid ASAR header")
            header = json.loads(stream.read(json_size))

            def read_member(name: str) -> bytes:
                path = PurePosixPath(name)
                if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("invalid ASAR entry path")
                node = header
                for part in path.parts:
                    node = node["files"][part]
                length = node["size"]
                if not isinstance(length, int) or length <= 0:
                    raise ValueError(f"empty ASAR entry: {name}")
                if node.get("unpacked"):
                    data = (resources / "app.asar.unpacked" / path).read_bytes()
                else:
                    offset = int(node["offset"])
                    if offset < 0 or 8 + header_size + offset + length > archive.stat().st_size:
                        raise ValueError(f"truncated ASAR entry: {name}")
                    stream.seek(8 + header_size + offset)
                    data = stream.read(length)
                if len(data) != length or not data.strip():
                    raise ValueError(f"incomplete ASAR entry: {name}")
                return data

            package = json.loads(read_member("package.json"))
            read_member(package["main"]).decode("utf-8")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        raise RuntimeError(f"The updated Desktop archive or main entry is invalid: {exc}") from exc

    index = resources / "app.asar.unpacked" / "dist" / "index.html"
    try:
        html = index.read_text(encoding="utf-8")
        if not any(_MODULE_TAG.search(match.group(0))
                   and match.group(0).lower().startswith("<script")
                   and not re.match(r"^[a-z]+:|^//", match.group(1), re.IGNORECASE)
                   for match in _HTML_TAG_WITH_URL.finditer(html)):
            raise ValueError("renderer has no local module entry")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"The updated Desktop renderer entry is invalid: {exc}") from exc


def checkout_root() -> Path:
    """The checkout this module was imported from: the only root the receipt may describe."""
    return Path(__file__).resolve().parent.parent


def verify_windows_desktop_update(project_root: Path | None = None) -> None:
    """Raise when a zero-exit updater left an incomplete or stale packaged app.

    The root defaults to the imported checkout, never the caller's cwd: the hand-off
    is spawned from HERMES_HOME by the pre-update Desktop, and a cwd-derived root
    reported a healthy install as "Desktop executable is missing" (Sep 2026).
    """
    if project_root is None:
        project_root = checkout_root()
    desktop = project_root / "apps" / "desktop"
    executable = _desktop_packaged_executable(desktop)
    if executable is None:
        raise RuntimeError("The updated Desktop executable is missing")
    error = _desktop_exe_integrity_error(executable)
    if error:
        raise RuntimeError(f"The updated Desktop executable is invalid: {error}")
    _verify_packaged_entry(executable.parent / "resources")
    if _desktop_build_needed(desktop, project_root, source_mode=False):
        raise RuntimeError("The updated Desktop build is stale, unstamped, or incomplete")
