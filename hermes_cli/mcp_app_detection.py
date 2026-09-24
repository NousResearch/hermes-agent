"""Read-only, backend-local application signals for reviewed MCP catalog labels.

This module never runs apps, loads a terminal backend or contacts an MCP server.
Only curated matches may leave it; inventory names and paths remain private.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import time

_MAX_APPLICATIONS = 16
_APP_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+-]{0,79}")


def validate_applications(labels: object) -> list[str]:
    """Accept bounded display labels/aliases, never paths, commands or regexes."""
    if not isinstance(labels, list) or len(labels) > _MAX_APPLICATIONS:
        raise ValueError("suggest.applications must be a list of at most 16 app labels")
    for label in labels:
        if (not isinstance(label, str) or not _APP_LABEL.fullmatch(label)
                or label != label.strip() or ".." in label or " --" in label):
            raise ValueError("suggest.applications must contain safe app labels (1–80 characters)")
    return list(labels)


def _application_roots() -> list[Path]:
    # OS home, NOT HERMES_HOME: profiles do not change the backend machine's apps.
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/Applications"), Path("/System/Applications"), home / "Applications"]
    if sys.platform == "linux":
        return [Path("/usr/share/applications"), Path("/usr/local/share/applications"),
                home / ".local/share/applications"]
    if sys.platform == "win32":
        return [Path(os.environ.get("ProgramFiles", "C:/Program Files")),
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
                Path(os.environ.get("LOCALAPPDATA", str(home / "AppData/Local"))) / "Programs"]
    return []


class _DiscoveryLimit(Exception):
    """A partial scan cannot claim absence."""


class _Scan:
    def __init__(self, labels: set[str]):
        self.patterns = {label: re.compile(r"(?<!\w)" + re.escape(label) + r"(?!\w)", re.I)
                         for label in labels}
        self.matched: set[str] = set()
        self.readable = False
        self.unavailable = False
        self.remaining = 4096
        self.directories = 128
        self.deadline = time.monotonic() + 2.0

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0 or time.monotonic() > self.deadline:
            raise _DiscoveryLimit

    def record(self, name: str):
        # Never retain the inventory, only which reviewed labels matched it.
        self.matched.update(label for label, pattern in self.patterns.items() if pattern.search(name))

    def directory(self, root: Path, depth: int = 0):
        self.directories -= 1
        if self.directories < 0:
            raise _DiscoveryLimit
        try:
            with os.scandir(root) as children:
                self.readable = True
                for child in children:
                    self.tick()
                    if child.is_symlink():
                        continue
                    is_dir = child.is_dir(follow_symlinks=False)
                    if sys.platform == "darwin" and is_dir and child.name.lower().endswith(".app"):
                        self.record(child.name[:-4])
                        continue  # Never descend into bundles or run their executables.
                    if sys.platform == "win32" and is_dir:
                        self.record(child.name)
                    if sys.platform == "linux" and child.name.endswith(".desktop") and child.is_file(follow_symlinks=False):
                        self.desktop(Path(child.path))
                    if is_dir and depth < 2:
                        self.directory(Path(child.path), depth + 1)
        except FileNotFoundError:
            pass  # An optional standard root need not exist.
        except OSError:
            self.unavailable = True

    def desktop(self, path: Path):
        self.record(path.stem)
        # Name is data; deliberately ignore Exec, TryExec, URLs and localized commands.
        with path.open(encoding="utf-8", errors="replace") as stream:
            text = stream.read(32769)
        if len(text) > 32768:
            self.unavailable = True
            return
        in_entry = False
        for line in text.splitlines():
            if line.startswith("["):
                in_entry = line.strip() == "[Desktop Entry]"
            elif in_entry and line.startswith("Name="):
                self.record(line[5:])

    def path_candidates(self):
        # Exact single-token candidates only. No multiword-to-command heuristics.
        candidates = {label: tuple(dict.fromkeys((label, label.lower()))) for label in self.patterns
                      if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label)}
        parts = os.environ.get("PATH", "").split(os.pathsep)
        if len(parts) > 64:
            self.unavailable = True
        for part in dict.fromkeys(parts[:64]):
            if not part or len(part) > 4096 or not Path(part).is_absolute():
                continue  # Never search the working directory.
            try:
                self.tick()
                with os.scandir(part):
                    self.readable = True
                for label, spellings in candidates.items():
                    self.tick()
                    # Passing an absolute candidate also prevents Windows which() from
                    # silently prepending the current directory to an explicit PATH.
                    if any(shutil.which(str(Path(part) / candidate)) for candidate in spellings):
                        self.matched.add(label)
            except FileNotFoundError:
                continue
            except OSError:
                self.unavailable = True


def discover_catalog_apps(applications: dict[str, list[str]]) -> dict:
    """Bounded backend-native signals, not proof an integration is configured.

    ``ok`` means the supported scan completed, not an exhaustive OS inventory.
    A missing permission, exhausted budget or unsupported OS is ``unavailable``;
    positive matches still stand, but empty arrays then mean unknown.
    """
    criteria = {name: validate_applications(labels) for name, labels in applications.items()}
    labels = {label for values in criteria.values() for label in values}
    supported = sys.platform in {"darwin", "linux", "win32"} and len(labels) <= 256
    scan = _Scan(labels if supported else set())
    if supported:
        try:
            for root in _application_roots():
                scan.directory(root)
            scan.path_candidates()
        except (OSError, RuntimeError, _DiscoveryLimit):
            scan.unavailable = True
    else:
        scan.unavailable = True
    return {
        "matches": {name: list(dict.fromkeys(label for label in labels if label in scan.matched))
                    for name, labels in criteria.items()},
        "discovery": {"scope": "backend", "status": "unavailable" if scan.unavailable or not scan.readable else "ok",
                      "platform": sys.platform},
    }
